#!/usr/bin/env python3
"""Audit CA-ECAD credit magnitude from frozen, no-update grouped rollouts.

This tool deliberately reuses :func:`compute_ca_ecad_credits` rather than
reimplementing the V11.1 formula.  It answers whether acquisition credit is
large enough to plausibly influence PPO, without changing model weights,
advantages, rewards, or optimizer state.
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from search_r1.llm_agent.ca_ecad import (  # noqa: E402
    canonical_document_id,
    compute_ca_ecad_credits,
    write_json,
    write_jsonl,
)


def _read_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'invalid JSON at {path}:{line_number}') from exc
    return rows


def _nearest(values, quantile):
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    return ordered[int(round((len(ordered) - 1) * quantile))]


def _absolute_summary(values):
    values = [abs(float(value)) for value in values]
    if not values:
        return {
            'count': 0,
            'mean_abs': 0.0,
            'std_abs': 0.0,
            'p50_abs': 0.0,
            'p90_abs': 0.0,
            'max_abs': 0.0,
            'rate_abs_gt_0_01': 0.0,
            'rate_abs_gt_0_05': 0.0,
        }
    mean = sum(values) / len(values)
    return {
        'count': len(values),
        'mean_abs': mean,
        'std_abs': math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)),
        'p50_abs': _nearest(values, 0.50),
        'p90_abs': _nearest(values, 0.90),
        'max_abs': max(values),
        'rate_abs_gt_0_01': sum(value > 0.01 for value in values) / len(values),
        'rate_abs_gt_0_05': sum(value > 0.05 for value in values) / len(values),
    }


def _segment_token_count(spans, segment_id):
    count = 0
    for span in spans.get(str(segment_id), []):
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError(f'invalid policy span for segment {segment_id}: {span!r}')
        start, end = int(span[0]), int(span[1])
        if start < 0 or end < start:
            raise ValueError(f'invalid policy span bounds for segment {segment_id}: {span!r}')
        count += end - start
    if count <= 0:
        raise ValueError(f'CA-ECAD segment {segment_id} has no policy tokens')
    return count


def _mode_diversity(source_rows):
    """Summarize |{Top1(z_i^k)}| inside each prompt group at every turn."""

    groups = defaultdict(list)
    for row in source_rows:
        groups[str(row['group_uid'])].append(row)

    values_by_turn = defaultdict(list)
    all_same_by_turn = defaultdict(int)
    groups_by_turn = defaultdict(int)
    for rows in groups.values():
        max_turns = max((len(row['ordered_search_document_ids']) for row in rows), default=0)
        for turn_index in range(max_turns):
            modes = set()
            for row in rows:
                history = row['ordered_search_document_ids']
                if len(history) <= turn_index or not history[turn_index]:
                    continue
                top1 = canonical_document_id(history[turn_index][0])
                if top1 is not None:
                    modes.add(top1)
            if not modes:
                continue
            turn = turn_index + 1
            count = len(modes)
            values_by_turn[turn].append(count)
            groups_by_turn[turn] += 1
            all_same_by_turn[turn] += int(count == 1)

    result = {}
    for turn, values in sorted(values_by_turn.items()):
        result[f'k_{turn}'] = {
            'prompt_group_count': len(values),
            'distinct_top1_mode_mean': sum(values) / len(values),
            'distinct_top1_mode_p50': _nearest(values, 0.50),
            'distinct_top1_mode_p90': _nearest(values, 0.90),
            'all_same_top1_mode_rate': all_same_by_turn[turn] / groups_by_turn[turn],
            'multiple_top1_mode_rate': sum(value > 1 for value in values) / len(values),
        }
    return result


def _load_state(path, alpha, eta, kappa):
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if payload.get('version') != 1:
        raise ValueError('expected a version=1 CA-ECAD Phase-3 state file')
    prior = float(payload.get('success_prior'))
    if not math.isfinite(prior) or not 0.0 <= prior <= 1.0:
        raise ValueError('CA-ECAD state success_prior must be finite and in [0, 1]')
    expected = {'alpha': alpha, 'eta': eta, 'kappa': kappa}
    actual = payload.get('hyperparameters', {})
    for key, value in expected.items():
        if float(actual.get(key, float('nan'))) != value:
            raise ValueError(f'CA-ECAD state mismatch for {key}: {actual.get(key)!r} != {value!r}')
    return prior


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--prior-state-path', required=True)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--eta', type=float, default=0.25)
    parser.add_argument('--kappa', type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = os.path.abspath(os.path.expanduser(args.input_dir))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir or input_dir))
    state_path = os.path.abspath(os.path.expanduser(args.prior_state_path))
    rollout_paths = sorted(glob.glob(os.path.join(input_dir, 'phase2_step_*_rollouts.jsonl')))
    if not rollout_paths:
        raise FileNotFoundError(f'no no-update rollout files found under {input_dir}')
    if os.path.exists(os.path.join(output_dir, 'ca_ecad_signal_audit.json')):
        raise FileExistsError(f'refusing to overwrite signal audit in {output_dir}')

    source_rows = []
    for path in rollout_paths:
        source_rows.extend(_read_jsonl(path))
    required = {'group_uid', 'reward', 'ordered_search_document_ids', 'policy_segment_spans'}
    for index, row in enumerate(source_rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f'rollout {index} is missing {missing}')

    prior = _load_state(state_path, args.alpha, args.eta, args.kappa)
    result = compute_ca_ecad_credits(
        group_uids=[row['group_uid'] for row in source_rows],
        rewards=[row['reward'] for row in source_rows],
        search_histories=[row['ordered_search_document_ids'] for row in source_rows],
        success_prior=prior,
        alpha=args.alpha,
        eta=args.eta,
        kappa=args.kappa,
    )

    acquisition_by_turn = defaultdict(list)
    acquisition_mass_by_turn = defaultdict(float)
    acquisition_tokens_by_turn = defaultdict(int)
    utilization_values = []
    utilization_mass = 0.0
    utilization_tokens = 0
    audit_rows = []
    for source, credit in zip(source_rows, result['records']):
        spans = source['policy_segment_spans']
        turn_rows = []
        for turn in credit['turns']:
            segment_id = int(turn['valid_search_turn'])
            token_count = _segment_token_count(spans, segment_id)
            advantage = float(turn['acquisition_advantage'])
            acquisition_by_turn[segment_id].append(advantage)
            acquisition_tokens_by_turn[segment_id] += token_count
            acquisition_mass_by_turn[segment_id] += abs(advantage) * token_count
            turn_rows.append({
                'valid_search_turn': segment_id,
                'top1_document_id': turn['ordered_document_ids'][0] if turn['ordered_document_ids'] else None,
                'peer_support': turn['peer_support'],
                'environment_value': turn['environment_value'],
                'acquisition_advantage': advantage,
                'policy_token_count': token_count,
                'absolute_token_signal_mass': abs(advantage) * token_count,
            })
        final_segment = len(credit['turns']) + 1
        final_token_count = _segment_token_count(spans, final_segment)
        utilization = float(credit['utilization_advantage'] if credit['turns'] else credit['no_search_advantage'])
        utilization_values.append(utilization)
        utilization_tokens += final_token_count
        utilization_mass += abs(utilization) * final_token_count
        audit_rows.append({
            'diagnostic_step': source.get('diagnostic_step'),
            'prompt_uid': source.get('prompt_uid'),
            'group_uid': source['group_uid'],
            'reward': credit['reward'],
            'baseline': credit['baseline'],
            'search_turns': turn_rows,
            'utilization_advantage': utilization,
            'utilization_policy_token_count': final_token_count,
            'utilization_absolute_token_signal_mass': abs(utilization) * final_token_count,
            'conservation_error': credit['conservation_error'],
        })

    acquisition_summary = {}
    total_acquisition_mass = 0.0
    total_acquisition_tokens = 0
    for turn, values in sorted(acquisition_by_turn.items()):
        absolute = _absolute_summary(values)
        absolute.update({
            'policy_token_count': acquisition_tokens_by_turn[turn],
            'absolute_token_signal_mass': acquisition_mass_by_turn[turn],
            'mean_abs_token_advantage': (
                acquisition_mass_by_turn[turn] / acquisition_tokens_by_turn[turn]
                if acquisition_tokens_by_turn[turn] else 0.0
            ),
        })
        acquisition_summary[f'k_{turn}'] = absolute
        total_acquisition_mass += acquisition_mass_by_turn[turn]
        total_acquisition_tokens += acquisition_tokens_by_turn[turn]

    total_mass = total_acquisition_mass + utilization_mass
    summary = {
        'version': 1,
        'scope': (
            'Frozen-policy, no-optimizer signal audit. Token signal mass is an '
            'absolute-advantage-times-policy-token proxy, not a separately backpropagated gradient norm.'
        ),
        'input_rollout_files': rollout_paths,
        'prior_state_path': state_path,
        'success_prior': prior,
        'hyperparameters': {'alpha': args.alpha, 'eta': args.eta, 'kappa': args.kappa},
        'rollout_count': len(source_rows),
        'prompt_group_count': len({str(row['group_uid']) for row in source_rows}),
        'acquisition_by_search_round': acquisition_summary,
        'utilization': {
            **_absolute_summary(utilization_values),
            'policy_token_count': utilization_tokens,
            'absolute_token_signal_mass': utilization_mass,
            'mean_abs_token_advantage': utilization_mass / utilization_tokens if utilization_tokens else 0.0,
        },
        'absolute_policy_signal_mass': {
            'acquisition': total_acquisition_mass,
            'utilization': utilization_mass,
            'total': total_mass,
            'acquisition_share': total_acquisition_mass / total_mass if total_mass else 0.0,
            'utilization_share': utilization_mass / total_mass if total_mass else 0.0,
        },
        'environment_mode_diversity_by_search_round': _mode_diversity(source_rows),
        'max_conservation_error': result['metrics']['ca_ecad/phase2/max_conservation_error'],
    }
    if summary['max_conservation_error'] > 1e-12:
        raise RuntimeError(f"credit conservation failed: {summary['max_conservation_error']}")

    os.makedirs(output_dir, exist_ok=True)
    write_jsonl(os.path.join(output_dir, 'ca_ecad_signal_credit_records.jsonl'), audit_rows)
    write_json(os.path.join(output_dir, 'ca_ecad_signal_audit.json'), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
