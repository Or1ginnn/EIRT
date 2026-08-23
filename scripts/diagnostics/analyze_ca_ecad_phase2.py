#!/usr/bin/env python3
"""Aggregate no-update CA-ECAD Phase-2 rollouts and build frozen credits."""

import argparse
import glob
import json
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from search_r1.llm_agent.ca_ecad import compute_ca_ecad_credits, write_json, write_jsonl


def _read_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'invalid JSON at {path}:{line_number}') from exc
    return records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--min-calibration-prompts', type=int, default=256)
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--eta', type=float, default=0.25)
    parser.add_argument('--kappa', type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = os.path.abspath(os.path.expanduser(args.input_dir))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir or input_dir))
    rollout_paths = sorted(glob.glob(os.path.join(input_dir, 'phase2_step_*_rollouts.jsonl')))
    if not rollout_paths:
        raise FileNotFoundError(f'no Phase-2 rollout files found under {input_dir}')

    records = []
    for path in rollout_paths:
        records.extend(_read_jsonl(path))
    if not records:
        raise ValueError('Phase-2 rollout files are empty')

    required = {'group_uid', 'reward', 'ordered_search_document_ids'}
    for row_index, record in enumerate(records):
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f'rollout {row_index} is missing {missing}')

    group_uids = [str(record['group_uid']) for record in records]
    rewards = [float(record['reward']) for record in records]
    histories = [record['ordered_search_document_ids'] for record in records]
    prompt_group_count = len(set(group_uids))
    if prompt_group_count < args.min_calibration_prompts:
        raise ValueError(
            f'need at least {args.min_calibration_prompts} no-update prompt groups; '
            f'found {prompt_group_count}'
        )

    success_prior = sum(rewards) / len(rewards)
    result = compute_ca_ecad_credits(
        group_uids=group_uids,
        rewards=rewards,
        search_histories=histories,
        success_prior=success_prior,
        alpha=args.alpha,
        eta=args.eta,
        kappa=args.kappa,
    )
    metrics = result['metrics']
    checks = {
        'enough_calibration_prompts': prompt_group_count >= args.min_calibration_prompts,
        'all_top1_ids_present': metrics['ca_ecad/phase2/top1_document_id_presence_rate'] == 1.0,
        'peer_support_observed': metrics['ca_ecad/phase2/top1_peer_supported_turn_count'] > 0,
        'mixed_outcome_peer_observed': metrics['ca_ecad/phase2/mixed_outcome_peer_turn_count'] > 0,
        'telescoping_conservation_exact': metrics['ca_ecad/phase2/max_conservation_error'] <= 1e-12,
    }
    summary = {
        'input_files': rollout_paths,
        'output_dir': output_dir,
        'rollout_count': len(records),
        'prompt_group_count': prompt_group_count,
        'success_prior': success_prior,
        'hyperparameters': {
            'alpha': args.alpha,
            'eta': args.eta,
            'kappa': args.kappa,
        },
        'metrics': metrics,
        'phase2_checks': checks,
        'ready_for_phase3_implementation': all(checks.values()),
        'interpretation': (
            'This is a no-update structural gate. It does not establish QA improvement or a paper claim.'
        ),
    }

    credit_records = []
    for source_record, credit_record in zip(records, result['records']):
        credit_records.append({
            'diagnostic_step': source_record.get('diagnostic_step'),
            'prompt_uid': source_record.get('prompt_uid'),
            'row_index_after_balance': source_record.get('row_index_after_balance'),
            **credit_record,
        })

    write_jsonl(os.path.join(output_dir, 'phase2_credit_records.jsonl'), credit_records)
    write_json(os.path.join(output_dir, 'phase2_analysis.json'), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
