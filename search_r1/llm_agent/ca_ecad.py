"""Pure-Python CA-ECAD diagnostics and Phase-3 credit construction.

This module intentionally has no torch/ray dependency.  The online rollout
path records batch-aligned traces; this module validates and analyzes those
records without updating model parameters, while Phase 3 reuses the same
validated construction to create token-level policy advantages.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


INVALID_SEGMENT_ID = -1
ENVIRONMENT_TURN_ID = 0
ACTION_INVALID = 0
ACTION_SEARCH = 1
ACTION_ANSWER = 2


def canonical_document_id(value: Any) -> Optional[str]:
    """Return a stable non-empty document ID, or ``None`` when unavailable."""

    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def summarize_absolute_credit(values: Sequence[float]) -> Dict[str, float]:
    """Summarize credit magnitude without allowing opposite signs to cancel."""

    magnitudes = [abs(float(value)) for value in values]
    if not magnitudes:
        return {
            "count": 0.0,
            "mean_abs": 0.0,
            "std_abs": 0.0,
            "p50_abs": 0.0,
            "p90_abs": 0.0,
            "rate_abs_gt_0_01": 0.0,
            "rate_abs_gt_0_05": 0.0,
        }
    ordered = sorted(magnitudes)

    def nearest(quantile: float) -> float:
        return float(ordered[int(round((len(ordered) - 1) * quantile))])

    mean = sum(magnitudes) / len(magnitudes)
    return {
        "count": float(len(magnitudes)),
        "mean_abs": float(mean),
        "std_abs": float(math.sqrt(sum((value - mean) ** 2 for value in magnitudes) / len(magnitudes))),
        "p50_abs": nearest(0.50),
        "p90_abs": nearest(0.90),
        "rate_abs_gt_0_01": float(sum(value > 0.01 for value in magnitudes) / len(magnitudes)),
        "rate_abs_gt_0_05": float(sum(value > 0.05 for value in magnitudes) / len(magnitudes)),
    }


def ordered_document_ids(retrieval_result: Sequence[Mapping[str, Any]]) -> Tuple[Optional[str], ...]:
    """Extract ranked corpus IDs without sorting or title fallback."""

    ids: List[Optional[str]] = []
    for rank, item in enumerate(retrieval_result):
        if not isinstance(item, Mapping):
            raise TypeError(f"retrieval result at rank {rank} is not a mapping")
        document = item.get("document", item)
        if not isinstance(document, Mapping):
            raise TypeError(f"document at rank {rank} is not a mapping")
        ids.append(canonical_document_id(document.get("id")))
    return tuple(ids)


def top1_peer_key(group_uid: Any, valid_search_turn: int, ordered_ids: Sequence[Any]) -> Optional[Tuple[str, int, str]]:
    """Build ``(group UID, valid-search turn, Top-1 doc ID)`` or ``None``."""

    if valid_search_turn < 1:
        raise ValueError("valid_search_turn must be one-indexed")
    if not ordered_ids:
        return None
    top1 = canonical_document_id(ordered_ids[0])
    if top1 is None:
        return None
    return str(group_uid), int(valid_search_turn), top1


def build_policy_segment_ids(
    generation_turn_ids: Sequence[int],
    turn_action_codes: Sequence[int],
    policy_mask: Sequence[int],
) -> List[int]:
    """Partition every model token into acquisition turns or final utilization.

    Positive ``generation_turn_ids`` identify model-generation calls, zero is
    reserved for environment observations, and negative values are padding.
    Invalid model turns are folded into the next executed search segment; if no
    later search exists, they belong to the final utilization segment.
    """

    if not (len(generation_turn_ids) == len(policy_mask)):
        raise ValueError("generation_turn_ids and policy_mask must have the same length")

    search_generation_turns = [
        turn_index + 1
        for turn_index, action in enumerate(turn_action_codes)
        if int(action) == ACTION_SEARCH
    ]
    segment_ids: List[int] = []
    for position, (generation_turn, is_policy) in enumerate(zip(generation_turn_ids, policy_mask)):
        is_policy = bool(is_policy)
        generation_turn = int(generation_turn)
        if not is_policy:
            segment_ids.append(INVALID_SEGMENT_ID)
            continue
        if generation_turn <= 0:
            raise ValueError(f"policy token at position {position} has no positive generation turn")

        segment = len(search_generation_turns) + 1
        for search_index, search_generation_turn in enumerate(search_generation_turns, start=1):
            if generation_turn <= search_generation_turn:
                segment = search_index
                break
        segment_ids.append(segment)

    return segment_ids


def contiguous_policy_spans(segment_ids: Sequence[int], policy_mask: Sequence[int]) -> Dict[str, List[List[int]]]:
    """Return half-open token spans for each positive CA-ECAD segment."""

    if len(segment_ids) != len(policy_mask):
        raise ValueError("segment_ids and policy_mask must have the same length")
    spans: Dict[str, List[List[int]]] = defaultdict(list)
    start: Optional[int] = None
    current: Optional[int] = None
    for position, (segment, is_policy) in enumerate(zip(segment_ids, policy_mask)):
        segment = int(segment)
        effective = segment if bool(is_policy) and segment > 0 else None
        if effective == current:
            continue
        if current is not None and start is not None:
            spans[str(current)].append([start, position])
        current = effective
        start = position if effective is not None else None
    if current is not None and start is not None:
        spans[str(current)].append([start, len(segment_ids)])
    return dict(spans)


def validate_segment_partition(
    segment_ids: Sequence[int],
    policy_mask: Sequence[int],
    search_count: int,
) -> None:
    """Fail when policy ownership is incomplete or an impossible segment appears."""

    if len(segment_ids) != len(policy_mask):
        raise ValueError("segment_ids and policy_mask must have the same length")
    max_segment = int(search_count) + 1
    for position, (segment, is_policy) in enumerate(zip(segment_ids, policy_mask)):
        segment = int(segment)
        if bool(is_policy):
            if not 1 <= segment <= max_segment:
                raise ValueError(f"policy token {position} has invalid segment {segment}")
        elif segment != INVALID_SEGMENT_ID:
            raise ValueError(f"masked token {position} must use segment {INVALID_SEGMENT_ID}")


def _validate_credit_inputs(
    group_uids: Sequence[Any],
    rewards: Sequence[float],
    search_histories: Sequence[Sequence[Sequence[Any]]],
    success_prior: float,
    alpha: float,
    eta: float,
    kappa: float,
) -> None:
    if not (len(group_uids) == len(rewards) == len(search_histories)):
        raise ValueError("group_uids, rewards, and search_histories must align")
    if not group_uids:
        raise ValueError("at least one rollout is required")
    if not 0.0 <= float(success_prior) <= 1.0:
        raise ValueError("success_prior must lie in [0, 1]")
    if alpha <= 0 or kappa <= 0 or not 0 <= eta <= 1:
        raise ValueError("require alpha>0, kappa>0, and eta in [0, 1]")
    for reward in rewards:
        if not math.isfinite(float(reward)) or float(reward) not in (0.0, 1.0):
            raise ValueError("CA-ECAD v11.1 requires binary outcome rewards")


def compute_ca_ecad_credits(
    group_uids: Sequence[Any],
    rewards: Sequence[float],
    search_histories: Sequence[Sequence[Sequence[Any]]],
    success_prior: float,
    alpha: float = 2.0,
    eta: float = 0.25,
    kappa: float = 1.0,
) -> Dict[str, Any]:
    """Compute the frozen v11.1 LOO baseline and telescoping credits offline."""

    _validate_credit_inputs(group_uids, rewards, search_histories, success_prior, alpha, eta, kappa)
    uids = [str(uid) for uid in group_uids]
    rewards = [float(reward) for reward in rewards]
    normalized_histories = [
        [tuple(canonical_document_id(doc_id) for doc_id in turn) for turn in history]
        for history in search_histories
    ]

    group_members: Dict[str, List[int]] = defaultdict(list)
    for index, uid in enumerate(uids):
        group_members[uid].append(index)

    records: List[Dict[str, Any]] = []
    conservation_errors: List[float] = []
    for index, (uid, reward, history) in enumerate(zip(uids, rewards, normalized_histories)):
        members = group_members[uid]
        group_size = len(members)
        other_reward_sum = sum(rewards[peer] for peer in members if peer != index)
        baseline = (other_reward_sum + alpha * success_prior) / (group_size - 1 + alpha)

        previous_value = baseline
        turns: List[Dict[str, Any]] = []
        for zero_based_turn, ordered_ids in enumerate(history):
            valid_search_turn = zero_based_turn + 1
            peer_key = top1_peer_key(uid, valid_search_turn, ordered_ids)
            peers: List[int] = []
            if peer_key is not None:
                for peer in members:
                    if peer == index or len(normalized_histories[peer]) <= zero_based_turn:
                        continue
                    peer_ids = normalized_histories[peer][zero_based_turn]
                    if top1_peer_key(uid, valid_search_turn, peer_ids) == peer_key:
                        peers.append(peer)

            peer_successes = sum(rewards[peer] for peer in peers)
            support = len(peers)
            current_value = (
                peer_successes + eta * reward + kappa * baseline
            ) / (support + eta + kappa)
            acquisition_advantage = current_value - previous_value
            turns.append({
                "valid_search_turn": valid_search_turn,
                "ordered_document_ids": list(ordered_ids),
                "peer_key": list(peer_key) if peer_key is not None else None,
                "peer_support": support,
                "peer_successes": peer_successes,
                "environment_value": current_value,
                "acquisition_advantage": acquisition_advantage,
            })
            previous_value = current_value

        utilization_advantage = reward - previous_value
        acquisition_sum = sum(turn["acquisition_advantage"] for turn in turns)
        conservation_error = acquisition_sum + utilization_advantage - (reward - baseline)
        conservation_errors.append(abs(conservation_error))
        records.append({
            "group_uid": uid,
            "reward": reward,
            "baseline": baseline,
            "turns": turns,
            "utilization_advantage": utilization_advantage,
            "no_search_advantage": reward - baseline if not turns else None,
            "conservation_error": conservation_error,
        })

    metrics = compute_peer_diagnostics(uids, rewards, normalized_histories)
    metrics["ca_ecad/phase2/max_conservation_error"] = max(conservation_errors, default=0.0)
    metrics["ca_ecad/phase2/success_prior"] = float(success_prior)
    return {"records": records, "metrics": metrics}


def credit_values_by_segment(credit_record: Mapping[str, Any]) -> Dict[int, float]:
    """Map a validated credit record onto its positive policy segment IDs.

    Segment ``k`` owns the acquisition credit for valid search turn ``k``;
    the final segment ``K + 1`` owns the utilization credit.  A no-search
    rollout has only segment ``1``, which receives its outcome-minus-LOO
    baseline.  Keeping this mapping pure and explicit prevents the online
    trainer from silently assigning environment-observation tokens credit.
    """

    turns = credit_record.get("turns", [])
    if not isinstance(turns, Sequence):
        raise TypeError("credit record turns must be a sequence")

    values: Dict[int, float] = {}
    for expected_turn, turn in enumerate(turns, start=1):
        if not isinstance(turn, Mapping):
            raise TypeError("credit record turn must be a mapping")
        valid_search_turn = int(turn.get("valid_search_turn", -1))
        if valid_search_turn != expected_turn:
            raise ValueError("credit record search turns must be contiguous and one-indexed")
        value = float(turn.get("acquisition_advantage"))
        if not math.isfinite(value):
            raise ValueError("acquisition advantage must be finite")
        values[expected_turn] = value

    final_segment = len(turns) + 1
    if turns:
        utilization = float(credit_record.get("utilization_advantage"))
    else:
        utilization = float(credit_record.get("no_search_advantage"))
    if not math.isfinite(utilization):
        raise ValueError("utilization advantage must be finite")
    values[final_segment] = utilization
    return values


def compute_peer_diagnostics(
    group_uids: Sequence[Any],
    rewards: Sequence[float],
    search_histories: Sequence[Sequence[Sequence[Any]]],
) -> Dict[str, float]:
    """Compute structural peer diagnostics without needing a success prior."""

    if not (len(group_uids) == len(rewards) == len(search_histories)):
        raise ValueError("diagnostic inputs must align")
    uids = [str(uid) for uid in group_uids]
    rewards = [float(reward) for reward in rewards]
    histories = [
        [tuple(canonical_document_id(doc_id) for doc_id in turn) for turn in history]
        for history in search_histories
    ]
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, uid in enumerate(uids):
        groups[uid].append(index)

    mode_members: Dict[Tuple[str, int, str], List[int]] = defaultdict(list)
    total_turns = 0
    eligible_turns = 0
    total_document_slots = 0
    present_document_ids = 0
    duplicate_document_id_turns = 0
    unique_top1_ids = set()
    turn_count_by_k: Counter[int] = Counter()
    for index, (uid, history) in enumerate(zip(uids, histories)):
        for zero_based_turn, ordered_ids in enumerate(history):
            valid_search_turn = zero_based_turn + 1
            total_turns += 1
            turn_count_by_k[valid_search_turn] += 1
            key = top1_peer_key(uid, valid_search_turn, ordered_ids)
            total_document_slots += len(ordered_ids)
            present_document_ids += sum(document_id is not None for document_id in ordered_ids)
            present_turn_ids = [document_id for document_id in ordered_ids if document_id is not None]
            duplicate_document_id_turns += int(len(present_turn_ids) != len(set(present_turn_ids)))
            if key is not None:
                eligible_turns += 1
                unique_top1_ids.add(key[2])
                mode_members[key].append(index)

    supported_turns = 0
    mixed_turns = 0
    disagreement_numerator = 0
    disagreement_denominator = 0
    supported_count_by_k: Counter[int] = Counter()
    for key, members in mode_members.items():
        if len(members) < 2:
            continue
        supported_turns += len(members)
        supported_count_by_k[key[1]] += len(members)
        member_rewards = [rewards[index] for index in members]
        if min(member_rewards) < max(member_rewards):
            mixed_turns += len(members)
        for index in members:
            peer_rewards = [rewards[peer] for peer in members if peer != index]
            peer_mean = sum(peer_rewards) / len(peer_rewards)
            if peer_mean == 0.5:
                continue
            disagreement_denominator += 1
            disagreement_numerator += int((peer_mean > 0.5) != bool(rewards[index]))

    all_zero_groups = 0
    all_zero_repeated_groups = 0
    for uid, members in groups.items():
        if any(rewards[index] != 0.0 for index in members):
            continue
        all_zero_groups += 1
        repeated = any(
            len(mode_members[key]) > 1
            for key in mode_members
            if key[0] == uid
        )
        all_zero_repeated_groups += int(repeated)

    top1_pairs = 0
    matching_top3_pairs = 0
    for key, members in mode_members.items():
        if len(members) < 2:
            continue
        zero_based_turn = key[1] - 1
        for left, right in combinations(members, 2):
            top1_pairs += 1
            left_top3 = histories[left][zero_based_turn][:3]
            right_top3 = histories[right][zero_based_turn][:3]
            matching_top3_pairs += int(left_top3 == right_top3)

    metrics: Dict[str, float] = {
        "ca_ecad/phase2/rollout_count": float(len(uids)),
        "ca_ecad/phase2/prompt_group_count": float(len(groups)),
        "ca_ecad/phase2/valid_search_turn_count": float(total_turns),
        "ca_ecad/phase2/peer_eligible_turn_rate": eligible_turns / total_turns if total_turns else 0.0,
        "ca_ecad/phase2/top1_document_id_presence_rate": eligible_turns / total_turns if total_turns else 0.0,
        "ca_ecad/phase2/all_document_id_presence_rate": (
            present_document_ids / total_document_slots if total_document_slots else 0.0
        ),
        "ca_ecad/phase2/duplicate_document_id_turn_rate": (
            duplicate_document_id_turns / total_turns if total_turns else 0.0
        ),
        "ca_ecad/phase2/unique_top1_document_count": float(len(unique_top1_ids)),
        "ca_ecad/phase2/top1_peer_supported_turn_rate": supported_turns / total_turns if total_turns else 0.0,
        "ca_ecad/phase2/top1_peer_supported_turn_count": float(supported_turns),
        "ca_ecad/phase2/peer_free_turn_rate": 1.0 - supported_turns / total_turns if total_turns else 0.0,
        "ca_ecad/phase2/mixed_outcome_peer_turn_rate": mixed_turns / supported_turns if supported_turns else 0.0,
        "ca_ecad/phase2/mixed_outcome_peer_turn_count": float(mixed_turns),
        "ca_ecad/phase2/all_zero_group_count": float(all_zero_groups),
        "ca_ecad/phase2/all_zero_repeated_mode_group_count": float(all_zero_repeated_groups),
        "ca_ecad/phase2/all_zero_repeated_mode_rate": (
            all_zero_repeated_groups / all_zero_groups if all_zero_groups else 0.0
        ),
        "ca_ecad/phase2/self_peer_disagreement_rate": (
            disagreement_numerator / disagreement_denominator if disagreement_denominator else 0.0
        ),
        "ca_ecad/phase2/ordered_top3_match_given_top1_match": (
            matching_top3_pairs / top1_pairs if top1_pairs else 0.0
        ),
    }
    all_peer_supports = []
    for index, (uid, history) in enumerate(zip(uids, histories)):
        for zero_based_turn, ordered_ids in enumerate(history):
            key = top1_peer_key(uid, zero_based_turn + 1, ordered_ids)
            support = 0 if key is None else len(mode_members[key]) - 1
            all_peer_supports.append(max(0, support))
    sorted_nonzero_supports = sorted(support for support in all_peer_supports if support > 0)

    def _nearest_rank(values: Sequence[int], quantile: float) -> float:
        if not values:
            return 0.0
        rank = int(round((len(values) - 1) * quantile))
        return float(values[rank])

    metrics.update({
        "ca_ecad/phase2/peer_support_mean": (
            sum(all_peer_supports) / len(all_peer_supports) if all_peer_supports else 0.0
        ),
        "ca_ecad/phase2/peer_support_max": float(max(all_peer_supports, default=0)),
        "ca_ecad/phase2/nonzero_peer_support_p25": _nearest_rank(sorted_nonzero_supports, 0.25),
        "ca_ecad/phase2/nonzero_peer_support_p50": _nearest_rank(sorted_nonzero_supports, 0.50),
        "ca_ecad/phase2/nonzero_peer_support_p75": _nearest_rank(sorted_nonzero_supports, 0.75),
    })
    for valid_search_turn, count in sorted(turn_count_by_k.items()):
        metrics[f"ca_ecad/phase2/valid_search_turn_count/k_{valid_search_turn}"] = float(count)
        metrics[f"ca_ecad/phase2/top1_peer_supported_turn_rate/k_{valid_search_turn}"] = (
            supported_count_by_k[valid_search_turn] / count if count else 0.0
        )

    # Peer support near G-1 can mean either useful agreement or a collapsed
    # prompt group.  Log the prompt-level number of distinct Top-1 modes so
    # that those cases are distinguishable online.
    mode_count_by_turn: Dict[int, List[int]] = defaultdict(list)
    for uid, members in groups.items():
        max_turn_count = max((len(histories[index]) for index in members), default=0)
        for zero_based_turn in range(max_turn_count):
            top1_modes = {
                canonical_document_id(histories[index][zero_based_turn][0])
                for index in members
                if len(histories[index]) > zero_based_turn and histories[index][zero_based_turn]
            }
            top1_modes.discard(None)
            if top1_modes:
                mode_count_by_turn[zero_based_turn + 1].append(len(top1_modes))
    for valid_search_turn, counts in sorted(mode_count_by_turn.items()):
        ordered_counts = sorted(counts)
        metrics[f"ca_ecad/phase2/distinct_top1_mode_mean/k_{valid_search_turn}"] = (
            sum(counts) / len(counts)
        )
        metrics[f"ca_ecad/phase2/distinct_top1_mode_p50/k_{valid_search_turn}"] = _nearest_rank(
            ordered_counts, 0.50
        )
        metrics[f"ca_ecad/phase2/distinct_top1_mode_p90/k_{valid_search_turn}"] = _nearest_rank(
            ordered_counts, 0.90
        )
        metrics[f"ca_ecad/phase2/all_same_top1_mode_rate/k_{valid_search_turn}"] = (
            sum(count == 1 for count in counts) / len(counts)
        )
        metrics[f"ca_ecad/phase2/multiple_top1_mode_rate/k_{valid_search_turn}"] = (
            sum(count > 1 for count in counts) / len(counts)
        )
    return metrics


def write_json(path: Union[os.PathLike, str], payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_jsonl(path: Union[os.PathLike, str], records: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, path)
