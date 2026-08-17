#!/usr/bin/env python3
"""Aggregate V7 Phase-2 no-update geometry records and apply frozen gates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1)
        for offset in range(start, end):
            ranks[order[offset]] = rank
        start = end
    return ranks


def spearman(left, right):
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left = average_ranks(left)
    right = average_ranks(right)
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    return 0.0 if left_norm * right_norm == 0 else numerator / (left_norm * right_norm)


def load_records(path):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or "per_state" not in record:
                raise ValueError(f"Malformed record at line {line_number}")
            records.append(record)
    return records


def summarize(records, *, min_batches, frozen_radius):
    if not records:
        raise ValueError("V7 geometry artifact is empty")
    steps = [int(record["outer_update_step"]) for record in records]
    if len(set(steps)) != len(steps):
        raise ValueError("V7 geometry artifact contains duplicate outer-update steps")
    if any(abs(float(record["accept_radius"]) - frozen_radius) > 1e-15 for record in records):
        raise ValueError("V7 records do not all use the frozen accept radius")

    required = (
        "token", "k4", "reference", "repeat", "ess_k4",
        "ess_reference", "valid_probe_count",
    )
    vectors = {key: [] for key in required}
    batch_reference_means = []
    invariant_failures = []
    for record in records:
        state = record["per_state"]
        lengths = []
        for key in required:
            values = [float(value) for value in state.get(key, [])]
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite {key} value at step {record['outer_update_step']}")
            vectors[key].extend(values)
            lengths.append(len(values))
        if len(set(lengths)) != 1 or lengths[0] < 2:
            raise ValueError(
                f"V7 per-state vectors are misaligned at step {record['outer_update_step']}"
            )
        batch_reference_means.append(statistics.fmean(state["reference"]))
        stats = record.get("statistics", {})
        if float(stats.get("final_restore_max_abs", 1.0)) != 0.0:
            invariant_failures.append(f"step {record['outer_update_step']}: restore")
        if float(stats.get("optimizer_state_restored_empty", 0.0)) != 1.0:
            invariant_failures.append(f"step {record['outer_update_step']}: optimizer")
        if float(stats.get("no_update_committed", 0.0)) != 1.0:
            invariant_failures.append(f"step {record['outer_update_step']}: commit")
        if float(stats.get("proposal_update_norm", 0.0)) <= 0.0:
            invariant_failures.append(f"step {record['outer_update_step']}: zero proposal")

    token = vectors["token"]
    k4 = vectors["k4"]
    reference = vectors["reference"]
    repeat = vectors["repeat"]
    jitter = [abs(left - right) for left, right in zip(reference, repeat)]
    agreement = statistics.fmean(
        (left <= frozen_radius) == (right <= frozen_radius)
        for left, right in zip(k4, reference)
    )
    k_rho = spearman(k4, reference)
    token_rho = spearman(token, reference)
    token_median = statistics.median(token)
    reference_median = statistics.median(reference)
    mismatch_rate = statistics.fmean(
        (left > token_median) != (right > reference_median)
        for left, right in zip(token, reference)
    )
    high_token_low_env_rate = statistics.fmean(
        (left > token_median) and (right <= reference_median)
        for left, right in zip(token, reference)
    )
    low_token_high_env_rate = statistics.fmean(
        (left <= token_median) and (right > reference_median)
        for left, right in zip(token, reference)
    )
    between_batch_change = [
        abs(right - left)
        for left, right in zip(batch_reference_means, batch_reference_means[1:])
    ]
    between_batch_scale = (
        statistics.median(between_batch_change)
        if between_batch_change
        else 0.0
    )
    jitter_mean = statistics.fmean(jitter)
    jitter_ratio = jitter_mean / max(between_batch_scale, 1e-15)

    enough_batches = len(records) >= min_batches
    estimator_pass = (
        k_rho >= 0.7
        and agreement >= 0.8
        and jitter_ratio <= 0.1
        and not invariant_failures
    )
    geometry_mismatch = abs(token_rho) < 0.95 and mismatch_rate > 0.05
    if not enough_batches:
        verdict = "INCONCLUSIVE_NEED_MORE_BATCHES"
    elif not estimator_pass:
        verdict = "ESTIMATOR_NO_GO"
    elif not geometry_mismatch:
        verdict = "CORE_GEOMETRY_NO_GO"
    else:
        verdict = "GO_TO_V7_PHASE_3_CALIBRATION"

    return {
        "verdict": verdict,
        "audit_batch_count": len(records),
        "eligible_state_count": len(reference),
        "frozen_accept_radius": frozen_radius,
        "k4_reference_spearman": k_rho,
        "accept_reject_agreement": agreement,
        "repeat_jitter_mean": jitter_mean,
        "between_batch_reference_change_median": between_batch_scale,
        "jitter_to_between_batch_ratio": jitter_ratio,
        "token_environment_spearman": token_rho,
        "token_environment_mismatch_rate": mismatch_rate,
        "high_token_low_environment_rate": high_token_low_env_rate,
        "low_token_high_environment_rate": low_token_high_env_rate,
        "mean_reference_ess": statistics.fmean(vectors["ess_reference"]),
        "mean_valid_probe_count": statistics.fmean(vectors["valid_probe_count"]),
        "estimator_pass": estimator_pass,
        "geometry_mismatch_observed": geometry_mismatch,
        "no_update_invariant_pass": not invariant_failures,
        "invariant_failures": invariant_failures,
        "thresholds": {
            "minimum_batches": min_batches,
            "minimum_k4_reference_spearman": 0.7,
            "minimum_accept_reject_agreement": 0.8,
            "maximum_jitter_ratio": 0.1,
            "maximum_abs_token_environment_spearman_for_mismatch": 0.95,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--min-batches", type=int, default=20)
    parser.add_argument("--frozen-radius", type=float, default=1e-3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(
        load_records(args.artifact),
        min_batches=args.min_batches,
        frozen_radius=args.frozen_radius,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
