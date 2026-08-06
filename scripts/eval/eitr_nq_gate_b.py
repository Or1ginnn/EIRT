#!/usr/bin/env python3
"""Run the EITR NQ branch-to-terminal downstream-consequence diagnostic."""

import argparse
import json
import math
import re
import string
from collections import defaultdict
from pathlib import Path

import numpy as np

from eitr_nq_gate_a import (
    dump_json,
    load_vllm,
    log,
    make_sampling_params,
    normalize_text,
    rankdata,
    read_jsonl,
    spearman,
)


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
PLAIN_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*(?:final\s+)?answer\s*:\s*([^\n]+)",
    re.IGNORECASE,
)

LINGUISTIC_PREDICTORS = [
    "policy_embedding_distance",
    "token_jaccard_distance",
    "normalized_edit_distance",
    "normalized_logprob_gap",
]

ENVIRONMENT_PREDICTORS = [
    "retrieval_score_js_distance",
    "document_jaccard_distance",
    "rbo_distance",
    "evidence_embedding_distance",
]

PRIMARY_LANGUAGE_PREDICTOR = "policy_embedding_distance"
PRIMARY_ENVIRONMENT_PREDICTOR = "retrieval_score_js_distance"


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_tagged_answer(text):
    matches = ANSWER_RE.findall(str(text))
    return matches[-1].strip() if matches else None


def extract_answer(text):
    tagged = extract_tagged_answer(text)
    if tagged is not None:
        return tagged, "answer_tag"
    matches = PLAIN_ANSWER_RE.findall(str(text))
    if matches:
        return matches[-1].strip(), "plain_answer"
    return None, None


def rescore_outcome(row):
    row = dict(row)
    tagged_answer = extract_tagged_answer(row.get("response", ""))
    answer, answer_format = extract_answer(row.get("response", ""))
    gold_answers = row.get("gold_answers", [])
    row.update(
        {
            "answer": answer,
            "answer_format": answer_format,
            "normalized_answer": normalize_answer(answer or ""),
            "answer_present": answer is not None,
            "answer_em": em_check(answer, gold_answers),
            "answer_subem": subem_check(answer, gold_answers),
            "strict_tag_answer_present": tagged_answer is not None,
            "strict_tag_answer_em": em_check(tagged_answer, gold_answers),
        }
    )
    return row


def em_check(prediction, gold_answers):
    if prediction is None:
        return 0
    normalized = normalize_answer(prediction)
    return int(any(normalized == normalize_answer(gold) for gold in gold_answers))


def subem_check(prediction, gold_answers):
    if prediction is None:
        return 0
    normalized = normalize_answer(prediction)
    return int(
        any(normalize_answer(gold) and normalize_answer(gold) in normalized for gold in gold_answers)
    )


def branch_key(state_id, sample_index):
    return f"{state_id}::{int(sample_index)}"


def format_information(query, retrieval, max_document_chars):
    sections = ["<information>", f"[Query] {query}"]
    for item in retrieval:
        title = str(item.get("title") or "").replace("\n", " ").strip()
        contents = str(item.get("contents") or "").strip()
        if max_document_chars > 0 and len(contents) > max_document_chars:
            contents = contents[:max_document_chars].rsplit(" ", 1)[0] + " ..."
        sections.append(f'Doc {item.get("rank", len(sections) - 1)}(Title: "{title}") {contents}')
    sections.append("</information>")
    return "\n".join(sections)


def build_branch_prompt(state, sample, max_document_chars):
    observation = format_information(
        sample["query"], sample.get("retrieval", []), max_document_chars
    )
    return (
        state["prompt"]
        + state["prefix"]
        + " "
        + sample["query"]
        + " </search>\n\n"
        + observation
        + "\n\n"
    )


def selected_states(states, state_offset, max_states):
    states = states[state_offset:]
    if max_states is not None:
        states = states[:max_states]
    return states


def load_completed(path):
    path = Path(path)
    completed = {}
    if not path.exists():
        return completed
    for row in read_jsonl(path):
        row = rescore_outcome(row)
        completed[branch_key(row["state_id"], row["sample_index"])] = row
    return completed


def summarize_outcomes(rows, expected_branches, selected_state_count):
    answer_count = sum(row["answer_present"] for row in rows)
    answer_em_count = sum(row["answer_em"] for row in rows)
    strict_answer_count = sum(row["strict_tag_answer_present"] for row in rows)
    strict_answer_em_count = sum(row["strict_tag_answer_em"] for row in rows)
    second_search_count = sum(row["second_search"] for row in rows)
    clipped_count = sum(row["response_clipped"] for row in rows)
    return {
        "selected_state_count": selected_state_count,
        "expected_valid_branches": expected_branches,
        "completed_branches": len(rows),
        "answer_count": answer_count,
        "answer_rate": answer_count / len(rows) if rows else 0.0,
        "answer_em_count": answer_em_count,
        "answer_em_rate": answer_em_count / len(rows) if rows else 0.0,
        "answer_subem_count": sum(row["answer_subem"] for row in rows),
        "strict_tag_answer_count": strict_answer_count,
        "strict_tag_answer_rate": strict_answer_count / len(rows) if rows else 0.0,
        "strict_tag_answer_em_count": strict_answer_em_count,
        "strict_tag_answer_em_rate": strict_answer_em_count / len(rows) if rows else 0.0,
        "second_search_count": second_search_count,
        "second_search_rate": second_search_count / len(rows) if rows else 0.0,
        "clipped_count": clipped_count,
        "clipped_rate": clipped_count / len(rows) if rows else 0.0,
    }


def collect(args):
    states = selected_states(read_jsonl(args.input_states), args.state_offset, args.max_states)
    state_lookup = {state["id"]: state for state in states}
    branches = []
    for state in states:
        for sample in state["samples"]:
            if sample.get("valid"):
                branches.append((state, sample))

    completed = load_completed(args.output_jsonl) if args.resume else {}
    pending = [
        (state, sample)
        for state, sample in branches
        if branch_key(state["id"], sample["sample_index"]) not in completed
    ]
    log(
        f"Selected {len(states)} states and {len(branches)} valid branches; "
        f"{len(completed)} already complete, {len(pending)} pending"
    )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not pending:
        rows = list(completed.values())
        report = summarize_outcomes(rows, len(branches), len(states))
        report.update(
            {
                "status": "complete",
                "model": args.model,
                "input_states": args.input_states,
                "output_jsonl": str(output_path),
                "protocol": "fixed first query and cached top-k retrieval, then greedy single-query continuation",
                "temperature": 0.0,
                "max_response_tokens": args.max_response_tokens,
                "max_document_chars": args.max_document_chars,
                "seed": args.seed,
            }
        )
        dump_json(args.collection_report, report)
        log(json.dumps(report, indent=2))
        return

    tokenizer, llm = load_vllm(args)
    params = make_sampling_params(
        temperature=0.0,
        max_tokens=args.max_response_tokens,
        stop=["</answer>", "<search>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )

    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for begin in range(0, len(pending), args.generation_batch_size):
            batch = pending[begin : begin + args.generation_batch_size]
            prompts = [
                build_branch_prompt(state, sample, args.max_document_chars)
                for state, sample in batch
            ]
            outputs = llm.generate(prompts, params, use_tqdm=False)
            for (state, sample), output in zip(batch, outputs):
                candidate = output.outputs[0]
                response = candidate.text
                tagged_answer = extract_tagged_answer(response)
                answer, answer_format = extract_answer(response)
                stop_reason = getattr(candidate, "stop_reason", None)
                finish_reason = getattr(candidate, "finish_reason", None)
                row = {
                    "branch_key": branch_key(state["id"], sample["sample_index"]),
                    "state_id": state["id"],
                    "source_index": state.get("source_index"),
                    "question": state["question"],
                    "gold_answers": state["gold_answers"],
                    "sample_index": sample["sample_index"],
                    "query": sample["query"],
                    "gold_evidence_hit": bool(sample.get("gold_evidence_hit")),
                    "retrieved_doc_ids": [item["doc_id"] for item in sample.get("retrieval", [])],
                    "response": response,
                    "answer": answer,
                    "answer_format": answer_format,
                    "normalized_answer": normalize_answer(answer or ""),
                    "answer_present": answer is not None,
                    "answer_em": em_check(answer, state["gold_answers"]),
                    "answer_subem": subem_check(answer, state["gold_answers"]),
                    "strict_tag_answer_present": tagged_answer is not None,
                    "strict_tag_answer_em": em_check(tagged_answer, state["gold_answers"]),
                    "second_search": "<search>" in response.lower(),
                    "response_clipped": finish_reason == "length",
                    "finish_reason": finish_reason,
                    "stop_reason": stop_reason,
                    "response_token_count": len(getattr(candidate, "token_ids", []) or []),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed[row["branch_key"]] = row
            handle.flush()

            rows = list(completed.values())
            report = summarize_outcomes(rows, len(branches), len(states))
            report.update(
                {
                    "status": "running" if len(completed) < len(branches) else "complete",
                    "model": args.model,
                    "input_states": args.input_states,
                    "output_jsonl": str(output_path),
                    "protocol": "fixed first query and cached top-k retrieval, then greedy single-query continuation",
                    "temperature": 0.0,
                    "max_response_tokens": args.max_response_tokens,
                    "max_document_chars": args.max_document_chars,
                    "seed": args.seed,
                }
            )
            dump_json(args.collection_report, report)
            log(
                f"Completed {len(completed)}/{len(branches)} branches; "
                f"EM={report['answer_em_rate']:.4f}, answer_rate={report['answer_rate']:.4f}"
            )

    # Keep this lookup alive until generation is complete; it also catches duplicate state ids.
    assert len(state_lookup) == len(states)


def roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(scores)
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    result = 0.0
    index = 0
    while index < len(labels):
        end = index + 1
        while end < len(labels) and scores[end] == scores[index]:
            end += 1
        group = labels[index:end]
        true_positives += int(group.sum())
        false_positives += int(len(group) - group.sum())
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        result += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return float(result)


def predictor_metrics(rows, predictor, label_key):
    selected = [row for row in rows if row.get(predictor) is not None]
    labels = [row[label_key] for row in selected]
    scores = [row[predictor] for row in selected]
    return {
        "pair_count": len(selected),
        "positive_count": int(sum(labels)),
        "positive_rate": float(np.mean(labels)) if labels else None,
        "spearman": spearman(scores, labels),
        "auroc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
    }


def metric_value(rows, predictor, label_key, metric):
    selected = [row for row in rows if row.get(predictor) is not None]
    labels = [row[label_key] for row in selected]
    scores = [row[predictor] for row in selected]
    if metric == "auroc":
        return roc_auc(labels, scores)
    if metric == "average_precision":
        return average_precision(labels, scores)
    if metric == "spearman":
        return spearman(scores, labels)
    raise ValueError(f"Unsupported bootstrap metric: {metric}")


def bootstrap_metric_difference(
    rows,
    left_predictor,
    right_predictor,
    label_key,
    metric,
    samples,
    seed,
):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["state_id"]].append(row)
    state_ids = list(grouped)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(samples):
        chosen = rng.choice(state_ids, size=len(state_ids), replace=True)
        sampled = [row for state_id in chosen for row in grouped[state_id]]
        left = metric_value(sampled, left_predictor, label_key, metric)
        right = metric_value(sampled, right_predictor, label_key, metric)
        if left is not None and right is not None and math.isfinite(left) and math.isfinite(right):
            differences.append(left - right)
    if not differences:
        return {"estimate_count": 0, "mean": None, "95_ci": [None, None], "p_gt_zero": None}
    return {
        "estimate_count": len(differences),
        "mean": float(np.mean(differences)),
        "95_ci": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
        "p_gt_zero": float(np.mean(np.asarray(differences) > 0)),
    }


def join_pairs(pair_rows, outcome_lookup):
    joined = []
    for pair in pair_rows:
        left = outcome_lookup.get(branch_key(pair["state_id"], pair["left_sample_index"]))
        right = outcome_lookup.get(branch_key(pair["state_id"], pair["right_sample_index"]))
        if left is None or right is None:
            continue
        row = dict(pair)
        row.update(
            {
                "left_answer": left["answer"],
                "right_answer": right["answer"],
                "left_answer_em": left["answer_em"],
                "right_answer_em": right["answer_em"],
                "left_answer_present": left["answer_present"],
                "right_answer_present": right["answer_present"],
                "left_strict_tag_answer_present": left["strict_tag_answer_present"],
                "right_strict_tag_answer_present": right["strict_tag_answer_present"],
                "left_second_search": left["second_search"],
                "right_second_search": right["second_search"],
                "return_difference": abs(left["answer_em"] - right["answer_em"]),
                "answer_disagreement": int(
                    left["normalized_answer"] != right["normalized_answer"]
                ),
                "evidence_hit_difference": abs(
                    int(left["gold_evidence_hit"]) - int(right["gold_evidence_hit"])
                ),
            }
        )
        joined.append(row)
    return joined


def state_summary(outcomes):
    grouped = defaultdict(list)
    for row in outcomes:
        grouped[row["state_id"]].append(row)
    eligible = [rows for rows in grouped.values() if len(rows) >= 2]
    variable = [rows for rows in eligible if len({row["answer_em"] for row in rows}) > 1]
    return {
        "state_count": len(grouped),
        "states_with_at_least_two_branches": len(eligible),
        "states_with_return_variation": len(variable),
        "return_variation_state_rate": len(variable) / len(eligible) if eligible else 0.0,
        "mean_branch_em": float(np.mean([row["answer_em"] for row in outcomes])),
        "mean_branch_answer_rate": float(
            np.mean([row["answer_present"] for row in outcomes])
        ),
        "mean_branch_strict_tag_answer_rate": float(
            np.mean([row["strict_tag_answer_present"] for row in outcomes])
        ),
        "mean_branch_strict_tag_em": float(
            np.mean([row["strict_tag_answer_em"] for row in outcomes])
        ),
        "mean_within_state_reward_std": float(
            np.mean([np.std([row["answer_em"] for row in rows]) for rows in eligible])
        )
        if eligible
        else None,
    }


def representative_examples(joined, outcome_lookup, limit):
    type_a = sorted(
        [
            row
            for row in joined
            if row["policy_embedding_distance"] >= 0
            and row["document_jaccard_distance"] == 0
            and row["return_difference"] == 0
        ],
        key=lambda row: row["policy_embedding_distance"],
        reverse=True,
    )[:limit]
    type_b = sorted(
        [
            row
            for row in joined
            if row["policy_embedding_distance"] <= 0.05
            and row["document_jaccard_distance"] >= 0.8
            and row["return_difference"] == 1
        ],
        key=lambda row: row["document_jaccard_distance"],
        reverse=True,
    )[:limit]

    def expand(row):
        row = dict(row)
        left = outcome_lookup[branch_key(row["state_id"], row["left_sample_index"])]
        right = outcome_lookup[branch_key(row["state_id"], row["right_sample_index"])]
        row["left_response"] = left["response"]
        row["right_response"] = right["response"]
        return row

    return {"behaviorally_redundant": [expand(row) for row in type_a], "causal_mismatch": [expand(row) for row in type_b]}


def analyze(args):
    outcomes = [rescore_outcome(row) for row in read_jsonl(args.input_outcomes)]
    outcome_lookup = {row["branch_key"]: row for row in outcomes}
    pair_rows = read_jsonl(args.input_pairs)
    joined = join_pairs(pair_rows, outcome_lookup)
    if not joined:
        raise RuntimeError("No Gate-A pairs could be joined to branch outcomes")

    predictors = LINGUISTIC_PREDICTORS + ENVIRONMENT_PREDICTORS
    label_summaries = {}
    for label_key in ["return_difference", "answer_disagreement", "evidence_hit_difference"]:
        label_summaries[label_key] = {
            predictor: predictor_metrics(joined, predictor, label_key)
            for predictor in predictors
        }

    primary_environment = label_summaries["return_difference"][PRIMARY_ENVIRONMENT_PREDICTOR]
    primary_language = label_summaries["return_difference"][PRIMARY_LANGUAGE_PREDICTOR]
    auc_gain = primary_environment["auroc"] - primary_language["auroc"]
    ap_gain = primary_environment["average_precision"] - primary_language["average_precision"]
    rho_gain = primary_environment["spearman"] - primary_language["spearman"]
    bootstrap_auc = bootstrap_metric_difference(
        joined,
        PRIMARY_ENVIRONMENT_PREDICTOR,
        PRIMARY_LANGUAGE_PREDICTOR,
        "return_difference",
        "auroc",
        args.bootstrap_samples,
        args.seed,
    )

    per_state = state_summary(outcomes)
    discordant_pairs = sum(row["return_difference"] for row in joined)
    sufficient = (
        per_state["states_with_return_variation"] >= args.min_variable_states
        and discordant_pairs >= args.min_discordant_pairs
    )
    gate_pass = bool(
        sufficient
        and auc_gain >= args.min_primary_auc_gain
        and bootstrap_auc["95_ci"][0] is not None
        and bootstrap_auc["95_ci"][0] > 0
    )

    best_language = max(
        LINGUISTIC_PREDICTORS,
        key=lambda key: label_summaries["return_difference"][key]["auroc"] or -math.inf,
    )
    best_environment = max(
        ENVIRONMENT_PREDICTORS,
        key=lambda key: label_summaries["return_difference"][key]["auroc"] or -math.inf,
    )
    best_language_auc = label_summaries["return_difference"][best_language]["auroc"]
    best_environment_auc = label_summaries["return_difference"][best_environment]["auroc"]
    best_observed_bootstrap_auc = bootstrap_metric_difference(
        joined,
        best_environment,
        best_language,
        "return_difference",
        "auroc",
        args.robustness_bootstrap_samples,
        args.seed + 1,
    )

    both_answered = [
        row
        for row in joined
        if row["left_answer_present"] and row["right_answer_present"]
    ]
    both_strict = [
        row
        for row in joined
        if row["left_strict_tag_answer_present"]
        and row["right_strict_tag_answer_present"]
    ]

    def robustness_summary(rows, bootstrap_seed):
        environment = predictor_metrics(
            rows, PRIMARY_ENVIRONMENT_PREDICTOR, "return_difference"
        )
        language = predictor_metrics(rows, best_language, "return_difference")
        gain = (
            environment["auroc"] - language["auroc"]
            if environment["auroc"] is not None and language["auroc"] is not None
            else None
        )
        bootstrap = bootstrap_metric_difference(
            rows,
            PRIMARY_ENVIRONMENT_PREDICTOR,
            best_language,
            "return_difference",
            "auroc",
            args.robustness_bootstrap_samples,
            bootstrap_seed,
        )
        return {
            "pair_count": len(rows),
            "discordant_return_pair_count": int(
                sum(row["return_difference"] for row in rows)
            ),
            "environment_predictor": PRIMARY_ENVIRONMENT_PREDICTOR,
            "environment_metrics": environment,
            "strongest_language_predictor_on_full_data": best_language,
            "language_metrics": language,
            "auroc_gain_environment_minus_language": gain,
            "state_cluster_bootstrap_auroc_gain": bootstrap,
        }

    report = {
        "scope": "NQ single-query Gate-B branch-to-terminal diagnostic",
        "protocol": "same fixed state and sampled query, cached top-k retrieval, greedy continuation to answer",
        "outcome_count": len(outcomes),
        "joined_pair_count": len(joined),
        "discordant_return_pair_count": int(discordant_pairs),
        "discordant_return_pair_rate": discordant_pairs / len(joined),
        "outcome_summary": summarize_outcomes(
            outcomes, len(outcomes), per_state["state_count"]
        ),
        "state_summary": per_state,
        "primary_comparison": {
            "environment_predictor": PRIMARY_ENVIRONMENT_PREDICTOR,
            "language_predictor": PRIMARY_LANGUAGE_PREDICTOR,
            "environment_metrics": primary_environment,
            "language_metrics": primary_language,
            "auroc_gain_environment_minus_language": auc_gain,
            "average_precision_gain_environment_minus_language": ap_gain,
            "spearman_gain_environment_minus_language": rho_gain,
            "state_cluster_bootstrap_auroc_gain": bootstrap_auc,
        },
        "exploratory_best_predictors": {
            "best_language": best_language,
            "best_language_metrics": label_summaries["return_difference"][best_language],
            "best_environment": best_environment,
            "best_environment_metrics": label_summaries["return_difference"][best_environment],
            "auroc_gain_best_environment_minus_best_language": (
                best_environment_auc - best_language_auc
            ),
            "state_cluster_bootstrap_auroc_gain": best_observed_bootstrap_auc,
        },
        "robustness": {
            "both_branches_have_semantic_answer": robustness_summary(
                both_answered, args.seed + 2
            ),
            "both_branches_have_strict_answer_tag": robustness_summary(
                both_strict, args.seed + 3
            ),
            "note": (
                "These conditioned subsets remove termination and format failures but "
                "can condition on a mediator, so they are robustness checks rather "
                "than the primary causal estimand."
            ),
        },
        "predictor_metrics": label_summaries,
        "predeclared_gate": {
            "min_variable_states": args.min_variable_states,
            "min_discordant_pairs": args.min_discordant_pairs,
            "min_primary_auc_gain": args.min_primary_auc_gain,
            "requires_cluster_bootstrap_95_ci_lower_gt_zero": True,
            "data_sufficient": sufficient,
            "gate_b_pass": gate_pass,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(output_dir / "summary.json", report)
    dump_json(
        output_dir / "representative_examples.json",
        representative_examples(joined, outcome_lookup, args.example_count),
    )
    with (output_dir / "pairs_with_outcomes.jsonl").open("w", encoding="utf-8") as handle:
        for row in joined:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(json.dumps(report, ensure_ascii=False, indent=2))


def self_test(_args):
    assert normalize_answer("The Michelangelo!") == "michelangelo"
    assert em_check("The Michelangelo!", ["Michelangelo"]) == 1
    assert extract_answer("x<answer>a</answer>y<answer>b</answer>") == ("b", "answer_tag")
    assert extract_answer("<think>x</think>\nAnswer: Michelangelo") == (
        "Michelangelo",
        "plain_answer",
    )
    assert abs(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) - 1.0) < 1e-12
    assert abs(roc_auc([0, 1], [0.5, 0.5]) - 0.5) < 1e-12
    assert abs(average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) - 1.0) < 1e-12
    log("self-test passed")


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.set_defaults(func=collect)
    collect_parser.add_argument("--input_states", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--output_jsonl", required=True)
    collect_parser.add_argument("--collection_report", required=True)
    collect_parser.add_argument("--state_offset", type=int, default=0)
    collect_parser.add_argument("--max_states", type=int)
    collect_parser.add_argument("--generation_batch_size", type=int, default=32)
    collect_parser.add_argument("--max_response_tokens", type=int, default=320)
    collect_parser.add_argument("--max_document_chars", type=int, default=1800)
    collect_parser.add_argument("--dtype", default="bfloat16")
    collect_parser.add_argument("--tensor_parallel_size", type=int, default=1)
    collect_parser.add_argument("--gpu_memory_utilization", type=float, default=0.72)
    collect_parser.add_argument("--max_model_len", type=int, default=4096)
    collect_parser.add_argument("--enforce_eager", action="store_true")
    collect_parser.add_argument("--seed", type=int, default=20260805)
    collect_parser.add_argument("--resume", action="store_true")

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.set_defaults(func=analyze)
    analyze_parser.add_argument("--input_outcomes", required=True)
    analyze_parser.add_argument("--input_pairs", required=True)
    analyze_parser.add_argument("--output_dir", required=True)
    analyze_parser.add_argument("--bootstrap_samples", type=int, default=500)
    analyze_parser.add_argument("--robustness_bootstrap_samples", type=int, default=300)
    analyze_parser.add_argument("--min_variable_states", type=int, default=100)
    analyze_parser.add_argument("--min_discordant_pairs", type=int, default=500)
    analyze_parser.add_argument("--min_primary_auc_gain", type=float, default=0.03)
    analyze_parser.add_argument("--example_count", type=int, default=10)
    analyze_parser.add_argument("--seed", type=int, default=20260805)

    test_parser = subparsers.add_parser("self-test")
    test_parser.set_defaults(func=self_test)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
