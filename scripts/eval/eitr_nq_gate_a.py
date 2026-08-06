#!/usr/bin/env python3
"""Collect and analyze the EITR Gate-A single-query diagnostic on NQ."""

import argparse
import hashlib
import inspect
import json
import math
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


SINGLE_QUERY_PROMPT = """You are a search-augmented reasoning agent.
Use <think>...</think> for reasoning and <search>...</search> for a search action.
Before answering, you must first issue exactly one concise search query.
Do not use multiple queries, the || separator, <plan>, or <information>.
The search engine will provide the information after your query.
Question: {question}
"""

WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def log(message):
    print(message, flush=True)


def jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=jsonable)
        handle.write("\n")


def normalize_question(question):
    question = " ".join(str(question).strip().split())
    if question and question[-1] != "?":
        question += "?"
    return question


def normalize_text(text):
    return " ".join(WORD_RE.findall(str(text).lower()))


def normalize_gold(gold):
    if gold is None:
        return []
    if hasattr(gold, "tolist"):
        gold = gold.tolist()
    if isinstance(gold, str):
        return [gold]
    return [str(item) for item in gold]


def filter_supported_kwargs(cls, kwargs):
    parameters = inspect.signature(cls).parameters
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def load_nq_rows(path, num_states, seed, start_index):
    import pandas as pd

    frame = pd.read_parquet(path)
    rows = []
    for index, item in frame.iterrows():
        rows.append(
            {
                "id": str(item.get("id") or f"nq_{index}"),
                "question": normalize_question(item["question"]),
                "gold_answers": normalize_gold(item.get("golden_answers")),
                "source_index": int(index),
            }
        )

    if start_index:
        rows = rows[start_index:]
    rng = random.Random(seed)
    rng.shuffle(rows)
    # Prefix generation occasionally answers without searching. Keep a reserve.
    reserve = min(len(rows), max(num_states, int(math.ceil(num_states * 1.35))))
    return rows[:reserve]


def build_chat_prompt(tokenizer, question):
    content = SINGLE_QUERY_PROMPT.format(question=question)
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return content


def clean_query(raw_text):
    text = str(raw_text).strip()
    text = text.split("||", 1)[0]
    text = text.split("</search>", 1)[0]
    text = text.split("\n", 1)[0]
    text = text.strip(" \t\r\n\"'`-:;")
    text = text.rstrip(">")
    text = " ".join(text.split())
    if not text or len(text) < 2 or len(text) > 256:
        return ""
    if "<" in text or ">" in text:
        return ""
    return text


def output_logprob(output):
    value = getattr(output, "cumulative_logprob", None)
    if value is None:
        return None
    return float(value)


def load_vllm(args):
    import transformers
    from vllm import LLM

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    kwargs = filter_supported_kwargs(
        LLM,
        {
            "model": args.model,
            "tokenizer": args.model,
            "trust_remote_code": True,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "enforce_eager": args.enforce_eager,
        },
    )
    log(f"Loading vLLM model from {args.model}")
    return tokenizer, LLM(**kwargs)


def make_sampling_params(**kwargs):
    from vllm import SamplingParams

    return SamplingParams(**filter_supported_kwargs(SamplingParams, kwargs))


def generate_prefixes(llm, prompts, max_tokens):
    params = make_sampling_params(
        temperature=0.0,
        max_tokens=max_tokens,
        stop=["<search>", "</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, params, use_tqdm=True)
    prefixes = []
    for output in outputs:
        candidate = output.outputs[0]
        text = candidate.text
        marker = text.find("<search>")
        if marker >= 0:
            prefixes.append(text[: marker + len("<search>")])
        elif getattr(candidate, "stop_reason", None) == "<search>":
            prefixes.append(text + "<search>")
        else:
            prefixes.append("")
    return prefixes


def sample_queries(llm, state_prompts, args):
    params = make_sampling_params(
        n=args.samples_per_state,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_query_tokens,
        stop=["</search>", "||", "<"],
        include_stop_str_in_output=False,
        logprobs=1,
        seed=args.seed,
    )
    collected = []
    for begin in range(0, len(state_prompts), args.generation_batch_size):
        batch = state_prompts[begin : begin + args.generation_batch_size]
        outputs = llm.generate(batch, params, use_tqdm=False)
        collected.extend(outputs)
        log(f"Sampled query actions for {len(collected)}/{len(state_prompts)} states")
    return collected


def retrieve_queries(url, queries, topk, timeout, batch_size, retries):
    import requests

    all_results = []
    for begin in range(0, len(queries), batch_size):
        batch = queries[begin : begin + batch_size]
        payload = {"queries": batch, "topk": topk, "return_scores": True}
        error = None
        for attempt in range(retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=timeout)
                response.raise_for_status()
                result = response.json()["result"]
                if len(result) != len(batch):
                    raise RuntimeError(
                        f"Retriever returned {len(result)} rows for {len(batch)} queries"
                    )
                all_results.extend(result)
                error = None
                break
            except Exception as exc:  # noqa: BLE001 - retain the HTTP error context
                error = exc
                if attempt < retries:
                    time.sleep(2 ** attempt)
        if error is not None:
            raise RuntimeError(f"Retriever failed at query offset {begin}: {error}")
        log(f"Retrieved evidence for {len(all_results)}/{len(queries)} queries")
    return all_results


def document_id(document):
    value = document.get("id")
    if value is not None:
        return str(value)
    contents = str(document.get("contents") or document.get("text") or "")
    return "sha1:" + hashlib.sha1(contents.encode("utf-8")).hexdigest()


def normalize_retrieval(items):
    rows = []
    for rank, item in enumerate(items, start=1):
        document = item.get("document", item)
        contents = str(document.get("contents") or document.get("text") or "")
        title = str(document.get("title") or contents.split("\n", 1)[0])
        rows.append(
            {
                "rank": rank,
                "doc_id": document_id(document),
                "title": title,
                "contents": contents,
                "score": float(item.get("score", 0.0)),
            }
        )
    return rows


def contains_gold(retrieval, gold_answers):
    evidence = normalize_text(" ".join(item["contents"] for item in retrieval))
    return any(normalize_text(gold) and normalize_text(gold) in evidence for gold in gold_answers)


def collect(args):
    rows = load_nq_rows(args.input_parquet, args.num_states, args.seed, args.start_index)
    tokenizer, llm = load_vllm(args)
    prompts = [build_chat_prompt(tokenizer, row["question"]) for row in rows]
    prefixes = generate_prefixes(llm, prompts, args.max_prefix_tokens)

    states = []
    skipped = []
    for row, prompt, prefix in zip(rows, prompts, prefixes):
        if not prefix:
            skipped.append({"id": row["id"], "reason": "no_search_boundary"})
            continue
        states.append({**row, "prompt": prompt, "prefix": prefix})
        if len(states) >= args.num_states:
            break
    if len(states) < args.num_states:
        raise RuntimeError(
            f"Only {len(states)}/{args.num_states} candidate states reached <search>"
        )

    state_prompts = [item["prompt"] + item["prefix"] for item in states]
    generated = sample_queries(llm, state_prompts, args)

    flat_queries = []
    query_slots = []
    for state_index, output in enumerate(generated):
        for sample_index, candidate in enumerate(output.outputs):
            query = clean_query(candidate.text)
            slot = {
                "sample_index": sample_index,
                "raw_text": candidate.text,
                "query": query,
                "valid": bool(query),
                "cumulative_logprob": output_logprob(candidate),
                "token_count": len(getattr(candidate, "token_ids", []) or []),
                "retrieval": [],
            }
            states[state_index].setdefault("samples", []).append(slot)
            if query:
                query_slots.append(slot)
                flat_queries.append(query)

    retrieval_results = retrieve_queries(
        args.retriever_url,
        flat_queries,
        args.topk,
        args.retriever_timeout,
        args.retrieval_batch_size,
        args.retriever_retries,
    )
    for slot, result in zip(query_slots, retrieval_results):
        slot["retrieval"] = normalize_retrieval(result)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid_queries = 0
    unique_counts = []
    evidence_hits = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for state in states:
            for sample in state["samples"]:
                if sample["valid"]:
                    valid_queries += 1
                    sample["gold_evidence_hit"] = contains_gold(
                        sample["retrieval"], state["gold_answers"]
                    )
                    evidence_hits += int(sample["gold_evidence_hit"])
                else:
                    sample["gold_evidence_hit"] = False
            unique_counts.append(
                len({normalize_text(item["query"]) for item in state["samples"] if item["valid"]})
            )
            handle.write(json.dumps(state, ensure_ascii=False, default=jsonable) + "\n")

    report = {
        "protocol": "same deterministic prefix ending at <search>; K independent single-query continuations",
        "model": args.model,
        "input_parquet": args.input_parquet,
        "retriever_url": args.retriever_url,
        "num_states": len(states),
        "samples_per_state": args.samples_per_state,
        "expected_queries": len(states) * args.samples_per_state,
        "valid_queries": valid_queries,
        "invalid_queries": len(states) * args.samples_per_state - valid_queries,
        "mean_unique_queries_per_state": statistics.mean(unique_counts),
        "gold_evidence_hit_rate": evidence_hits / valid_queries if valid_queries else 0.0,
        "skipped_prefix_states": skipped,
        "topk": args.topk,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "output_jsonl": str(output_path),
    }
    dump_json(args.collection_report, report)
    log(json.dumps(report, ensure_ascii=False, indent=2))


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tokenize_words(text):
    return WORD_RE.findall(str(text).lower())


def jaccard_distance(left, right):
    left = set(left)
    right = set(right)
    if not left and not right:
        return 0.0
    return 1.0 - len(left & right) / len(left | right)


def normalized_edit_distance(left, right):
    left = str(left)
    right = str(right)
    if not left and not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + int(char_left != char_right),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right), 1)


def cosine_distance(left, right):
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0:
        return 0.0
    return float(np.clip(1.0 - float(np.dot(left, right)) / denom, 0.0, 2.0))


def rbo_score(left, right, p=0.9):
    depth = max(len(left), len(right))
    if depth == 0:
        return 1.0
    left_seen = set()
    right_seen = set()
    weighted = 0.0
    agreement = 0.0
    for index in range(depth):
        if index < len(left):
            left_seen.add(left[index])
        if index < len(right):
            right_seen.add(right[index])
        agreement = len(left_seen & right_seen) / (index + 1)
        weighted += (1.0 - p) * agreement * (p ** index)
    return float(weighted + agreement * (p ** depth))


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


def js_distance(left_rows, right_rows):
    ids = sorted({item["doc_id"] for item in left_rows + right_rows})
    if not ids:
        return 0.0
    left_weights = softmax([item["score"] for item in left_rows])
    right_weights = softmax([item["score"] for item in right_rows])
    left_map = {item["doc_id"]: left_weights[index] for index, item in enumerate(left_rows)}
    right_map = {item["doc_id"]: right_weights[index] for index, item in enumerate(right_rows)}
    left = np.asarray([left_map.get(doc_id, 0.0) for doc_id in ids])
    right = np.asarray([right_map.get(doc_id, 0.0) for doc_id in ids])
    middle = 0.5 * (left + right)

    def kl_divergence(source, target):
        mask = source > 0
        return float(np.sum(source[mask] * np.log(source[mask] / target[mask])))

    divergence = 0.5 * kl_divergence(left, middle) + 0.5 * kl_divergence(right, middle)
    return float(math.sqrt(max(divergence, 0.0)))


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + end - 1) / 2.0 + 1.0
        position = end
    return ranks


def spearman(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return None
    left_rank = rankdata(left[mask])
    right_rank = rankdata(right[mask])
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def encode_texts(model_path, texts, prefixes, device, batch_size, max_length):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if len(texts) != len(prefixes):
        raise ValueError("texts and prefixes must have the same length")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_kwargs = {"trust_remote_code": True}
    if str(device).startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModel.from_pretrained(model_path, **model_kwargs)
    model.eval().to(device)

    outputs = []
    with torch.no_grad():
        for begin in range(0, len(texts), batch_size):
            batch = [
                f"{prefixes[index]}{texts[index]}"
                for index in range(begin, min(begin + batch_size, len(texts)))
            ]
            tokens = tokenizer(
                batch,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            result = model(**tokens, return_dict=True)
            hidden = result.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).bool()
            pooled = hidden.masked_fill(~mask, 0.0).sum(dim=1) / mask.sum(dim=1)
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            outputs.append(pooled.cpu().numpy())
            log(f"Embedded {min(begin + batch_size, len(texts))}/{len(texts)} texts")
    embeddings = np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0))
    del model, tokenizer
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return embeddings


def weighted_evidence_vector(retrieval, document_embeddings):
    available = [item for item in retrieval if item["doc_id"] in document_embeddings]
    if not available:
        return None
    weights = softmax([item["score"] for item in available])
    vectors = np.stack([document_embeddings[item["doc_id"]] for item in available])
    vector = np.sum(vectors * weights[:, None], axis=0)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def bootstrap_correlation(pair_rows, state_ids, left_key, right_key, samples, seed):
    grouped = defaultdict(list)
    for row in pair_rows:
        grouped[row["state_id"]].append(row)
    rng = np.random.default_rng(seed)
    estimates = []
    state_ids = list(state_ids)
    for _ in range(samples):
        chosen = rng.choice(state_ids, size=len(state_ids), replace=True)
        rows = [row for state_id in chosen for row in grouped[state_id]]
        value = spearman([row[left_key] for row in rows], [row[right_key] for row in rows])
        if value is not None:
            estimates.append(value)
    if not estimates:
        return [None, None]
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def summarize_correlations(pair_rows, state_ids, args):
    linguistic = [
        "policy_embedding_distance",
        "token_jaccard_distance",
        "normalized_edit_distance",
        "normalized_logprob_gap",
        "retriever_query_embedding_distance",
    ]
    behavioral = [
        "evidence_embedding_distance",
        "document_jaccard_distance",
        "rbo_distance",
        "retrieval_score_js_distance",
    ]
    summary = {}
    for left_key in linguistic:
        summary[left_key] = {}
        for right_key in behavioral:
            rows = [
                row
                for row in pair_rows
                if row.get(left_key) is not None and row.get(right_key) is not None
            ]
            rho = spearman([row[left_key] for row in rows], [row[right_key] for row in rows])
            state_values = []
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["state_id"]].append(row)
            for state_rows in grouped.values():
                value = spearman(
                    [row[left_key] for row in state_rows],
                    [row[right_key] for row in state_rows],
                )
                if value is not None:
                    state_values.append(value)
            ci = bootstrap_correlation(
                rows,
                state_ids,
                left_key,
                right_key,
                args.bootstrap_samples,
                args.seed,
            )
            summary[left_key][right_key] = {
                "overall_spearman": rho,
                "state_cluster_bootstrap_95_ci": ci,
                "state_spearman_median": (
                    float(np.median(state_values)) if state_values else None
                ),
                "state_spearman_iqr": (
                    [float(np.quantile(state_values, 0.25)), float(np.quantile(state_values, 0.75))]
                    if state_values
                    else [None, None]
                ),
                "pair_count": len(rows),
                "state_count_with_defined_rho": len(state_values),
            }
    return summary


def nearest_neighbor_agreement(states, query_embeddings):
    agreements = []
    ranks = []
    eligible = 0
    for state in states:
        samples = [item for item in state["samples"] if item["valid"]]
        for anchor in samples:
            candidates = [
                item
                for item in samples
                if item["sample_index"] != anchor["sample_index"]
                and normalize_text(item["query"]) != normalize_text(anchor["query"])
            ]
            if len(candidates) < 2:
                continue
            anchor_key = (state["id"], anchor["sample_index"])
            if anchor_key not in query_embeddings:
                continue
            lexical = []
            behavioral = []
            valid_candidates = []
            for candidate in candidates:
                key = (state["id"], candidate["sample_index"])
                if key not in query_embeddings:
                    continue
                valid_candidates.append(candidate)
                lexical.append(cosine_distance(query_embeddings[anchor_key], query_embeddings[key]))
                behavioral.append(
                    jaccard_distance(
                        [item["doc_id"] for item in anchor["retrieval"]],
                        [item["doc_id"] for item in candidate["retrieval"]],
                    )
                )
            if len(valid_candidates) < 2:
                continue
            eligible += 1
            lexical_order = np.argsort(lexical)
            behavior_order = np.argsort(behavioral)
            lexical_best = int(lexical_order[0])
            min_behavior = behavioral[int(behavior_order[0])]
            behavior_best = {
                index for index, value in enumerate(behavioral) if value == min_behavior
            }
            agreements.append(int(lexical_best in behavior_best))
            lexical_behavior = behavioral[lexical_best]
            ranks.append(1 + sum(value < lexical_behavior for value in behavioral))
    return {
        "eligible_anchors": eligible,
        "nearest_neighbor_agreement_rate_allowing_behavior_ties": (
            float(np.mean(agreements)) if agreements else None
        ),
        "language_nn_median_rank_in_document_space": float(np.median(ranks)) if ranks else None,
        "random_agreement_reference": None,
    }


def analyze(args):
    states = read_jsonl(args.input_jsonl)
    valid_slots = []
    documents = {}
    for state in states:
        for sample in state.get("samples", []):
            if not sample.get("valid") or not sample.get("retrieval"):
                continue
            valid_slots.append((state, sample))
            for document in sample["retrieval"]:
                documents.setdefault(document["doc_id"], document["contents"])

    query_texts = [sample["query"] for _, sample in valid_slots]
    policy_query_vectors = encode_texts(
        args.language_embedding_model,
        query_texts,
        [""] * len(query_texts),
        args.embedding_device,
        args.language_embedding_batch_size,
        args.query_embedding_max_length,
    )
    policy_query_embeddings = {
        (state["id"], sample["sample_index"]): policy_query_vectors[index]
        for index, (state, sample) in enumerate(valid_slots)
    }

    retriever_query_vectors = encode_texts(
        args.embedding_model,
        query_texts,
        ["query: "] * len(query_texts),
        args.embedding_device,
        args.embedding_batch_size,
        args.query_embedding_max_length,
    )
    retriever_query_embeddings = {
        (state["id"], sample["sample_index"]): retriever_query_vectors[index]
        for index, (state, sample) in enumerate(valid_slots)
    }

    document_ids = list(documents)
    document_vectors = encode_texts(
        args.embedding_model,
        [documents[item] for item in document_ids],
        ["passage: "] * len(document_ids),
        args.embedding_device,
        args.embedding_batch_size,
        args.document_embedding_max_length,
    )
    document_embeddings = {
        doc_id: document_vectors[index] for index, doc_id in enumerate(document_ids)
    }
    evidence_vectors = {}
    for state, sample in valid_slots:
        key = (state["id"], sample["sample_index"])
        vector = weighted_evidence_vector(sample["retrieval"], document_embeddings)
        if vector is not None:
            evidence_vectors[key] = vector

    pair_rows = []
    state_pair_counts = defaultdict(int)
    for state in states:
        samples = [item for item in state.get("samples", []) if item.get("valid")]
        for left_index in range(len(samples)):
            for right_index in range(left_index + 1, len(samples)):
                left = samples[left_index]
                right = samples[right_index]
                left_key = (state["id"], left["sample_index"])
                right_key = (state["id"], right["sample_index"])
                if left_key not in policy_query_embeddings or right_key not in policy_query_embeddings:
                    continue
                left_docs = left["retrieval"]
                right_docs = right["retrieval"]
                left_ids = [item["doc_id"] for item in left_docs]
                right_ids = [item["doc_id"] for item in right_docs]
                left_logprob = left.get("cumulative_logprob")
                right_logprob = right.get("cumulative_logprob")
                left_tokens = max(int(left.get("token_count") or 0), 1)
                right_tokens = max(int(right.get("token_count") or 0), 1)
                logprob_gap = None
                if left_logprob is not None and right_logprob is not None:
                    logprob_gap = abs(left_logprob / left_tokens - right_logprob / right_tokens)
                row = {
                    "state_id": state["id"],
                    "question": state["question"],
                    "left_sample_index": left["sample_index"],
                    "right_sample_index": right["sample_index"],
                    "left_query": left["query"],
                    "right_query": right["query"],
                    "same_normalized_query": normalize_text(left["query"]) == normalize_text(right["query"]),
                    "token_jaccard_distance": jaccard_distance(
                        tokenize_words(left["query"]), tokenize_words(right["query"])
                    ),
                    "normalized_edit_distance": normalized_edit_distance(
                        normalize_text(left["query"]), normalize_text(right["query"])
                    ),
                    "policy_embedding_distance": cosine_distance(
                        policy_query_embeddings[left_key], policy_query_embeddings[right_key]
                    ),
                    "retriever_query_embedding_distance": cosine_distance(
                        retriever_query_embeddings[left_key], retriever_query_embeddings[right_key]
                    ),
                    "normalized_logprob_gap": logprob_gap,
                    "document_jaccard_distance": jaccard_distance(left_ids, right_ids),
                    "rbo_distance": 1.0 - rbo_score(left_ids, right_ids, p=args.rbo_p),
                    "retrieval_score_js_distance": js_distance(left_docs, right_docs),
                    "evidence_embedding_distance": (
                        cosine_distance(evidence_vectors[left_key], evidence_vectors[right_key])
                        if left_key in evidence_vectors and right_key in evidence_vectors
                        else None
                    ),
                    "same_document_set": set(left_ids) == set(right_ids),
                    "same_ranked_documents": left_ids == right_ids,
                    "left_gold_evidence_hit": bool(left.get("gold_evidence_hit")),
                    "right_gold_evidence_hit": bool(right.get("gold_evidence_hit")),
                }
                pair_rows.append(row)
                state_pair_counts[state["id"]] += 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / "pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as handle:
        for row in pair_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=jsonable) + "\n")

    state_ids = sorted(state_pair_counts)
    correlations = summarize_correlations(pair_rows, state_ids, args)
    primary_left = "policy_embedding_distance"
    primary_right = "document_jaccard_distance"
    left_values = np.asarray([row[primary_left] for row in pair_rows], dtype=np.float64)
    right_values = np.asarray([row[primary_right] for row in pair_rows], dtype=np.float64)
    left_q25, left_q75 = np.quantile(left_values, [0.25, 0.75])
    right_q25, right_q75 = np.quantile(right_values, [0.25, 0.75])
    type_a = [
        row for row in pair_rows
        if row[primary_left] >= left_q75 and row[primary_right] <= right_q25
    ]
    type_b = [
        row for row in pair_rows
        if row[primary_left] <= left_q25 and row[primary_right] >= right_q75
    ]
    concrete_type_a = [
        row for row in pair_rows
        if row["token_jaccard_distance"] >= 0.5 and row["same_document_set"]
    ]
    concrete_type_b = [
        row for row in pair_rows
        if row["token_jaccard_distance"] <= 0.25
        and row["document_jaccard_distance"] >= 0.8
    ]
    mismatch_state_coverage = {
        "type_a_state_count": len({row["state_id"] for row in type_a}),
        "type_b_state_count": len({row["state_id"] for row in type_b}),
        "concrete_type_a_state_count": len({row["state_id"] for row in concrete_type_a}),
        "concrete_type_b_state_count": len({row["state_id"] for row in concrete_type_b}),
        "any_concrete_mismatch_state_count": len(
            {row["state_id"] for row in concrete_type_a + concrete_type_b}
        ),
    }
    mismatch_state_coverage.update(
        {
            key.replace("_count", "_rate"): value / len(states)
            for key, value in list(mismatch_state_coverage.items())
        }
    )
    nn_summary = nearest_neighbor_agreement(states, policy_query_embeddings)
    if args.samples_per_state_hint > 2:
        nn_summary["random_agreement_reference"] = 1.0 / (args.samples_per_state_hint - 1)

    primary = correlations[primary_left][primary_right]
    confidence = primary["state_cluster_bootstrap_95_ci"]
    preliminary_pass = (
        primary["overall_spearman"] is not None
        and abs(primary["overall_spearman"]) < args.max_abs_primary_rho
        and confidence[0] is not None
        and max(abs(confidence[0]), abs(confidence[1])) < args.max_abs_primary_ci
    )
    summary = {
        "scope": "NQ single-query Gate-A pilot; not multi-dataset paper evidence",
        "state_count": len(states),
        "state_count_with_pairs": len(state_ids),
        "valid_query_count": len(valid_slots),
        "unique_document_count": len(documents),
        "pair_count": len(pair_rows),
        "mean_pairs_per_state": statistics.mean(state_pair_counts.values()),
        "correlations": correlations,
        "primary_metric_pair": [primary_left, primary_right],
        "primary_result": primary,
        "quartile_thresholds": {
            "policy_embedding_distance_q25_q75": [float(left_q25), float(left_q75)],
            "document_jaccard_distance_q25_q75": [float(right_q25), float(right_q75)],
        },
        "type_a_high_language_low_environment": {
            "count": len(type_a),
            "rate": len(type_a) / len(pair_rows),
        },
        "type_b_low_language_high_environment": {
            "count": len(type_b),
            "rate": len(type_b) / len(pair_rows),
        },
        "concrete_type_a_token_far_same_doc_set": {
            "count": len(concrete_type_a),
            "rate": len(concrete_type_a) / len(pair_rows),
        },
        "concrete_type_b_token_close_doc_far": {
            "count": len(concrete_type_b),
            "rate": len(concrete_type_b) / len(pair_rows),
        },
        "mismatch_state_coverage": mismatch_state_coverage,
        "same_document_set_rate": float(np.mean([row["same_document_set"] for row in pair_rows])),
        "same_ranked_documents_rate": float(
            np.mean([row["same_ranked_documents"] for row in pair_rows])
        ),
        "nearest_neighbor": nn_summary,
        "preliminary_gate_a_pass": preliminary_pass,
        "pass_rule": {
            "max_abs_primary_rho": args.max_abs_primary_rho,
            "max_abs_primary_bootstrap_ci_endpoint": args.max_abs_primary_ci,
        },
        "important_limit": (
            "The normalized logprob gap is a sampled sequence-probability proxy, "
            "not an old-vs-new policy KL. A checkpoint-pair experiment is still required."
        ),
        "metric_note": (
            "The E5 query-to-evidence correlation is a coupled retriever-geometry control. "
            "The primary result uses Qwen policy embeddings versus document-set Jaccard."
        ),
        "artifacts": {
            "input": args.input_jsonl,
            "pairs": str(pairs_path),
        },
    }
    dump_json(output_dir / "summary.json", summary)

    examples_a = sorted(
        type_a,
        key=lambda row: row[primary_left] - row[primary_right],
        reverse=True,
    )[:5]
    examples_b = sorted(
        type_b,
        key=lambda row: row[primary_right] - row[primary_left],
        reverse=True,
    )[:5]
    dump_json(output_dir / "mismatch_examples.json", {"type_a": examples_a, "type_b": examples_b})

    report_lines = [
        "# EITR Gate A: NQ Single-Query Pilot",
        "",
        f"- States: {len(states)}",
        f"- Valid queries: {len(valid_slots)}",
        f"- Query pairs: {len(pair_rows)}",
        f"- Primary Spearman rho: {primary['overall_spearman']:.4f}",
        f"- State-cluster bootstrap 95% CI: [{confidence[0]:.4f}, {confidence[1]:.4f}]",
        f"- Type A rate (embedding quartiles): {len(type_a) / len(pair_rows):.2%}",
        f"- Type B rate (embedding quartiles): {len(type_b) / len(pair_rows):.2%}",
        f"- Token-far but exact-same-doc-set rate: {len(concrete_type_a) / len(pair_rows):.2%}",
        "- States with at least one concrete mismatch: "
        f"{mismatch_state_coverage['any_concrete_mismatch_state_rate']:.2%}",
        "- Language/retrieval nearest-neighbor agreement: "
        f"{nn_summary['nearest_neighbor_agreement_rate_allowing_behavior_ties']:.2%}",
        f"- Preliminary Gate A pass: {preliminary_pass}",
        "",
        "This is an NQ pilot. A paper claim still requires multiple QA datasets, retrievers,",
        "and an old-vs-new checkpoint experiment for true policy-drift comparison.",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    log(json.dumps(summary, ensure_ascii=False, indent=2, default=jsonable))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--input_parquet", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--retriever_url", default="http://127.0.0.1:8000/retrieve")
    collect_parser.add_argument("--output_jsonl", required=True)
    collect_parser.add_argument("--collection_report", required=True)
    collect_parser.add_argument("--num_states", type=int, default=20)
    collect_parser.add_argument("--samples_per_state", type=int, default=8)
    collect_parser.add_argument("--start_index", type=int, default=0)
    collect_parser.add_argument("--seed", type=int, default=20260805)
    collect_parser.add_argument("--temperature", type=float, default=1.0)
    collect_parser.add_argument("--top_p", type=float, default=1.0)
    collect_parser.add_argument("--topk", type=int, default=3)
    collect_parser.add_argument("--max_prefix_tokens", type=int, default=256)
    collect_parser.add_argument("--max_query_tokens", type=int, default=48)
    collect_parser.add_argument("--generation_batch_size", type=int, default=32)
    collect_parser.add_argument("--retrieval_batch_size", type=int, default=256)
    collect_parser.add_argument("--retriever_timeout", type=int, default=180)
    collect_parser.add_argument("--retriever_retries", type=int, default=2)
    collect_parser.add_argument("--dtype", default="bfloat16")
    collect_parser.add_argument("--tensor_parallel_size", type=int, default=1)
    collect_parser.add_argument("--gpu_memory_utilization", type=float, default=0.65)
    collect_parser.add_argument("--max_model_len", type=int, default=4096)
    collect_parser.add_argument("--enforce_eager", action="store_true")
    collect_parser.set_defaults(func=collect)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--input_jsonl", required=True)
    analyze_parser.add_argument("--output_dir", required=True)
    analyze_parser.add_argument("--embedding_model", required=True)
    analyze_parser.add_argument("--language_embedding_model", required=True)
    analyze_parser.add_argument("--embedding_device", default="cuda:0")
    analyze_parser.add_argument("--embedding_batch_size", type=int, default=128)
    analyze_parser.add_argument("--language_embedding_batch_size", type=int, default=32)
    analyze_parser.add_argument("--query_embedding_max_length", type=int, default=128)
    analyze_parser.add_argument("--document_embedding_max_length", type=int, default=256)
    analyze_parser.add_argument("--bootstrap_samples", type=int, default=500)
    analyze_parser.add_argument("--samples_per_state_hint", type=int, default=8)
    analyze_parser.add_argument("--rbo_p", type=float, default=0.9)
    analyze_parser.add_argument("--max_abs_primary_rho", type=float, default=0.3)
    analyze_parser.add_argument("--max_abs_primary_ci", type=float, default=0.4)
    analyze_parser.add_argument("--seed", type=int, default=20260805)
    analyze_parser.set_defaults(func=analyze)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 - command-line diagnostics need context
        log(f"ERROR: {exc}")
        raise
