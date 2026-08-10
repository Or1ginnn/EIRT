"""Environment-induced trust-region utilities for search-agent GRPO.

The primary estimator samples query-only probes from one exact search-state
token prefix. The retriever remains a black box: gradients flow through current
query log probabilities, while cached top-k document distributions are
constants. Sibling rollouts remain available only as an ablation fallback.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


EITR_BATCH_KEYS = (
    "eitr_probe_input_ids",
    "eitr_probe_attention_mask",
    "eitr_probe_position_ids",
    "eitr_probe_responses",
    "eitr_probe_response_mask",
    "eitr_probe_old_seq_logp",
    "eitr_probe_doc_probs",
    "eitr_probe_valid",
    "eitr_state_slot",
    "eitr_state_valid",
)

EITR_MODES = ("off", "probe_only", "eitr")
SAME_BATCH_SCALE_DIAGNOSTIC_LRS = (1e-5, 3e-5, 1e-4)


def _config_value(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if getter is not None:
        return getter(key, default)
    return getattr(config, key, default)


def resolve_eitr_mode(config: Any) -> str:
    """Resolve the Phase-2 mode while preserving the old ``enabled`` switch.

    ``mode`` is authoritative when it is set.  A missing/null mode falls back
    to the Gate-C ``enabled`` boolean so old launch commands still select the
    EITR loss instead of silently running the baseline.
    """
    configured_mode = _config_value(config, "mode", None)
    if configured_mode is None:
        return "eitr" if bool(_config_value(config, "enabled", False)) else "off"
    mode = str(configured_mode).strip().lower()
    if mode not in EITR_MODES:
        raise ValueError(
            f"Unsupported EITR mode={configured_mode!r}; expected one of {EITR_MODES}"
        )
    return mode


def same_batch_scale_diagnostic_lrs(config: Any) -> Tuple[float, ...]:
    """Return the fixed, debug-only scale sweep when explicitly enabled."""
    enabled = _config_value(config, "same_batch_scale_diagnostic", False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() == "true"
    return SAME_BATCH_SCALE_DIAGNOSTIC_LRS if bool(enabled) else ()


def eitr_update_direction_diagnostic_enabled(config: Any) -> bool:
    """Enable the one-shot +/- gradient direction audit only when requested."""
    enabled = _config_value(config, "update_direction_diagnostic", False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() == "true"
    return bool(enabled)


def eitr_score_path_noop_direction_audit_enabled(config: Any) -> bool:
    """Enable the default-off one-batch score-path audit only when requested."""
    enabled = _config_value(config, "score_path_noop_direction_audit", False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() == "true"
    return bool(enabled)


def same_batch_cache_signature(batch: Mapping[str, torch.Tensor]) -> Tuple[Tuple[Any, ...], ...]:
    """Describe cached EITR tensors without copying or mutating them."""
    return tuple(
        (
            key,
            int(batch[key].data_ptr()),
            tuple(batch[key].shape),
            str(batch[key].dtype),
        )
        for key in EITR_BATCH_KEYS
    )


def cached_probe_fingerprint(batches: Sequence[Mapping[str, torch.Tensor]]) -> str:
    """Content hash every cached field consumed by the EITR drift scorer."""
    digest = hashlib.sha256()
    for batch_index, batch in enumerate(batches):
        digest.update(f"batch:{batch_index}".encode())
        for key in EITR_BATCH_KEYS:
            if key not in batch:
                raise KeyError(f"Missing cached EITR probe field: {key}")
            value = batch[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Cached EITR probe field {key} is not a tensor")
            cpu_value = value.detach().cpu().contiguous()
            digest.update(key.encode())
            digest.update(str(cpu_value.dtype).encode())
            digest.update(str(tuple(cpu_value.shape)).encode())
            digest.update(cpu_value.numpy().tobytes())
    return digest.hexdigest()


def score_path_audit_event_sequence(grpo_step_count: int) -> Tuple[str, ...]:
    """Return the strict ordering enforced by the one-batch audit."""
    if grpo_step_count <= 0:
        raise ValueError("The audit requires at least one completed GRPO step")
    return (
        "probe_old_logp@v0",
        *(f"GRPO_STEP_{step}" for step in range(1, grpo_step_count + 1)),
        f"D_PRE_ALL_STATES@v{grpo_step_count}",
        "EITR_SGD_STEP_1",
        f"D_POST_ALL_STATES@v{grpo_step_count + 1}",
    )


def same_batch_sgd_candidates(
    parameters: Sequence[torch.Tensor],
    gradients: Sequence[Optional[torch.Tensor]],
    learning_rates: Sequence[float] = SAME_BATCH_SCALE_DIAGNOSTIC_LRS,
) -> Dict[float, Tuple[torch.Tensor, ...]]:
    """Pure CPU test helper proving candidate updates do not accumulate."""
    if len(parameters) != len(gradients):
        raise ValueError("parameters and gradients must have identical lengths")
    return {
        float(learning_rate): tuple(
            parameter.detach().clone()
            if gradient is None
            else parameter.detach().clone().add_(gradient, alpha=-float(learning_rate))
            for parameter, gradient in zip(parameters, gradients)
        )
        for learning_rate in learning_rates
    }


def directional_parameter_candidates(
    parameters: Sequence[torch.Tensor],
    gradients: Sequence[Optional[torch.Tensor]],
    epsilon: float,
) -> Dict[str, Tuple[torch.Tensor, ...]]:
    """Return exact +/- gradient candidates from an immutable common start."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if len(parameters) != len(gradients):
        raise ValueError("parameters and gradients must have identical lengths")
    return {
        direction: tuple(
            parameter.detach().clone()
            if gradient is None
            else parameter.detach().clone().add_(gradient, alpha=sign * epsilon)
            for parameter, gradient in zip(parameters, gradients)
        )
        for direction, sign in (("minus", -1.0), ("plus", 1.0))
    }


def coverage_weighted_state_scale(
    valid_state_count: int,
    global_rollout_state_count: float,
    *,
    world_size: int = 1,
) -> float:
    """Scale a local active-state mean into a global per-rollout mean.

    FSDP averages gradients across ranks. Multiplying the local active-state
    mean by ``local_active * world_size / global_rollouts`` therefore produces
    ``sum(mask * D_env) / global_rollouts`` after distributed reduction.
    """
    valid_state_count = int(valid_state_count)
    global_rollout_state_count = float(global_rollout_state_count)
    world_size = int(world_size)
    if valid_state_count < 0:
        raise ValueError("valid_state_count must be non-negative")
    if global_rollout_state_count < 0:
        raise ValueError("global_rollout_state_count must be non-negative")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if global_rollout_state_count == 0:
        return 0.0
    return valid_state_count * world_size / global_rollout_state_count


def rollout_averaged_env_drift(js_sum: float, global_rollout_state_count: float) -> float:
    """Aggregate V6 environment drift with the full rollout denominator."""
    js_sum = float(js_sum)
    global_rollout_state_count = float(global_rollout_state_count)
    if global_rollout_state_count < 0:
        raise ValueError("global_rollout_state_count must be non-negative")
    if global_rollout_state_count == 0:
        return 0.0
    return js_sum / global_rollout_state_count


def should_run_post_diagnostic(
    outer_update_step: int,
    frequency: int,
    *,
    correction_applied: bool,
) -> bool:
    """Schedule the extra post-correction query re-score."""
    outer_update_step = int(outer_update_step)
    frequency = int(frequency)
    if frequency < 0:
        raise ValueError("post diagnostic frequency must be non-negative")
    return bool(
        correction_applied
        and frequency > 0
        and outer_update_step > 0
        and outer_update_step % frequency == 0
    )


def eitr_probe_enabled_for_pass(
    config: Any,
    pass_index: int,
    *,
    grpo_passes: int = 1,
) -> bool:
    """Whether a global optimization pass is an EITR correction pass."""
    if int(pass_index) < 0:
        raise ValueError("PPO pass_index must be non-negative")
    if int(grpo_passes) <= 0:
        raise ValueError("grpo_passes must be positive")
    return resolve_eitr_mode(config) != "off" and int(pass_index) >= int(grpo_passes)


def eitr_loss_enabled_for_pass(
    config: Any,
    pass_index: int,
    *,
    grpo_passes: int = 1,
) -> bool:
    """Whether a PPO pass applies the fixed-lambda environment penalty."""
    return (
        eitr_probe_enabled_for_pass(config, pass_index, grpo_passes=grpo_passes)
        and resolve_eitr_mode(config) == "eitr"
    )


def validate_eitr_optimization_schedule(
    config: Any,
    ppo_epochs: int,
    correction_passes: int = 1,
) -> None:
    """Validate GRPO epochs followed by separate EITR-only corrections."""
    ppo_epochs = int(ppo_epochs)
    if ppo_epochs <= 0:
        raise ValueError("actor.ppo_epochs must be positive")
    mode = resolve_eitr_mode(config)
    if mode != "off" and int(correction_passes) <= 0:
        raise ValueError(
            f"EITR mode={mode} requires eitr.correction_passes >= 1"
        )
    if mode != "off" and int(correction_passes) != 1:
        raise ValueError(
            "V6 EITR requires exactly one full-batch correction pass"
        )


def build_eitr_correction_optimizer(
    parameters,
    config: Any,
    *,
    default_lr: float,
):
    """Build the stateless V6 correction optimizer without touching AdamW state."""
    if resolve_eitr_mode(config) != "eitr":
        return None
    optimizer_name = str(
        _config_value(config, "correction_optimizer", "sgd")
    ).strip().lower()
    if optimizer_name != "sgd":
        raise ValueError(
            "V6 EITR correction_optimizer must be 'sgd'"
        )
    configured_lr = _config_value(config, "correction_lr", None)
    correction_lr = float(default_lr if configured_lr is None else configured_lr)
    if correction_lr <= 0:
        raise ValueError("EITR correction_lr must be positive")
    return torch.optim.SGD(
        parameters,
        lr=correction_lr,
        momentum=0.0,
        weight_decay=0.0,
    )


def validate_eitr_config(
    config: Any,
    *,
    n_agent: int,
    max_queries_per_turn: int,
    rollout_n: int,
    rollout_response_length: Optional[int] = None,
    max_prompt_length: Optional[int] = None,
    rollout_max_model_len: Optional[int] = None,
    rollout_top_p: Optional[float] = None,
    rollout_top_k: Optional[int] = None,
) -> None:
    """Fail early when the Conditional EITR estimator assumptions do not hold."""
    mode = resolve_eitr_mode(config)
    probe_count = int(_config_value(config, "probe_count", 4))
    min_valid_probe_count = int(_config_value(config, "min_valid_probe_count", 2))
    probe_probability = float(_config_value(config, "probe_probability", 1.0))
    probe_oversample = int(_config_value(config, "probe_oversample", 0))
    probe_micro_batch_size = int(_config_value(config, "probe_micro_batch_size", 4))
    probe_logprob_micro_batch_size = int(
        _config_value(config, "probe_logprob_micro_batch_size", 4)
    )
    max_query_tokens = int(_config_value(config, "max_query_tokens", 512))
    max_action_tokens = int(_config_value(config, "max_action_tokens", 512))
    max_probe_prompt_tokens = int(_config_value(config, "max_probe_prompt_tokens", 4096))
    max_doc_support = int(_config_value(config, "max_doc_support", 32))
    score_temperature = float(_config_value(config, "retrieval_score_temperature", 0.1))
    min_state_coverage = float(_config_value(config, "min_state_coverage", 0.0))
    lambda_env = float(_config_value(config, "lambda_env", 0.1))
    correction_optimizer = str(
        _config_value(config, "correction_optimizer", "sgd")
    ).strip().lower()
    correction_lr = _config_value(config, "correction_lr", None)
    post_diagnostic_freq = int(_config_value(config, "post_diagnostic_freq", 0))
    log_ratio_clip = float(_config_value(config, "log_ratio_clip", 10.0))
    informative_js_threshold = float(_config_value(config, "informative_js_threshold", 0.01))
    min_informative_state_rate = float(
        _config_value(config, "min_informative_state_rate", 0.0)
    )

    if probe_count < 2:
        raise ValueError("EITR requires probe_count >= 2")
    if not 2 <= min_valid_probe_count <= probe_count:
        raise ValueError(
            "EITR min_valid_probe_count must be in [2, probe_count], got "
            f"{min_valid_probe_count} for probe_count={probe_count}"
        )
    if n_agent <= 0:
        raise ValueError("EITR requires rollout.n_agent > 0")
    if not 0.0 <= probe_probability <= 1.0:
        raise ValueError("EITR probe_probability must be in [0, 1]")
    if max_queries_per_turn != 1:
        raise ValueError(
            "The Gate C estimator requires retriever.max_queries_per_turn=1"
        )
    if rollout_n != 1:
        raise ValueError("The Gate C estimator currently requires rollout.n=1")
    if probe_oversample < 0:
        raise ValueError("EITR probe_oversample must be non-negative")
    if min(probe_micro_batch_size, probe_logprob_micro_batch_size) <= 0:
        raise ValueError("EITR probe micro-batch sizes must be positive")
    if min(
        max_query_tokens,
        max_action_tokens,
        max_probe_prompt_tokens,
        max_doc_support,
        probe_micro_batch_size,
    ) <= 0:
        raise ValueError("EITR token, support, and probe micro-batch limits must be positive")
    if rollout_response_length is not None and max_query_tokens != int(rollout_response_length):
        raise ValueError(
            "EITR max_query_tokens must equal data.max_response_length so real and "
            "counterfactual queries use the same turn budget; got "
            f"{max_query_tokens} != {int(rollout_response_length)}"
        )
    if max_action_tokens < max_query_tokens:
        raise ValueError(
            "EITR max_action_tokens must be at least max_query_tokens so tensor packing "
            "does not impose a smaller query support"
        )
    if max_prompt_length is not None and max_probe_prompt_tokens < int(max_prompt_length):
        raise ValueError(
            "EITR max_probe_prompt_tokens must be at least data.max_prompt_length so probe "
            "generation and probe log-prob computation use the exact same state; "
            f"got {max_probe_prompt_tokens} < {int(max_prompt_length)}"
        )
    if (
        rollout_max_model_len is not None
        and max_probe_prompt_tokens + max_query_tokens > int(rollout_max_model_len)
    ):
        raise ValueError(
            "EITR vLLM max_model_len must cover max_probe_prompt_tokens + "
            "max_query_tokens; got "
            f"{rollout_max_model_len} < {max_probe_prompt_tokens + max_query_tokens}"
        )
    if rollout_top_p is not None and abs(float(rollout_top_p) - 1.0) > 1e-8:
        raise ValueError(
            "EITR requires rollout.top_p=1.0 so sampled queries and recomputed "
            "full-softmax log probabilities describe the same policy"
        )
    if rollout_top_k is not None and int(rollout_top_k) != -1:
        raise ValueError(
            "EITR requires rollout.top_k=-1 so query sampling is not truncated"
        )
    if score_temperature <= 0:
        raise ValueError("EITR retrieval_score_temperature must be positive")
    if not 0.0 <= min_state_coverage <= 1.0:
        raise ValueError("EITR min_state_coverage must be in [0, 1]")
    if lambda_env < 0:
        raise ValueError("EITR lambda_env must be non-negative")
    if correction_optimizer != "sgd":
        raise ValueError("V6 EITR correction_optimizer must be 'sgd'")
    if correction_lr is not None and float(correction_lr) <= 0:
        raise ValueError("EITR correction_lr must be positive")
    if post_diagnostic_freq < 0:
        raise ValueError("EITR post_diagnostic_freq must be non-negative")
    if log_ratio_clip <= 0:
        raise ValueError("EITR log_ratio_clip must be positive")
    if informative_js_threshold < 0:
        raise ValueError("EITR informative_js_threshold must be non-negative")
    if not 0.0 <= min_informative_state_rate <= 1.0:
        raise ValueError("EITR min_informative_state_rate must be in [0, 1]")
    if mode == "off" and bool(_config_value(config, "enabled", False)):
        # An explicit mode is authoritative. This catches an otherwise very
        # easy-to-miss contradictory migration override.
        raise ValueError("EITR mode=off conflicts with legacy enabled=true")


def validate_sibling_group_layout(uids: Sequence[Any], *, n_agent: int, world_size: int) -> None:
    """Ensure each FSDP rank sees the same fixed sibling-group layout."""
    uid_strings = [str(uid) for uid in uids]
    if len(uid_strings) % world_size != 0:
        raise ValueError(f"EITR rollout batch {len(uid_strings)} is not divisible by world_size={world_size}")
    runs = []
    for uid in uid_strings:
        if not runs or runs[-1][0] != uid:
            runs.append([uid, 1])
        else:
            runs[-1][1] += 1
    bad_runs = [count for _, count in runs if count != n_agent]
    if bad_runs:
        raise ValueError(
            f"EITR expects contiguous uid groups of n_agent={n_agent}; bad group sizes={bad_runs[:8]}"
        )
    if len(runs) % world_size != 0:
        raise ValueError(
            f"EITR prompt groups {len(runs)} must be divisible by world_size={world_size}"
        )


def build_grpo_uids(data_sources: Sequence[Any], indices: Sequence[Any]) -> np.ndarray:
    """Build collision-free rollout group IDs for concatenated QA datasets."""
    if len(data_sources) != len(indices):
        raise ValueError(
            f"GRPO uid metadata length mismatch: sources={len(data_sources)}, indices={len(indices)}"
        )
    return np.asarray(
        [f"{str(source)}::{str(index)}" for source, index in zip(data_sources, indices)],
        dtype=object,
    )


def _softmax(values: Sequence[float], temperature: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    scaled = array / max(float(temperature), 1e-6)
    scaled -= np.max(scaled)
    weights = np.exp(scaled)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(array.shape, 1.0 / array.size, dtype=np.float64)
    return weights / total


def _effect_distribution(
    effect: Sequence[Mapping[str, Any]],
    support_lookup: Mapping[str, int],
    support_width: int,
    score_temperature: float,
) -> torch.Tensor:
    distribution = torch.zeros(support_width, dtype=torch.float32)
    if not effect:
        return distribution

    scores = []
    doc_ids = []
    for rank, item in enumerate(effect):
        doc_id = str(item.get("doc_id", "")).strip()
        if not doc_id or doc_id not in support_lookup:
            continue
        score = item.get("score")
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = -float(rank)
        if not np.isfinite(score):
            score = -float(rank)
        doc_ids.append(doc_id)
        scores.append(score)

    weights = _softmax(scores, score_temperature)
    for doc_id, weight in zip(doc_ids, weights):
        distribution[support_lookup[doc_id]] += float(weight)
    total = distribution.sum()
    if total > 0:
        distribution /= total
    return distribution


def probe_effect_diversity(
    doc_probs: torch.Tensor,
    probe_mask: torch.Tensor,
    *,
    informative_js_threshold: float = 0.01,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Measure whether same-state probes induce meaningfully different retrievals."""
    documents = doc_probs.float()
    mask = probe_mask.bool()
    if documents.ndim != 3 or mask.shape != documents.shape[:2]:
        raise ValueError("Expected doc_probs [states, probes, docs] and matching probe_mask")

    state_max_js = []
    state_mean_js = []
    state_top1_disagreement = []
    for state_documents, state_mask in zip(documents, mask):
        valid_documents = state_documents[state_mask]
        if valid_documents.size(0) < 2:
            state_max_js.append(documents.new_tensor(0.0))
            state_mean_js.append(documents.new_tensor(0.0))
            state_top1_disagreement.append(documents.new_tensor(0.0))
            continue

        valid_documents = valid_documents / valid_documents.sum(
            dim=-1, keepdim=True
        ).clamp_min(eps)
        pairwise_js = []
        for left_index in range(valid_documents.size(0)):
            for right_index in range(left_index + 1, valid_documents.size(0)):
                left = valid_documents[left_index]
                right = valid_documents[right_index]
                mixture = 0.5 * (left + right)
                left_kl = torch.sum(
                    torch.where(
                        left > 0,
                        left * (torch.log(left.clamp_min(eps)) - torch.log(mixture.clamp_min(eps))),
                        torch.zeros_like(left),
                    )
                )
                right_kl = torch.sum(
                    torch.where(
                        right > 0,
                        right * (torch.log(right.clamp_min(eps)) - torch.log(mixture.clamp_min(eps))),
                        torch.zeros_like(right),
                    )
                )
                pairwise_js.append(0.5 * (left_kl + right_kl))

        pairwise_js_tensor = torch.stack(pairwise_js)
        state_max_js.append(pairwise_js_tensor.max())
        state_mean_js.append(pairwise_js_tensor.mean())
        state_top1_disagreement.append(
            documents.new_tensor(
                float(torch.unique(valid_documents.argmax(dim=-1)).numel() > 1)
            )
        )

    if not state_max_js:
        empty = documents.new_zeros((0,))
        return {
            "state_max_js": empty,
            "state_mean_js": empty,
            "informative": empty.bool(),
            "top1_disagreement": empty,
        }

    state_max_js_tensor = torch.stack(state_max_js)
    return {
        "state_max_js": state_max_js_tensor,
        "state_mean_js": torch.stack(state_mean_js),
        "informative": state_max_js_tensor > float(informative_js_threshold),
        "top1_disagreement": torch.stack(state_top1_disagreement),
    }


def _probe_diversity_metrics(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: Any,
    total_state_count: int,
) -> Dict[str, float]:
    state_valid = tensors["eitr_state_valid"].bool()
    valid_state_count = int(state_valid.sum().item())
    informative_js_threshold = float(_config_value(config, "informative_js_threshold", 0.01))
    min_informative_state_rate = float(
        _config_value(config, "min_informative_state_rate", 0.0)
    )
    diversity = probe_effect_diversity(
        tensors["eitr_probe_doc_probs"][state_valid],
        tensors["eitr_probe_valid"][state_valid],
        informative_js_threshold=informative_js_threshold,
    )
    informative_count = int(diversity["informative"].sum().item())
    informative_rate = informative_count / max(valid_state_count, 1)
    informative_total_rate = informative_count / max(total_state_count, 1)
    pairwise_js_mean = (
        float(diversity["state_mean_js"].mean().item()) if valid_state_count else 0.0
    )
    pairwise_js_max = (
        float(diversity["state_max_js"].max().item()) if valid_state_count else 0.0
    )
    top1_disagreement_rate = (
        float(diversity["top1_disagreement"].mean().item()) if valid_state_count else 0.0
    )

    effective_probe_counts = tensors["eitr_probe_valid"][state_valid].sum(dim=-1).float()
    if effective_probe_counts.numel():
        effective_probe_count_mean = float(effective_probe_counts.mean().item())
        effective_probe_count_min = float(effective_probe_counts.min().item())
        effective_probe_count_max = float(effective_probe_counts.max().item())
    else:
        effective_probe_count_mean = 0.0
        effective_probe_count_min = 0.0
        effective_probe_count_max = 0.0

    return {
        "eitr/informative_probe_state_count": float(informative_count),
        "eitr/informative_probe_state_rate": float(informative_rate),
        "eitr/informative_probe_total_state_rate": float(informative_total_rate),
        "eitr/probe_effect_pairwise_js_mean": pairwise_js_mean,
        "eitr/probe_effect_pairwise_js_max": pairwise_js_max,
        "eitr/probe_effect_top1_disagreement_rate": top1_disagreement_rate,
        "eitr/informative_js_threshold": informative_js_threshold,
        "eitr/informative_probe_state_rate_below_threshold": float(
            min_informative_state_rate > 0
            and informative_rate < min_informative_state_rate
        ),
        "eitr/effective_probe_count_mean": effective_probe_count_mean,
        "eitr/effective_probe_count_min": effective_probe_count_min,
        "eitr/effective_probe_count_max": effective_probe_count_max,
    }


def _record_is_eligible(
    record: Any,
    response_ids: torch.Tensor,
    max_action_tokens: int,
    require_empty_prefix: bool,
) -> Tuple[bool, str]:
    if not isinstance(record, Mapping):
        return False, "missing_record"
    queries = record.get("queries") or []
    if len(queries) != 1:
        return False, "not_single_query"
    if require_empty_prefix and str(record.get("prefix_text", "")).strip():
        return False, "nonempty_prefix"
    action_ids = record.get("action_token_ids") or []
    if not action_ids:
        return False, "missing_action_tokens"
    if len(action_ids) > max_action_tokens:
        return False, "action_too_long"
    effect = record.get("retrieval_effect") or []
    if not effect:
        return False, "missing_retrieval_effect"
    expected = torch.as_tensor(action_ids, dtype=response_ids.dtype, device=response_ids.device)
    if response_ids.numel() < expected.numel() or not torch.equal(response_ids[: expected.numel()], expected):
        return False, "action_alignment_failed"
    return True, "ok"


def build_sibling_probe_tensors(
    *,
    prompts: torch.Tensor,
    attention_mask: torch.Tensor,
    responses: torch.Tensor,
    old_log_probs: torch.Tensor,
    uids: Sequence[Any],
    records: Sequence[Any],
    pad_token_id: int,
    config: Any,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Build one cached probe group on one representative row per prompt.

    Tensor dimension zero remains the ordinary rollout batch.  A representative
    row stores K compact prompt+query sequences, allowing arbitrary downstream
    data-parallel reordering without splitting a probe group.
    """
    batch_size, prompt_width = prompts.shape
    response_width = responses.shape[1]
    if len(uids) != batch_size or len(records) != batch_size:
        raise ValueError(
            f"EITR metadata length mismatch: batch={batch_size}, uids={len(uids)}, records={len(records)}"
        )

    probe_count = int(_config_value(config, "probe_count", 4))
    min_valid_probe_count = int(_config_value(config, "min_valid_probe_count", 2))
    max_action_tokens = int(_config_value(config, "max_action_tokens", 128))
    max_doc_support = int(_config_value(config, "max_doc_support", 32))
    score_temperature = float(_config_value(config, "retrieval_score_temperature", 0.1))
    require_empty_prefix = bool(_config_value(config, "require_empty_prefix", True))
    min_state_coverage = float(_config_value(config, "min_state_coverage", 0.0))

    if max_action_tokens <= 0 or max_doc_support <= 0:
        raise ValueError("EITR max_action_tokens and max_doc_support must be positive")

    sequence_width = prompt_width + max_action_tokens
    tensors = {
        "eitr_probe_input_ids": torch.full(
            (batch_size, probe_count, sequence_width), pad_token_id, dtype=torch.long
        ),
        "eitr_probe_attention_mask": torch.zeros(
            (batch_size, probe_count, sequence_width), dtype=torch.long
        ),
        "eitr_probe_position_ids": torch.zeros(
            (batch_size, probe_count, sequence_width), dtype=torch.long
        ),
        "eitr_probe_responses": torch.full(
            (batch_size, probe_count, max_action_tokens), pad_token_id, dtype=torch.long
        ),
        "eitr_probe_response_mask": torch.zeros(
            (batch_size, probe_count, max_action_tokens), dtype=torch.long
        ),
        "eitr_probe_old_seq_logp": torch.zeros((batch_size, probe_count), dtype=torch.float32),
        "eitr_probe_doc_probs": torch.zeros(
            (batch_size, probe_count, max_doc_support), dtype=torch.float32
        ),
        "eitr_probe_valid": torch.zeros((batch_size, probe_count), dtype=torch.long),
        "eitr_state_slot": torch.zeros(batch_size, dtype=torch.long),
        "eitr_state_valid": torch.zeros(batch_size, dtype=torch.long),
    }

    grouped_indices: Dict[str, list[int]] = defaultdict(list)
    for index, uid in enumerate(uids):
        grouped_indices[str(uid)].append(index)

    # Keep one fixed compute slot per uid group.  Even an ineligible group gets
    # a zero-loss dummy probe, so every FSDP rank executes identical forwards.
    for group_indices in grouped_indices.values():
        representative = group_indices[0]
        tensors["eitr_state_slot"][representative] = 1
        source_index = group_indices[0]
        record = records[source_index] if isinstance(records[source_index], Mapping) else {}
        dummy_action = list(record.get("action_token_ids") or [])[:max_action_tokens]
        if not dummy_action:
            dummy_action = [int(responses[source_index, 0].item())]
        dummy_action_ids = torch.as_tensor(dummy_action, dtype=torch.long)
        dummy_length = int(dummy_action_ids.numel())
        for probe_offset in range(probe_count):
            tensors["eitr_probe_input_ids"][representative, probe_offset, :prompt_width] = prompts[
                source_index
            ].long()
            tensors["eitr_probe_input_ids"][
                representative, probe_offset, prompt_width : prompt_width + dummy_length
            ] = dummy_action_ids
            tensors["eitr_probe_attention_mask"][representative, probe_offset, :prompt_width] = (
                attention_mask[source_index, :prompt_width].long()
            )
            tensors["eitr_probe_attention_mask"][
                representative, probe_offset, prompt_width : prompt_width + dummy_length
            ] = 1
            dummy_attention = tensors["eitr_probe_attention_mask"][representative, probe_offset]
            tensors["eitr_probe_position_ids"][representative, probe_offset] = (
                torch.cumsum(dummy_attention, dim=0) - 1
            ).clamp_min(0)
            tensors["eitr_probe_responses"][representative, probe_offset, :dummy_length] = (
                dummy_action_ids
            )

    eligible_by_group: Dict[str, list[int]] = defaultdict(list)
    rejection_counts: Counter[str] = Counter()
    for uid, indices in grouped_indices.items():
        for index in indices:
            eligible, reason = _record_is_eligible(
                records[index], responses[index], max_action_tokens, require_empty_prefix
            )
            if eligible:
                eligible_by_group[uid].append(index)
            else:
                rejection_counts[reason] += 1

    valid_state_count = 0
    partial_state_count = 0
    support_truncation_count = 0
    for uid, group_indices in grouped_indices.items():
        selected = eligible_by_group.get(uid, [])[:probe_count]
        if len(selected) < min_valid_probe_count:
            rejection_counts["insufficient_sibling_probes"] += 1
            continue
        if len(selected) < probe_count:
            partial_state_count += 1

        doc_ids = []
        seen_doc_ids = set()
        for index in selected:
            for item in records[index]["retrieval_effect"]:
                doc_id = str(item.get("doc_id", "")).strip()
                if doc_id and doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    doc_ids.append(doc_id)
        if len(doc_ids) > max_doc_support:
            support_truncation_count += 1
            doc_ids = doc_ids[:max_doc_support]
        if not doc_ids:
            rejection_counts["empty_doc_union"] += 1
            continue

        representative = group_indices[0]
        support_lookup = {doc_id: offset for offset, doc_id in enumerate(doc_ids)}
        for probe_offset, source_index in enumerate(selected):
            action_ids = torch.as_tensor(records[source_index]["action_token_ids"], dtype=torch.long)
            action_length = int(action_ids.numel())

            tensors["eitr_probe_input_ids"][representative, probe_offset].fill_(pad_token_id)
            tensors["eitr_probe_attention_mask"][representative, probe_offset].zero_()
            tensors["eitr_probe_responses"][representative, probe_offset].fill_(pad_token_id)
            tensors["eitr_probe_response_mask"][representative, probe_offset].zero_()

            tensors["eitr_probe_input_ids"][representative, probe_offset, :prompt_width] = prompts[
                source_index
            ].long()
            tensors["eitr_probe_input_ids"][
                representative, probe_offset, prompt_width : prompt_width + action_length
            ] = action_ids
            prompt_attention = attention_mask[source_index, :prompt_width].long()
            tensors["eitr_probe_attention_mask"][representative, probe_offset, :prompt_width] = (
                prompt_attention
            )
            tensors["eitr_probe_attention_mask"][
                representative, probe_offset, prompt_width : prompt_width + action_length
            ] = 1
            probe_attention = tensors["eitr_probe_attention_mask"][representative, probe_offset]
            tensors["eitr_probe_position_ids"][representative, probe_offset] = (
                torch.cumsum(probe_attention, dim=0) - 1
            ).clamp_min(0)
            tensors["eitr_probe_responses"][representative, probe_offset, :action_length] = action_ids
            tensors["eitr_probe_response_mask"][representative, probe_offset, :action_length] = 1
            tensors["eitr_probe_old_seq_logp"][representative, probe_offset] = old_log_probs[
                source_index, :action_length
            ].float().sum()
            effect_distribution = _effect_distribution(
                records[source_index]["retrieval_effect"],
                support_lookup,
                max_doc_support,
                score_temperature,
            )
            tensors["eitr_probe_doc_probs"][representative, probe_offset] = effect_distribution
            if effect_distribution.sum() > 0:
                tensors["eitr_probe_valid"][representative, probe_offset] = 1
            else:
                tensors["eitr_probe_response_mask"][representative, probe_offset].zero_()
                rejection_counts["empty_probe_distribution"] += 1

        effective_probe_count = int(tensors["eitr_probe_valid"][representative].sum().item())
        if effective_probe_count >= min_valid_probe_count:
            tensors["eitr_state_valid"][representative] = 1
            valid_state_count += 1
        else:
            tensors["eitr_probe_valid"][representative].zero_()
            tensors["eitr_probe_response_mask"][representative].zero_()
            rejection_counts["insufficient_effective_sibling_probes"] += 1

    if valid_state_count == 0:
        tensors["eitr_state_slot"].zero_()

    total_state_count = len(grouped_indices)
    coverage = valid_state_count / total_state_count if total_state_count else 0.0
    metrics = {
        "eitr/probe_state_count": float(valid_state_count),
        "eitr/probe_state_coverage": float(coverage),
        "eitr/probe_rollout_coverage": float(
            tensors["eitr_probe_valid"].sum().item() / max(batch_size, 1)
        ),
        "eitr/support_truncation_count": float(support_truncation_count),
        "eitr/rejected_rollout_count": float(sum(rejection_counts.values())),
        "eitr/partial_probe_state_count": float(partial_state_count),
        "eitr/probe_state_coverage_below_threshold": float(
            min_state_coverage > 0 and coverage < min_state_coverage
        ),
    }
    metrics.update(
        _probe_diversity_metrics(
            tensors,
            config=config,
            total_state_count=total_state_count,
        )
    )
    return tensors, metrics


def build_online_probe_tensors(
    *,
    prompts: torch.Tensor,
    attention_mask: torch.Tensor,
    responses: torch.Tensor,
    uids: Sequence[Any],
    probe_groups: Sequence[Any],
    pad_token_id: int,
    config: Any,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Build exact same-prefix online probes, one optional state per rollout row.

    Every rollout row owns a fixed tensor slot, including rows without a valid
    search.  Besides making Conditional EITR naturally mask invalid rows, this
    keeps the number of probe forwards identical across FSDP ranks.
    """
    batch_size, original_prompt_width = prompts.shape
    if len(uids) != batch_size or len(probe_groups) != batch_size:
        raise ValueError(
            f"EITR online metadata mismatch: batch={batch_size}, uids={len(uids)}, "
            f"probe_groups={len(probe_groups)}"
        )
    probe_count = int(_config_value(config, "probe_count", 4))
    min_valid_probe_count = int(_config_value(config, "min_valid_probe_count", 2))
    max_action_tokens = int(_config_value(config, "max_action_tokens", 128))
    max_prompt_tokens = int(_config_value(config, "max_probe_prompt_tokens", 4096))
    max_doc_support = int(_config_value(config, "max_doc_support", 32))
    score_temperature = float(_config_value(config, "retrieval_score_temperature", 0.1))
    min_state_coverage = float(_config_value(config, "min_state_coverage", 0.0))

    available_prompt_lengths = []
    for group in probe_groups:
        if isinstance(group, Mapping) and group.get("state_prompt_token_ids"):
            available_prompt_lengths.append(len(group["state_prompt_token_ids"]))
    prompt_width = min(max(available_prompt_lengths or [original_prompt_width]), max_prompt_tokens)
    sequence_width = prompt_width + max_action_tokens
    tensors = {
        "eitr_probe_input_ids": torch.full(
            (batch_size, probe_count, sequence_width), pad_token_id, dtype=torch.long
        ),
        "eitr_probe_attention_mask": torch.zeros(
            (batch_size, probe_count, sequence_width), dtype=torch.long
        ),
        "eitr_probe_position_ids": torch.zeros(
            (batch_size, probe_count, sequence_width), dtype=torch.long
        ),
        "eitr_probe_responses": torch.full(
            (batch_size, probe_count, max_action_tokens), pad_token_id, dtype=torch.long
        ),
        "eitr_probe_response_mask": torch.zeros(
            (batch_size, probe_count, max_action_tokens), dtype=torch.long
        ),
        "eitr_probe_old_seq_logp": torch.zeros((batch_size, probe_count), dtype=torch.float32),
        "eitr_probe_doc_probs": torch.zeros(
            (batch_size, probe_count, max_doc_support), dtype=torch.float32
        ),
        "eitr_probe_valid": torch.zeros((batch_size, probe_count), dtype=torch.long),
        "eitr_state_slot": torch.zeros(batch_size, dtype=torch.long),
        "eitr_state_valid": torch.zeros(batch_size, dtype=torch.long),
    }

    def fill_probe(representative, probe_offset, state_ids, action_ids, response_mask_value):
        state_ids = list(state_ids)[-prompt_width:]
        action_ids = list(action_ids)[:max_action_tokens]
        if not action_ids:
            action_ids = [pad_token_id]
        prompt_start = prompt_width - len(state_ids)
        action_length = len(action_ids)
        tensors["eitr_probe_input_ids"][representative, probe_offset, prompt_start:prompt_width] = (
            torch.as_tensor(state_ids, dtype=torch.long)
        )
        tensors["eitr_probe_input_ids"][
            representative, probe_offset, prompt_width : prompt_width + action_length
        ] = torch.as_tensor(action_ids, dtype=torch.long)
        tensors["eitr_probe_attention_mask"][representative, probe_offset, prompt_start:prompt_width] = 1
        tensors["eitr_probe_attention_mask"][
            representative, probe_offset, prompt_width : prompt_width + action_length
        ] = 1
        probe_attention = tensors["eitr_probe_attention_mask"][representative, probe_offset]
        tensors["eitr_probe_position_ids"][representative, probe_offset] = (
            torch.cumsum(probe_attention, dim=0) - 1
        ).clamp_min(0)
        tensors["eitr_probe_responses"][representative, probe_offset, :action_length] = torch.as_tensor(
            action_ids, dtype=torch.long
        )
        if response_mask_value:
            tensors["eitr_probe_response_mask"][representative, probe_offset, :action_length] = 1

    rejection_counts: Counter[str] = Counter()
    valid_state_count = 0
    partial_state_count = 0
    support_truncation_count = 0
    for representative, candidate_group in enumerate(probe_groups):
        # A physical slot exists for every row. Invalid rows retain the dummy
        # forward below and contribute exactly zero to the loss.
        tensors["eitr_state_slot"][representative] = 1
        group = candidate_group if isinstance(candidate_group, Mapping) else None

        prompt_mask = attention_mask[representative, :original_prompt_width].bool()
        dummy_state_ids = prompts[representative][prompt_mask].tolist()
        dummy_action_ids = [int(responses[representative, 0].item())]
        for probe_offset in range(probe_count):
            fill_probe(representative, probe_offset, dummy_state_ids, dummy_action_ids, False)

        if not group:
            rejection_counts["missing_online_probe_group"] += 1
            continue
        state_ids = list(group.get("state_prompt_token_ids") or [])
        if len(state_ids) > max_prompt_tokens:
            # Never silently shorten a probe state. The cached actions and
            # retrieval effects were sampled under the full prefix, so scoring
            # them under a shorter prefix would invalidate the importance ratio.
            rejection_counts["state_prompt_too_long"] += 1
            continue
        probes = list(group.get("probes") or [])
        eligible_probes = [
            probe
            for probe in probes
            if isinstance(probe, Mapping)
            and probe.get("action_token_ids")
            and len(probe["action_token_ids"]) <= max_action_tokens
            and probe.get("retrieval_effect")
        ][:probe_count]
        if not state_ids or len(eligible_probes) < min_valid_probe_count:
            rejection_counts["insufficient_online_probes"] += 1
            continue
        if len(eligible_probes) < probe_count:
            partial_state_count += 1

        doc_ids = []
        seen_doc_ids = set()
        for probe in eligible_probes:
            for item in probe["retrieval_effect"]:
                doc_id = str(item.get("doc_id", "")).strip()
                if doc_id and doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    doc_ids.append(doc_id)
        if len(doc_ids) > max_doc_support:
            support_truncation_count += 1
            doc_ids = doc_ids[:max_doc_support]
        if not doc_ids:
            rejection_counts["empty_doc_union"] += 1
            continue

        support_lookup = {doc_id: offset for offset, doc_id in enumerate(doc_ids)}
        for probe_offset, probe in enumerate(eligible_probes):
            tensors["eitr_probe_input_ids"][representative, probe_offset].fill_(pad_token_id)
            tensors["eitr_probe_attention_mask"][representative, probe_offset].zero_()
            tensors["eitr_probe_position_ids"][representative, probe_offset].zero_()
            tensors["eitr_probe_responses"][representative, probe_offset].fill_(pad_token_id)
            tensors["eitr_probe_response_mask"][representative, probe_offset].zero_()
            fill_probe(
                representative,
                probe_offset,
                state_ids,
                probe["action_token_ids"],
                True,
            )
            effect_distribution = _effect_distribution(
                probe["retrieval_effect"],
                support_lookup,
                max_doc_support,
                score_temperature,
            )
            tensors["eitr_probe_doc_probs"][representative, probe_offset] = effect_distribution
            if effect_distribution.sum() > 0:
                tensors["eitr_probe_valid"][representative, probe_offset] = 1
            else:
                tensors["eitr_probe_response_mask"][representative, probe_offset].zero_()
                rejection_counts["empty_probe_distribution"] += 1

        effective_probe_count = int(tensors["eitr_probe_valid"][representative].sum().item())
        if effective_probe_count >= min_valid_probe_count:
            tensors["eitr_state_valid"][representative] = 1
            valid_state_count += 1
        else:
            tensors["eitr_probe_valid"][representative].zero_()
            tensors["eitr_probe_response_mask"][representative].zero_()
            rejection_counts["insufficient_effective_online_probes"] += 1

    # With no usable state every rank can skip the probe forward entirely. If
    # at least one state is usable, all rows remain physical slots so later DP
    # partitioning cannot create mismatched FSDP forward counts.
    if valid_state_count == 0:
        tensors["eitr_state_slot"].zero_()

    total_state_count = batch_size
    coverage = valid_state_count / total_state_count if total_state_count else 0.0
    metrics = {
        "eitr/probe_state_count": float(valid_state_count),
        "eitr/probe_state_coverage": float(coverage),
        "eitr/probe_rollout_coverage": float(
            tensors["eitr_probe_valid"].sum().item() / max(batch_size, 1)
        ),
        "eitr/support_truncation_count": float(support_truncation_count),
        "eitr/rejected_state_count": float(sum(rejection_counts.values())),
        "eitr/partial_probe_state_count": float(partial_state_count),
        "eitr/probe_state_coverage_below_threshold": float(
            min_state_coverage > 0 and coverage < min_state_coverage
        ),
        "eitr/probe_retrieval_call_count": float(
            sum(
                int(group.get("extra_retrieval_calls", 0))
                for group in probe_groups
                if isinstance(group, Mapping)
            )
        ),
    }
    metrics.update(
        _probe_diversity_metrics(
            tensors,
            config=config,
            total_state_count=total_state_count,
        )
    )
    return tensors, metrics


def attach_eitr_probe_tensors(
    batch: Any,
    config: Any,
    pad_token_id: Optional[int] = None,
) -> Tuple[Any, Dict[str, float]]:
    uids = batch.non_tensor_batch.get("uid")
    if uids is None:
        raise RuntimeError("EITR requires GRPO uid groups")
    if pad_token_id is None:
        pad_token_id = int(batch.meta_info["pad_token_id"])
    probe_source = str(_config_value(config, "probe_source", "online_same_state"))
    collector_stats = batch.meta_info.get("eitr_probe_collection_stats") or {}
    if probe_source == "online_same_state":
        probe_groups = batch.meta_info.get("eitr_probe_groups")
        if probe_groups is None:
            raise RuntimeError("EITR rollout did not provide online same-state probe groups")
        try:
            tensors, metrics = build_online_probe_tensors(
                prompts=batch.batch["prompts"],
                attention_mask=batch.batch["attention_mask"],
                responses=batch.batch["responses"],
                uids=uids,
                probe_groups=probe_groups,
                pad_token_id=pad_token_id,
                config=config,
            )
        except RuntimeError as error:
            if collector_stats:
                raise RuntimeError(f"{error}; collector={dict(collector_stats)}") from error
            raise
    elif probe_source == "sibling_rollouts":
        records = batch.meta_info.get("eitr_first_search_records")
        if records is None:
            raise RuntimeError("EITR rollout did not provide first-search sibling records")
        tensors, metrics = build_sibling_probe_tensors(
            prompts=batch.batch["prompts"],
            attention_mask=batch.batch["attention_mask"],
            responses=batch.batch["responses"],
            old_log_probs=batch.batch["old_log_probs"],
            uids=uids,
            records=records,
            pad_token_id=pad_token_id,
            config=config,
        )
    else:
        raise ValueError(f"Unsupported EITR probe_source={probe_source}")
    for key, value in tensors.items():
        batch.batch[key] = value
    for key, value in collector_stats.items():
        metrics[f"eitr/collector_{key}"] = float(value)
    real_valid = float(collector_stats.get("real_search_valid", 0.0))
    selected = float(collector_stats.get("probe_state_selected", 0.0))
    generated = float(collector_stats.get("probe_candidate_generated", 0.0))
    accepted = float(collector_stats.get("probe_query_accepted", 0.0))
    effective = float(collector_stats.get("probe_state_effective", 0.0))
    rollout_count = float(len(uids))
    metrics.update({
        "eitr/real_valid_search_count": real_valid,
        # This action count may exceed one because a trajectory can search on
        # multiple turns.  The cross-mode trajectory-level rate is computed
        # from the environment's valid_search_stats in ray_trainer instead.
        "eitr/real_valid_search_per_rollout": real_valid / max(rollout_count, 1.0),
        "eitr/probe_selected_state_count": selected,
        "eitr/probe_candidate_valid_rate": accepted / max(generated, 1.0),
        "eitr/active_state_rate_given_selected": effective / max(selected, 1.0),
        "eitr/deferred_additional_search_state_count": float(
            collector_stats.get("additional_state_deferred", 0.0)
        ),
    })
    batch.meta_info.pop("eitr_probe_groups", None)
    batch.meta_info.pop("eitr_probe_state_groups", None)
    batch.meta_info.pop("eitr_first_search_records", None)
    batch.meta_info.pop("eitr_probe_collection_stats", None)
    return batch, metrics


def flatten_probe_logprob_inputs(batch: Any) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    state_slot = batch.batch["eitr_state_slot"].bool()
    tensors = {
        "input_ids": batch.batch["eitr_probe_input_ids"][state_slot].flatten(0, 1),
        "attention_mask": batch.batch["eitr_probe_attention_mask"][state_slot].flatten(0, 1),
        "position_ids": batch.batch["eitr_probe_position_ids"][state_slot].flatten(0, 1),
        "responses": batch.batch["eitr_probe_responses"][state_slot].flatten(0, 1),
    }
    response_mask = batch.batch["eitr_probe_response_mask"][state_slot].flatten(0, 1).float()
    return tensors, response_mask


def assign_probe_old_log_probs(
    batch: Any,
    token_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> Any:
    state_slot = batch.batch["eitr_state_slot"].bool()
    state_count = int(state_slot.sum().item())
    probe_count = batch.batch["eitr_probe_valid"].size(1)
    expected = state_count * probe_count
    if token_log_probs.size(0) != expected:
        raise ValueError(f"Expected {expected} probe log-prob rows, got {token_log_probs.size(0)}")
    sequence_log_probs = (token_log_probs.float() * response_mask.float()).sum(dim=-1)
    batch.batch["eitr_probe_old_seq_logp"][state_slot] = sequence_log_probs.view(
        state_count, probe_count
    )
    return batch


# Backward-compatible name for early Gate C prototypes.
attach_sibling_probe_tensors = attach_eitr_probe_tensors


def induced_js_from_cached_effects(
    *,
    current_seq_logp: torch.Tensor,
    old_seq_logp: torch.Tensor,
    doc_probs: torch.Tensor,
    probe_mask: torch.Tensor,
    log_ratio_clip: float = 10.0,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Estimate policy-induced retrieval JS using SNIS probe weights."""
    current = current_seq_logp.float()
    old = old_seq_logp.float()
    documents = doc_probs.float()
    mask = probe_mask.bool()
    if current.ndim != 2 or documents.ndim != 3:
        raise ValueError("Expected current/old [states, probes] and doc_probs [states, probes, docs]")
    if current.shape != old.shape or current.shape != mask.shape or current.shape != documents.shape[:2]:
        raise ValueError("EITR probe tensor shapes do not align")
    if torch.any(mask.sum(dim=-1) < 2):
        raise ValueError("Each EITR state needs at least two valid probes")

    raw_log_ratio = current - old
    clipped_log_ratio = raw_log_ratio.clamp(-float(log_ratio_clip), float(log_ratio_clip))
    masked_log_ratio = clipped_log_ratio.masked_fill(~mask, torch.finfo(torch.float32).min)
    current_weights = torch.softmax(masked_log_ratio, dim=-1)
    old_weights = mask.float() / mask.float().sum(dim=-1, keepdim=True)

    old_distribution = torch.einsum("sk,sku->su", old_weights, documents)
    current_distribution = torch.einsum("sk,sku->su", current_weights, documents)
    old_distribution = old_distribution / old_distribution.sum(dim=-1, keepdim=True).clamp_min(eps)
    current_distribution = current_distribution / current_distribution.sum(dim=-1, keepdim=True).clamp_min(eps)
    mixture = 0.5 * (old_distribution + current_distribution)

    old_kl = torch.sum(
        torch.where(
            old_distribution > 0,
            old_distribution * (torch.log(old_distribution.clamp_min(eps)) - torch.log(mixture.clamp_min(eps))),
            torch.zeros_like(old_distribution),
        ),
        dim=-1,
    )
    current_kl = torch.sum(
        torch.where(
            current_distribution > 0,
            current_distribution
            * (torch.log(current_distribution.clamp_min(eps)) - torch.log(mixture.clamp_min(eps))),
            torch.zeros_like(current_distribution),
        ),
        dim=-1,
    )
    js = 0.5 * (old_kl + current_kl)
    ess = 1.0 / current_weights.square().sum(dim=-1).clamp_min(eps)
    raw_log_ratio_abs = raw_log_ratio.abs()
    valid_probe_count = mask.float().sum(dim=-1).clamp_min(1.0)
    return {
        "js": js,
        "ess": ess,
        "current_weights": current_weights,
        "old_distribution": old_distribution,
        "current_distribution": current_distribution,
        "log_ratio_abs_max": raw_log_ratio_abs.masked_fill(~mask, 0.0).amax(dim=-1),
        "log_ratio_clipfrac": (
            ((raw_log_ratio_abs > float(log_ratio_clip)) & mask).float().sum(dim=-1)
            / valid_probe_count
        ),
    }
