"""Small logging views for formal runs without discarding debug metrics."""

from typing import Any, Dict, Mapping


CORE_METRIC_NAMES = {
    # Progress and task quality.
    "trainer/outer_update_step",
    "critic/score/mean",
    "critic/score/min",
    "critic/score/max",
    # Agent behavior.
    "env/finish_ratio",
    "env/number_of_executed_search",
    "env/trajectory_valid_search_rate",
    "eitr/real_valid_search_count",
    "eitr/real_valid_search_per_rollout",
    "eitr/probe_candidate_valid_rate",
    "eitr/effective_probe_count_mean",
    # EITR mechanism. Pre/post names are reserved for the V6 diagnostic pass.
    "actor/eitr_coverage",
    "actor/eitr_effective_lambda",
    "actor/eitr_global_induced_js",
    "actor/eitr_global_probe_ess",
    "actor/eitr_correction_grad_norm",
    "actor/eitr_correction_lr",
    "actor/eitr_correction_normalization_scale",
    "actor/eitr_correction_effective_lr",
    "actor/eitr_correction_unnormalized_update_norm",
    "actor/eitr_correction_predicted_update_norm",
    "actor/eitr_correction_max_update_norm",
    "actor/eitr_correction_min_update_norm",
    "actor/eitr_correction_skipped_weak_update",
    "actor/eitr_effective_step_scale",
    "actor/eitr_loss_applied",
    "actor/eitr_correction_optimizer_step_count",
    "actor/eitr_state_compaction_enabled",
    "actor/eitr_physical_state_count_before_compaction",
    "actor/eitr_compacted_state_forward_count",
    "actor/eitr_dummy_state_forward_count",
    "actor/eitr_state_forward_reduction_rate",
    "actor/eitr_probe_gradient_checkpointing_active",
    "actor/eitr_env_drift_pre",
    "actor/eitr_env_drift_post",
    "actor/eitr_env_drift_delta",
    "actor/eitr_env_drift_relative_reduction",
    "actor/eitr_post_diagnostic_ran",
    "actor/eitr_post_diagnostic_noop",
    "actor/optimizer_state_offloaded_for_eitr",
    "actor/batch_cpu_streaming",
    # Optimization stability.
    "actor/kl_loss",
    "actor/pg_loss",
    "actor/grad_norm",
    "actor/lr",
    "response_length/mean",
    "response_length/clip_ratio",
    # Cost.
    "timing_s/step",
    "timing_s/gen",
    "timing_s/update_actor",
    "timing_s/real_retrieval",
    "timing_s/eitr_probe_generation",
    "timing_s/eitr_probe_retrieval",
    "timing_s/eitr_probe_prepare",
    "timing_s/eitr_probe_old_logprob",
    "timing_s/ref",
    "timing_s/adv",
    "timing_s/grpo_update",
    "timing_s/eitr_correction",
    "timing_s/actor_optimizer_offload_after_grpo",
    "timing_s/actor_optimizer_load_before_grpo",
}

CORE_METRIC_PREFIXES = (
    "val/test_score/",
    "reward/",
)


def normalize_metrics_level(level: Any) -> str:
    normalized = str(level or "debug").strip().lower()
    if normalized not in {"core", "debug"}:
        raise ValueError(
            f"Unsupported trainer.metrics_level={level!r}; expected 'core' or 'debug'"
        )
    return normalized


def filter_metrics_for_logging(
    metrics: Mapping[str, Any],
    level: Any,
) -> Dict[str, Any]:
    """Return a compact formal view or the complete smoke/debug view."""
    normalized = normalize_metrics_level(level)
    if normalized == "debug":
        return dict(metrics)
    return {
        name: value
        for name, value in metrics.items()
        if name in CORE_METRIC_NAMES
        or any(name.startswith(prefix) for prefix in CORE_METRIC_PREFIXES)
    }
