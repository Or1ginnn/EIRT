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
    "eitr/real_valid_search_count",
    "eitr/probe_candidate_valid_rate",
    "eitr/effective_probe_count_mean",
    # EITR mechanism. Pre/post names are reserved for the V6 diagnostic pass.
    "actor/eitr_coverage",
    "actor/eitr_effective_lambda",
    "actor/eitr_global_induced_js",
    "actor/eitr_global_probe_ess",
    "actor/eitr_correction_grad_norm",
    "actor/eitr_loss_applied",
    "actor/eitr_correction_optimizer_step_count",
    "actor/eitr_env_drift_pre",
    "actor/eitr_env_drift_post",
    "actor/eitr_env_drift_delta",
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
