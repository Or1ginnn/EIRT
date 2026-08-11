import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "verl"
    / "trainer"
    / "ppo"
    / "metric_filter.py"
)
SPEC = importlib.util.spec_from_file_location("metric_filter_module", MODULE_PATH)
METRIC_FILTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRIC_FILTER)


class MetricFilterTest(unittest.TestCase):
    def test_debug_preserves_every_metric(self):
        metrics = {"actor/pg_loss": 1.0, "collector/internal_counter": 7.0}
        self.assertEqual(
            METRIC_FILTER.filter_metrics_for_logging(metrics, "debug"),
            metrics,
        )

    def test_core_keeps_formal_metrics_and_dynamic_validation(self):
        metrics = {
            "actor/pg_loss": 1.0,
            "actor/eitr_coverage": 0.5,
            "actor/eitr_env_drift_relative_reduction": 0.02,
            "actor/eitr_probe_gradient_checkpointing_active": 1.0,
            "env/trajectory_valid_search_rate": 0.4,
            "val/test_score/nq": 0.2,
            "reward/zero_rate": 0.3,
            "collector/internal_counter": 7.0,
            "actor/eitr_pass_1_log_ratio_abs_max": 0.4,
            "actor/optimizer_state_offloaded_for_eitr": 1.0,
            "actor/batch_cpu_streaming": 1.0,
            "timing_s/actor_optimizer_offload_after_grpo": 3.0,
            "timing_s/actor_optimizer_load_before_grpo": 2.0,
            "timing_s/eitr_probe_generation": 30.0,
            "timing_s/eitr_probe_retrieval": 4.0,
            "timing_s/eitr_probe_old_logprob": 12.0,
            "timing_s/grpo_update": 80.0,
            "timing_s/eitr_correction": 70.0,
        }
        self.assertEqual(
            METRIC_FILTER.filter_metrics_for_logging(metrics, "core"),
            {
                "actor/pg_loss": 1.0,
                "actor/eitr_coverage": 0.5,
                "actor/eitr_env_drift_relative_reduction": 0.02,
                "actor/eitr_probe_gradient_checkpointing_active": 1.0,
                "env/trajectory_valid_search_rate": 0.4,
                "val/test_score/nq": 0.2,
                "reward/zero_rate": 0.3,
                "actor/optimizer_state_offloaded_for_eitr": 1.0,
                "actor/batch_cpu_streaming": 1.0,
                "timing_s/actor_optimizer_offload_after_grpo": 3.0,
                "timing_s/actor_optimizer_load_before_grpo": 2.0,
                "timing_s/eitr_probe_generation": 30.0,
                "timing_s/eitr_probe_retrieval": 4.0,
                "timing_s/eitr_probe_old_logprob": 12.0,
                "timing_s/grpo_update": 80.0,
                "timing_s/eitr_correction": 70.0,
            },
        )

    def test_unknown_level_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "metrics_level"):
            METRIC_FILTER.filter_metrics_for_logging({}, "brief")


if __name__ == "__main__":
    unittest.main()
