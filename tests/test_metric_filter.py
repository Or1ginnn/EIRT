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
            "val/test_score/nq": 0.2,
            "reward/zero_rate": 0.3,
            "collector/internal_counter": 7.0,
            "actor/eitr_pass_1_log_ratio_abs_max": 0.4,
        }
        self.assertEqual(
            METRIC_FILTER.filter_metrics_for_logging(metrics, "core"),
            {
                "actor/pg_loss": 1.0,
                "actor/eitr_coverage": 0.5,
                "val/test_score/nq": 0.2,
                "reward/zero_rate": 0.3,
            },
        )

    def test_unknown_level_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "metrics_level"):
            METRIC_FILTER.filter_metrics_for_logging({}, "brief")


if __name__ == "__main__":
    unittest.main()
