import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/eval/summarize_v7_geometry_audit.py"
)
SPEC = importlib.util.spec_from_file_location("v7_geometry_summary", MODULE_PATH)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class V7GeometrySummaryTest(unittest.TestCase):
    def record(self, step, *, reverse_k4=False):
        reference = [0.0002, 0.0006, 0.0014, 0.0020]
        k4 = list(reversed(reference)) if reverse_k4 else [0.0003, 0.0005, 0.0012, 0.0022]
        return {
            "outer_update_step": step,
            "accept_radius": 0.001,
            "statistics": {
                "final_restore_max_abs": 0.0,
                "optimizer_state_restored_empty": 1.0,
                "no_update_committed": 1.0,
                "proposal_update_norm": 1e-5,
            },
            "per_state": {
                "token": [0.8, 0.1, 0.6, 0.2],
                "k4": k4,
                "reference": [value + step * 1e-5 for value in reference],
                "repeat": [value + step * 1e-5 + 1e-9 for value in reference],
                "ess_k4": [3.8] * 4,
                "ess_reference": [14.0] * 4,
                "valid_probe_count": [16.0] * 4,
            },
        }

    def test_twenty_clean_batches_go_to_phase3(self):
        result = SUMMARY.summarize(
            [self.record(step) for step in range(1, 21)],
            min_batches=20,
            frozen_radius=0.001,
        )
        self.assertEqual(result["verdict"], "GO_TO_V7_PHASE_3_CALIBRATION")
        self.assertTrue(result["estimator_pass"])
        self.assertTrue(result["no_update_invariant_pass"])

    def test_single_smoke_is_inconclusive_not_a_false_pass(self):
        result = SUMMARY.summarize(
            [self.record(1)], min_batches=20, frozen_radius=0.001
        )
        self.assertEqual(result["verdict"], "INCONCLUSIVE_NEED_MORE_BATCHES")

    def test_reversed_small_k_ranking_is_estimator_no_go(self):
        result = SUMMARY.summarize(
            [self.record(step, reverse_k4=True) for step in range(1, 21)],
            min_batches=20,
            frozen_radius=0.001,
        )
        self.assertEqual(result["verdict"], "ESTIMATOR_NO_GO")


if __name__ == "__main__":
    unittest.main()
