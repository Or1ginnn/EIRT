import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "verl"
    / "utils"
    / "reward_score"
    / "qa_em_format.py"
)
SPEC = importlib.util.spec_from_file_location("qa_em_format_module", MODULE_PATH)
QA_EM_FORMAT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QA_EM_FORMAT)


class SearchR1V03RewardTest(unittest.TestCase):
    GROUND_TRUTH = {"target": ["Paris"]}
    PREFIX = "prompt example <answer>demo</answer><|im_start|>assistant\n"

    def score(self, response):
        return QA_EM_FORMAT.compute_score_em(
            solution_str=self.PREFIX + response,
            ground_truth=self.GROUND_TRUTH,
            structure_format_score=0.2,
            final_format_score=0.1,
            retrieval_score=0,
        )

    def test_official_v03_reward_levels(self):
        self.assertEqual(
            self.score("<think>reasoning</think><answer>Paris</answer>"),
            1.0,
        )
        self.assertEqual(self.score("<answer>Paris</answer>"), 0.8)
        self.assertEqual(
            self.score("<think>reasoning</think><answer>London</answer>"),
            0.2,
        )
        self.assertEqual(self.score("<answer>London</answer>"), 0.1)
        self.assertEqual(self.score("London"), 0.0)


if __name__ == "__main__":
    unittest.main()
