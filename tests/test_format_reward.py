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
    PREFIX = (
        "instructions inside <answer> and </answer>; "
        "for example <answer>Beijing</answer>"
        "<|im_start|>assistant\n"
    )

    def score(self, response, retrieval_score=0, return_details=False):
        return QA_EM_FORMAT.compute_score_em(
            solution_str=self.PREFIX + response,
            ground_truth=self.GROUND_TRUTH,
            structure_format_score=0.2,
            final_format_score=0.1,
            retrieval_score=retrieval_score,
            return_details=return_details,
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

    def test_prompt_answer_examples_are_never_scored(self):
        self.assertEqual(self.score("<think>I do not know yet</think>"), 0.0)
        self.assertEqual(self.score(""), 0.0)

    def test_generated_answer_is_used_despite_multiple_prompt_examples(self):
        self.assertEqual(self.score("<answer>London</answer>"), 0.1)
        self.assertEqual(self.score("<answer>Paris</answer>"), 0.8)

    def test_unclosed_generated_answer_gets_zero(self):
        self.assertEqual(self.score("<answer>Paris"), 0.0)

    def test_empty_generated_answer_gets_zero(self):
        self.assertEqual(self.score("<answer></answer>"), 0.0)
        self.assertEqual(
            self.score("<think>reasoning</think><answer>   </answer>"),
            0.0,
        )

    def test_evidence_shaping_rewards_useful_retrieval_once(self):
        searched_hit = (
            "<think>search</think>"
            "<search>capital of France</search>"
            "<information>Paris is the capital of France.</information>"
            "<think>guess</think><answer>London</answer>"
        )
        searched_twice = (
            "<think>search</think>"
            "<search>capital of France</search>"
            "<information>Paris is the capital of France.</information>"
            "<think>search again</think>"
            "<search>France capital city</search>"
            "<information>The answer is Paris.</information>"
            "<think>guess</think><answer>London</answer>"
        )
        self.assertEqual(self.score(searched_hit, retrieval_score=0), 0.2)
        self.assertAlmostEqual(self.score(searched_hit, retrieval_score=0.1), 0.3)
        self.assertAlmostEqual(self.score(searched_twice, retrieval_score=0.1), 0.3)

    def test_evidence_shaping_does_not_reward_search_count_or_direct_answer(self):
        searched_miss = (
            "<think>search</think>"
            "<search>capital of France</search>"
            "<information>France is a country in Europe.</information>"
            "<think>guess</think><answer>London</answer>"
        )
        direct_wrong = "<think>guess</think><answer>London</answer>"
        direct_correct = "<think>know</think><answer>Paris</answer>"
        searched_correct = (
            "<think>search</think><search>capital</search>"
            "<information>Paris is the capital.</information>"
            "<think>answer</think><answer>Paris</answer>"
        )
        self.assertEqual(self.score(searched_miss, retrieval_score=0.1), 0.2)
        self.assertEqual(self.score(direct_wrong, retrieval_score=0.1), 0.2)
        self.assertEqual(self.score(direct_correct, retrieval_score=0.1), 1.0)
        self.assertEqual(self.score(searched_correct, retrieval_score=0.1), 1.0)

    def test_orphan_or_empty_information_never_receives_evidence_bonus(self):
        orphan_information = (
            "<think>guess</think><information>Paris</information>"
            "<answer>London</answer>"
        )
        empty_information = (
            "<think>search</think><search>capital</search>"
            "<information>   </information>"
            "<think>guess</think><answer>London</answer>"
        )
        self.assertNotEqual(
            self.score(orphan_information, retrieval_score=0.1),
            0.3,
        )
        self.assertEqual(self.score(empty_information, retrieval_score=0.1), 0.2)

    def test_evidence_match_uses_complete_normalized_token_sequence(self):
        false_substring = (
            "<think>search</think><search>US policy</search>"
            "<information>Russia announced a policy.</information>"
            "<think>guess</think><answer>wrong</answer>"
        )
        score = QA_EM_FORMAT.compute_score_em(
            solution_str=self.PREFIX + false_substring,
            ground_truth={"target": ["US"]},
            structure_format_score=0.2,
            final_format_score=0.1,
            retrieval_score=0.1,
        )
        self.assertEqual(score, 0.2)

        unicode_punctuation = (
            "<think>search</think><search>Paris</search>"
            "<information>Paris—the city—is in France.</information>"
            "<think>guess</think><answer>wrong</answer>"
        )
        self.assertAlmostEqual(
            self.score(unicode_punctuation, retrieval_score=0.1),
            0.3,
        )

    def test_evidence_reward_is_capped_by_correct_answer_score(self):
        response = (
            "<think>search</think><search>capital</search>"
            "<information>Paris is the capital.</information>"
            "<think>guess</think><answer>London</answer>"
        )
        self.assertEqual(self.score(response, retrieval_score=2.0), 1.0)

    def test_evidence_details_expose_only_trajectory_level_events(self):
        response = (
            "<think>search</think><search>capital</search>"
            "<information>Paris is the capital.</information>"
            "<think>guess</think><answer>London</answer>"
        )
        score, details = self.score(
            response,
            retrieval_score=0.1,
            return_details=True,
        )
        self.assertAlmostEqual(score, 0.3)
        self.assertEqual(
            details,
            {
                "format_valid": True,
                "answer_em": False,
                "has_executed_search": True,
                "answer_bearing_evidence": True,
                "evidence_bonus_applied": True,
            },
        )

    def test_generated_second_assistant_marker_cannot_hide_bad_prefix(self):
        response = (
            "malformed text<|im_start|>assistant\n"
            "<think>reasoning</think><answer>Paris</answer>"
        )
        self.assertEqual(self.score(response), 0.8)


if __name__ == "__main__":
    unittest.main()
