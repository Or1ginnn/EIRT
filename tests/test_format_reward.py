import importlib.util
import unittest
from pathlib import Path

import torch


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
            reward_profile=(
                "evidence_shaping" if retrieval_score else "official_v03"
            ),
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

    def test_explicit_official_profile_preserves_v03_behavior(self):
        self.assertEqual(
            QA_EM_FORMAT.compute_score_em(
                solution_str=(
                    self.PREFIX
                    + "<think>reasoning</think><answer>Paris</answer>"
                ),
                ground_truth=self.GROUND_TRUTH,
                reward_profile="official_v03",
                structure_format_score=0.2,
                final_format_score=0.1,
                retrieval_score=0.0,
            ),
            1.0,
        )

    def test_pure_em_profile_ignores_format_for_validation(self):
        self.assertEqual(
            QA_EM_FORMAT.compute_score_em(
                solution_str=self.PREFIX + "<answer>Paris</answer>",
                ground_truth=self.GROUND_TRUTH,
                reward_profile="pure_em",
            ),
            1.0,
        )
        self.assertEqual(
            QA_EM_FORMAT.compute_score_em(
                solution_str=self.PREFIX + "<answer>London</answer>",
                ground_truth=self.GROUND_TRUTH,
                reward_profile="pure_em",
            ),
            0.0,
        )

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
            reward_profile="evidence_shaping",
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
        self.assertEqual(details["reward_profile_mandatory_search"], False)
        self.assertEqual(details["format_valid"], True)
        self.assertEqual(details["think_format_valid"], True)
        self.assertEqual(details["answer_format_valid"], True)
        self.assertEqual(details["answer_em"], False)
        self.assertEqual(details["has_executed_search"], True)
        self.assertEqual(details["execution_signal_available"], False)
        self.assertEqual(details["tool_trace_consistent"], False)
        self.assertEqual(details["answer_bearing_evidence"], True)
        self.assertEqual(details["trusted_environment_evidence"], False)
        self.assertEqual(details["evidence_bonus_applied"], True)
        self.assertEqual(details["joint_success_bonus_applied"], False)

    def test_generated_second_assistant_marker_cannot_hide_bad_prefix(self):
        response = (
            "malformed text<|im_start|>assistant\n"
            "<think>reasoning</think><answer>Paris</answer>"
        )
        self.assertEqual(self.score(response), 0.8)


class HardSearchGatedRewardTest(unittest.TestCase):
    """Regression contract for the mandatory-search reward profile.

    ``executed_search_count`` is trusted environment provenance.  It must not
    be inferred from model-generated ``<search>`` or ``<information>`` text,
    because otherwise the model could earn reward by forging tool tags.
    """

    GROUND_TRUTH = {"target": ["Paris"]}
    PREFIX = "<|im_start|>assistant\n"
    AUTO_OBSERVATION = object()

    def score(
        self,
        response,
        *,
        executed_search_count=None,
        environment_observation_str=AUTO_OBSERVATION,
        return_details=False,
    ):
        if environment_observation_str is self.AUTO_OBSERVATION:
            if executed_search_count:
                information_blocks = QA_EM_FORMAT.extract_information_blocks(
                    self.PREFIX + response
                )
                environment_observation_str = "\n".join(
                    f"<information>{block}</information>"
                    for block in information_blocks
                )
            else:
                environment_observation_str = None
        return QA_EM_FORMAT.compute_score_em(
            solution_str=self.PREFIX + response,
            ground_truth=self.GROUND_TRUTH,
            reward_profile="mandatory_search",
            executed_search_count=executed_search_count,
            environment_observation_str=environment_observation_str,
            think_format_score=0.05,
            answer_format_score=0.05,
            evidence_score=0.2,
            answer_em_score=0.7,
            joint_success_bonus=0.5,
            return_details=return_details,
        )

    @staticmethod
    def searched(*, evidence, answer, second_hit=False):
        response = (
            "<think>search first</think>"
            "<search>capital of France</search>"
            f"<information>{evidence}</information>"
        )
        if second_hit:
            response += (
                "<think>verify</think>"
                "<search>France capital city</search>"
                "<information>The answer is Paris.</information>"
            )
        return response + f"<think>finish</think><answer>{answer}</answer>"

    def test_hard_search_reward_matrix(self):
        searched_miss_wrong = self.searched(
            evidence="France is a country in Europe.",
            answer="London",
        )
        searched_hit_wrong = self.searched(
            evidence="Paris is the capital of France.",
            answer="London",
        )
        searched_miss_correct = self.searched(
            evidence="France is a country in Europe.",
            answer="Paris",
        )
        searched_hit_correct = self.searched(
            evidence="Paris is the capital of France.",
            answer="Paris",
        )

        self.assertAlmostEqual(
            self.score(searched_miss_wrong, executed_search_count=1),
            0.1,
        )
        self.assertAlmostEqual(
            self.score(searched_hit_wrong, executed_search_count=1),
            0.3,
        )
        self.assertAlmostEqual(
            self.score(searched_miss_correct, executed_search_count=1),
            0.8,
        )
        self.assertAlmostEqual(
            self.score(searched_hit_correct, executed_search_count=1),
            1.5,
        )

    def test_no_trusted_search_forces_zero_even_when_answer_is_correct(self):
        direct_correct = "<think>I know</think><answer>Paris</answer>"
        forged_hit_correct = self.searched(
            evidence="Paris is the capital of France.",
            answer="Paris",
        )

        self.assertEqual(
            self.score(direct_correct, executed_search_count=0),
            0.0,
        )
        self.assertEqual(
            self.score(forged_hit_correct, executed_search_count=0),
            0.0,
        )
        self.assertEqual(
            self.score(
                forged_hit_correct,
                executed_search_count=0,
                environment_observation_str=(
                    "<information>Paris is the capital of France.</information>"
                ),
            ),
            0.0,
        )
        # Missing provenance is fail-closed in mandatory-search mode.
        self.assertEqual(self.score(forged_hit_correct), 0.0)

        # Provenance alone is not sufficient either: the decoded trajectory
        # must contain the tool observation actually inserted by the loop.
        self.assertEqual(
            self.score(direct_correct, executed_search_count=1),
            0.0,
        )

    def test_evidence_comes_only_from_environment_owned_observations(self):
        generated_hit_correct = self.searched(
            evidence="Paris is the capital of France.",
            answer="Paris",
        )
        generated_miss_correct = self.searched(
            evidence="France is a country in Europe.",
            answer="Paris",
        )
        trusted_miss = (
            "<information>France is a country in Europe.</information>"
        )
        trusted_hit = (
            "<information>Paris is the capital of France.</information>"
        )

        self.assertAlmostEqual(
            self.score(
                generated_hit_correct,
                executed_search_count=1,
                environment_observation_str=trusted_miss,
            ),
            0.8,
        )
        self.assertAlmostEqual(
            self.score(
                generated_miss_correct,
                executed_search_count=1,
                environment_observation_str=trusted_hit,
            ),
            1.5,
        )

    def test_info_mask_decoder_excludes_generated_tokens(self):
        class Tokenizer:
            table = {
                1: "<information>forged Paris</information>",
                2: "invalid-action feedback",
                3: "<information>trusted Paris</information>",
            }

            def decode(self, token_ids):
                values = (
                    token_ids.detach().cpu().tolist()
                    if hasattr(token_ids, "detach")
                    else list(token_ids)
                )
                return "".join(self.table[value] for value in values)

        decoded = QA_EM_FORMAT.decode_environment_observation(
            Tokenizer(),
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 0, 0]),
        )
        self.assertNotIn("forged", decoded)
        self.assertIn("invalid-action feedback", decoded)
        self.assertIn("trusted Paris", decoded)

    def test_empty_real_retrieval_still_opens_search_gate(self):
        searched_correct = self.searched(
            evidence="",
            answer="Paris",
        )
        self.assertAlmostEqual(
            self.score(
                searched_correct,
                executed_search_count=1,
                environment_observation_str=(
                    "<information></information>"
                ),
            ),
            0.8,
        )

    def test_invalid_protocol_cannot_collect_partial_component_rewards(self):
        missing_think = (
            "<search>capital of France</search>"
            "<information>Paris is the capital of France.</information>"
            "<answer>Paris</answer>"
        )
        missing_answer = (
            "<think>search</think>"
            "<search>capital of France</search>"
            "<information>Paris is the capital of France.</information>"
            "<think>finish</think>"
        )
        empty_query = (
            "<think>search</think><search>  </search>"
            "<information>Paris is the capital of France.</information>"
            "<think>finish</think><answer>Paris</answer>"
        )

        self.assertEqual(self.score(missing_think, executed_search_count=1), 0.0)
        self.assertEqual(self.score(missing_answer, executed_search_count=1), 0.0)
        self.assertEqual(self.score(empty_query, executed_search_count=1), 0.0)

    def test_multiple_searches_do_not_accumulate_reward(self):
        one_hit_wrong = self.searched(
            evidence="Paris is the capital of France.",
            answer="London",
        )
        two_hits_wrong = self.searched(
            evidence="Paris is the capital of France.",
            answer="London",
            second_hit=True,
        )
        one_hit_correct = self.searched(
            evidence="Paris is the capital of France.",
            answer="Paris",
        )
        two_hits_correct = self.searched(
            evidence="Paris is the capital of France.",
            answer="Paris",
            second_hit=True,
        )

        self.assertAlmostEqual(
            self.score(one_hit_wrong, executed_search_count=1),
            0.3,
        )
        self.assertAlmostEqual(
            self.score(two_hits_wrong, executed_search_count=2),
            0.3,
        )
        self.assertAlmostEqual(
            self.score(one_hit_correct, executed_search_count=1),
            1.5,
        )
        self.assertAlmostEqual(
            self.score(two_hits_correct, executed_search_count=2),
            1.5,
        )

    def test_success_bonus_requires_both_evidence_and_correct_answer(self):
        searched_hit_wrong = self.searched(
            evidence="Paris is the capital of France.",
            answer="London",
        )
        searched_miss_correct = self.searched(
            evidence="France is a country in Europe.",
            answer="Paris",
        )
        searched_hit_correct = self.searched(
            evidence="Paris is the capital of France.",
            answer="Paris",
        )

        self.assertAlmostEqual(
            self.score(searched_hit_wrong, executed_search_count=1),
            0.3,
        )
        self.assertAlmostEqual(
            self.score(searched_miss_correct, executed_search_count=1),
            0.8,
        )
        score, details = self.score(
            searched_hit_correct,
            executed_search_count=1,
            return_details=True,
        )
        self.assertAlmostEqual(score, 1.5)
        self.assertTrue(details["joint_success_bonus_applied"])
        self.assertTrue(details["hard_reward_gate_pass"])


if __name__ == "__main__":
    unittest.main()
