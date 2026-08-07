import unittest
import importlib.util
import sys
import types
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

# The rollout-format unit test does not need the full VeRL runtime. Keep this
# suite runnable in a lightweight CPU environment by stubbing its two imports.
verl_stub = types.ModuleType("verl")
verl_stub.DataProto = object
verl_utils_stub = types.ModuleType("verl.utils")
verl_tracking_stub = types.ModuleType("verl.utils.tracking")
verl_tracking_stub.Tracking = object
sys.modules.setdefault("verl", verl_stub)
sys.modules.setdefault("verl.utils", verl_utils_stub)
sys.modules.setdefault("verl.utils.tracking", verl_tracking_stub)

from search_r1.llm_agent.generation import LLMGenerationManager

MODULE_PATH = Path(__file__).resolve().parents[1] / "verl" / "trainer" / "ppo" / "eitr.py"
SPEC = importlib.util.spec_from_file_location("eitr_module", MODULE_PATH)
EITR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EITR)

build_sibling_probe_tensors = EITR.build_sibling_probe_tensors
build_online_probe_tensors = EITR.build_online_probe_tensors
induced_js_from_cached_effects = EITR.induced_js_from_cached_effects
probe_effect_diversity = EITR.probe_effect_diversity
update_dual_beta = EITR.update_dual_beta
validate_eitr_config = EITR.validate_eitr_config
validate_sibling_group_layout = EITR.validate_sibling_group_layout


class EITRMathTest(unittest.TestCase):
    def test_induced_js_is_zero_at_old_policy_and_has_finite_gradient(self):
        old = torch.zeros(1, 4)
        current = old.clone().requires_grad_(True)
        docs = torch.eye(4).unsqueeze(0)
        mask = torch.ones(1, 4, dtype=torch.bool)

        result = induced_js_from_cached_effects(
            current_seq_logp=current,
            old_seq_logp=old,
            doc_probs=docs,
            probe_mask=mask,
        )
        self.assertAlmostEqual(result["js"].item(), 0.0, places=7)
        result["js"].sum().backward()
        self.assertTrue(torch.isfinite(current.grad).all())

        shifted = torch.tensor([[2.0, -2.0, -2.0, -2.0]], requires_grad=True)
        shifted_result = induced_js_from_cached_effects(
            current_seq_logp=shifted,
            old_seq_logp=old,
            doc_probs=docs,
            probe_mask=mask,
        )
        self.assertGreater(shifted_result["js"].item(), 0.0)
        shifted_result["js"].sum().backward()
        self.assertTrue(torch.isfinite(shifted.grad).all())

    def test_dual_update_respects_target_and_bounds(self):
        self.assertAlmostEqual(update_dual_beta(0.1, 0.03, 0.01, 0.5, 10.0), 0.11)
        self.assertEqual(update_dual_beta(0.0, 0.0, 1.0, 1.0, 10.0), 0.0)
        self.assertEqual(update_dual_beta(9.9, 1.0, 0.0, 1.0, 10.0), 10.0)


class EITRProbeBatchTest(unittest.TestCase):
    @staticmethod
    def _record(action_ids, doc_prefix):
        return {
            "queries": [f"query {doc_prefix}"],
            "prefix_text": "",
            "action_token_ids": action_ids,
            "retrieval_effect": [
                {"doc_id": f"{doc_prefix}-a", "score": 1.0},
                {"doc_id": f"{doc_prefix}-b", "score": 0.5},
            ],
        }

    def test_builds_representative_probe_groups_and_dummy_slots(self):
        batch_size = 10
        prompt_width = 4
        response_width = 6
        prompts = torch.tensor([[0, 11, 12, 13]] * batch_size)
        attention_mask = torch.ones(batch_size, prompt_width + response_width, dtype=torch.long)
        responses = torch.zeros(batch_size, response_width, dtype=torch.long)
        old_log_probs = torch.full((batch_size, response_width), -0.25)
        records = []
        for index in range(batch_size):
            action_ids = [21 + index, 31 + index]
            responses[index, :2] = torch.tensor(action_ids)
            records.append(self._record(action_ids, f"d{index}"))

        # The second uid group has only three eligible search actions.
        records[8] = None
        records[9] = None
        uids = ["q0"] * 5 + ["q1"] * 5
        tensors, metrics = build_sibling_probe_tensors(
            prompts=prompts,
            attention_mask=attention_mask,
            responses=responses,
            old_log_probs=old_log_probs,
            uids=uids,
            records=records,
            pad_token_id=0,
            config={
                "probe_count": 4,
                "max_action_tokens": 8,
                "max_doc_support": 16,
                "retrieval_score_temperature": 0.1,
            },
        )

        self.assertEqual(tensors["eitr_state_slot"].nonzero().flatten().tolist(), [0, 5])
        self.assertEqual(tensors["eitr_state_valid"].nonzero().flatten().tolist(), [0])
        self.assertEqual(int(tensors["eitr_probe_valid"][0].sum()), 4)
        self.assertEqual(int(tensors["eitr_probe_valid"][5].sum()), 0)
        self.assertAlmostEqual(metrics["eitr/probe_state_coverage"], 0.5)
        self.assertTrue(torch.allclose(tensors["eitr_probe_doc_probs"][0].sum(dim=-1), torch.ones(4)))

    def test_layout_and_config_guards(self):
        validate_sibling_group_layout(["a"] * 5 + ["b"] * 5, n_agent=5, world_size=2)
        with self.assertRaises(ValueError):
            validate_sibling_group_layout(["a", "b"] * 5, n_agent=5, world_size=2)
        validate_eitr_config(
            {"max_probe_prompt_tokens": 16},
            n_agent=5,
            max_queries_per_turn=1,
            rollout_n=1,
            max_prompt_length=16,
        )
        with self.assertRaises(ValueError):
            validate_eitr_config({}, n_agent=3, max_queries_per_turn=1, rollout_n=1)
        with self.assertRaisesRegex(ValueError, "exact same state"):
            validate_eitr_config(
                {"max_probe_prompt_tokens": 15},
                n_agent=5,
                max_queries_per_turn=1,
                rollout_n=1,
                max_prompt_length=16,
            )
        with self.assertRaisesRegex(ValueError, "min_informative_state_rate"):
            validate_eitr_config(
                {"min_informative_state_rate": 1.1},
                n_agent=5,
                max_queries_per_turn=1,
                rollout_n=1,
            )

    def test_online_probes_share_one_fixed_state(self):
        prompts = torch.tensor([[0, 11, 12, 13]] * 5)
        attention_mask = torch.ones(5, 10, dtype=torch.long)
        responses = torch.tensor([[51, 52, 0, 0, 0, 0]] * 5)
        probes = []
        for index in range(4):
            probes.append({
                "query": f"q{index}",
                "action_token_ids": [60 + index, 70],
                "retrieval_effect": [{"doc_id": f"doc-{index}", "score": 1.0}],
            })
        groups = [{
            "state_prompt_token_ids": [11, 12, 13, 50],
            "extra_retrieval_calls": 3,
            "probes": probes,
        }] + [None] * 4
        tensors, metrics = build_online_probe_tensors(
            prompts=prompts,
            attention_mask=attention_mask,
            responses=responses,
            uids=["q0"] * 5,
            probe_groups=groups,
            pad_token_id=0,
            config={"probe_count": 4, "max_action_tokens": 8, "max_doc_support": 8},
        )
        self.assertEqual(metrics["eitr/probe_state_coverage"], 1.0)
        self.assertEqual(metrics["eitr/probe_retrieval_call_count"], 3.0)
        self.assertEqual(metrics["eitr/informative_probe_state_rate"], 1.0)
        self.assertEqual(metrics["eitr/probe_effect_top1_disagreement_rate"], 1.0)
        self.assertGreater(metrics["eitr/probe_effect_pairwise_js_max"], 0.0)
        self.assertEqual(tensors["eitr_state_slot"].nonzero().flatten().tolist(), [0])
        self.assertEqual(tensors["eitr_state_valid"].nonzero().flatten().tolist(), [0])
        state_prefixes = tensors["eitr_probe_input_ids"][0, :, :4]
        self.assertTrue(torch.equal(state_prefixes, state_prefixes[0].expand_as(state_prefixes)))

    def test_online_probe_state_is_never_silently_truncated(self):
        groups = [{
            "state_prompt_token_ids": [11, 12, 13, 14, 15],
            "extra_retrieval_calls": 3,
            "probes": [
                {
                    "query": f"q{index}",
                    "action_token_ids": [60 + index, 70],
                    "retrieval_effect": [{"doc_id": f"doc-{index}", "score": 1.0}],
                }
                for index in range(4)
            ],
        }] + [None] * 4

        with self.assertRaisesRegex(RuntimeError, "state_prompt_too_long"):
            build_online_probe_tensors(
                prompts=torch.tensor([[0, 11, 12, 13]] * 5),
                attention_mask=torch.ones(5, 10, dtype=torch.long),
                responses=torch.tensor([[51, 52, 0, 0, 0, 0]] * 5),
                uids=["q0"] * 5,
                probe_groups=groups,
                pad_token_id=0,
                config={
                    "probe_count": 4,
                    "max_action_tokens": 8,
                    "max_doc_support": 8,
                    "max_probe_prompt_tokens": 4,
                    "min_state_coverage": 1.0,
                },
            )

    def test_identical_retrieval_effects_are_not_informative(self):
        documents = torch.tensor([[[0.8, 0.2], [0.8, 0.2], [0.8, 0.2], [0.8, 0.2]]])
        result = probe_effect_diversity(
            documents,
            torch.ones(1, 4, dtype=torch.bool),
            informative_js_threshold=0.01,
        )
        self.assertFalse(result["informative"].item())
        self.assertAlmostEqual(result["state_max_js"].item(), 0.0, places=7)
        self.assertEqual(result["top1_disagreement"].item(), 0.0)

        probes = [
            {
                "query": f"paraphrase {index}",
                "action_token_ids": [60 + index, 70],
                "retrieval_effect": [
                    {"doc_id": "same-a", "score": 1.0},
                    {"doc_id": "same-b", "score": 0.5},
                ],
            }
            for index in range(4)
        ]
        groups = [{
            "state_prompt_token_ids": [11, 12, 13, 50],
            "extra_retrieval_calls": 3,
            "probes": probes,
        }] + [None] * 4
        with self.assertRaisesRegex(RuntimeError, "retrieval-effect diversity is too low"):
            build_online_probe_tensors(
                prompts=torch.tensor([[0, 11, 12, 13]] * 5),
                attention_mask=torch.ones(5, 10, dtype=torch.long),
                responses=torch.tensor([[51, 52, 0, 0, 0, 0]] * 5),
                uids=["q0"] * 5,
                probe_groups=groups,
                pad_token_id=0,
                config={
                    "probe_count": 4,
                    "max_action_tokens": 8,
                    "max_doc_support": 8,
                    "min_informative_state_rate": 0.1,
                },
            )


class SearchR1CompatibilityTest(unittest.TestCase):
    class _ContextSensitiveTokenizer:
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=False, **kwargs):
            table = {
                "<search>": [900],
                "</search>": [901],
                "<think>x</think>\n<search>": [101, 102, 103],
                "<think>x</think>\n<search>who wrote Hamlet</search>": [101, 102, 103, 104, 105],
            }
            return {"input_ids": table[text]}

        def decode(self, token_ids, skip_special_tokens=True):
            if token_ids[:5] == [101, 102, 103, 104, 105]:
                return "<think>x</think>\n<search>who wrote Hamlet</search>"
            if token_ids[:3] == [101, 102, 103]:
                return "<think>x</think>\n<search>"
            return ""

    def test_online_collector_handles_context_sensitive_search_tokens(self):
        manager = object.__new__(LLMGenerationManager)
        manager.tokenizer = self._ContextSensitiveTokenizer()
        manager.config = SimpleNamespace(
            eitr_probe_count=1,
            eitr_probe_oversample=0,
            eitr_n_agent=1,
            max_prompt_length=16,
        )
        manager._eitr_probe_groups = [None]
        manager._eitr_probe_collection_stats = Counter()
        manager._eitr_first_search_records = [{
            "queries": ["who wrote Hamlet"],
            "prefix_text": "<think>x</think>\n",
            "search_open_text": "<think>x</think>\n<search>",
            "action_text": "<think>x</think>\n<search>who wrote Hamlet</search>",
            "retrieval_effect": [{"doc_id": "hamlet", "score": 1.0}],
        }]
        rollings = SimpleNamespace(batch={
            "input_ids": torch.tensor([[0, 11, 12]]),
            "attention_mask": torch.tensor([[0, 1, 1]]),
        })

        manager._collect_eitr_same_state_probes(
            rollings=rollings,
            raw_responses_ids=torch.tensor([[101, 102, 103, 104, 105, 0]]),
        )

        group = manager._eitr_probe_groups[0]
        self.assertIsNotNone(group)
        self.assertEqual(group["state_prompt_token_ids"], [11, 12, 101, 102, 103])
        self.assertEqual(group["probes"][0]["action_token_ids"], [104, 105])
        self.assertEqual(manager._eitr_probe_collection_stats["state_group_collected"], 1)

    def test_disabled_eitr_preserves_single_query_observation_format(self):
        manager = object.__new__(LLMGenerationManager)
        manager.config = SimpleNamespace(collect_eitr_probes=False)
        manager.batch_search = lambda queries: [[{
            "document": {"contents": "Hamlet\nHamlet was written by William Shakespeare."},
            "score": 1.0,
        }]]

        observations, dones, valid_actions, searches = manager.execute_predictions(
            ["<think>I should search.</think><search>who wrote Hamlet</search>"],
            pad_token="<pad>",
            active_mask=[True],
        )

        self.assertEqual(
            observations,
            [
                "\n\n<information>"
                "Doc 1(Title: Hamlet) Hamlet was written by William Shakespeare."
                "</information>\n\n"
            ],
        )
        self.assertEqual(dones, [0])
        self.assertEqual(valid_actions, [1])
        self.assertEqual(searches, [1])

        final_observations, _, _, final_searches = manager.execute_predictions(
            ["<search>another query</search>"],
            pad_token="<pad>",
            active_mask=[True],
            do_search=False,
        )
        self.assertEqual(final_observations, ["\n\n<information></information>\n\n"])
        self.assertEqual(final_searches, [1])


if __name__ == "__main__":
    unittest.main()
