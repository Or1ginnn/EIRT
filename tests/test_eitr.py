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

import search_r1.llm_agent.generation as generation_module
from search_r1.llm_agent.generation import LLMGenerationManager

MODULE_PATH = Path(__file__).resolve().parents[1] / "verl" / "trainer" / "ppo" / "eitr.py"
SPEC = importlib.util.spec_from_file_location("eitr_module", MODULE_PATH)
EITR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EITR)

TRACKING_MODULE_PATH = Path(__file__).resolve().parents[1] / "verl" / "utils" / "tracking.py"
TRACKING_SPEC = importlib.util.spec_from_file_location("tracking_module", TRACKING_MODULE_PATH)
TRACKING_MODULE = importlib.util.module_from_spec(TRACKING_SPEC)
TRACKING_SPEC.loader.exec_module(TRACKING_MODULE)

build_sibling_probe_tensors = EITR.build_sibling_probe_tensors
build_online_probe_tensors = EITR.build_online_probe_tensors
induced_js_from_cached_effects = EITR.induced_js_from_cached_effects
probe_effect_diversity = EITR.probe_effect_diversity
resolve_eitr_mode = EITR.resolve_eitr_mode
eitr_probe_enabled_for_pass = EITR.eitr_probe_enabled_for_pass
eitr_loss_enabled_for_pass = EITR.eitr_loss_enabled_for_pass
validate_eitr_optimization_schedule = EITR.validate_eitr_optimization_schedule
validate_eitr_config = EITR.validate_eitr_config
validate_sibling_group_layout = EITR.validate_sibling_group_layout


class TrackingFlushTest(unittest.TestCase):
    def test_finish_flushes_backends_once(self):
        class FinishBackend:
            def __init__(self):
                self.calls = 0

            def finish(self):
                self.calls += 1

        class FlushBackend:
            def __init__(self):
                self.calls = 0

            def flush(self):
                self.calls += 1

        finish_backend = FinishBackend()
        flush_backend = FlushBackend()
        tracking = TRACKING_MODULE.Tracking.__new__(TRACKING_MODULE.Tracking)
        tracking._finished = False
        tracking.logger = {
            "wandb": finish_backend,
            "console": flush_backend,
        }

        tracking.finish()
        tracking.finish()

        self.assertEqual(finish_backend.calls, 1)
        self.assertEqual(flush_backend.calls, 1)

    def test_smoke_logs_to_console_and_wandb(self):
        runner = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "train"
            / "train_eitr_nq_gate_c_smoke.sh"
        ).read_text()
        self.assertIn("trainer.logger=\"['console','wandb']\"", runner)


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

    def test_phase2_modes_and_correction_schedule(self):
        self.assertEqual(resolve_eitr_mode({}), "off")
        self.assertEqual(resolve_eitr_mode({"enabled": True}), "eitr")
        self.assertEqual(resolve_eitr_mode({"mode": "probe_only"}), "probe_only")
        with self.assertRaisesRegex(ValueError, "Unsupported EITR mode"):
            resolve_eitr_mode({"mode": "unknown"})

        self.assertFalse(eitr_probe_enabled_for_pass({"mode": "eitr"}, 0, grpo_passes=1))
        self.assertTrue(eitr_probe_enabled_for_pass({"mode": "eitr"}, 1, grpo_passes=1))
        self.assertFalse(eitr_loss_enabled_for_pass({"mode": "probe_only"}, 1, grpo_passes=1))
        self.assertTrue(eitr_loss_enabled_for_pass({"mode": "eitr"}, 1, grpo_passes=1))
        validate_eitr_optimization_schedule({"mode": "eitr"}, ppo_epochs=1, correction_passes=1)

    def test_variable_effective_k_has_finite_gradient(self):
        documents = torch.eye(4).unsqueeze(0)
        old = torch.zeros(1, 4)
        for effective_k in (2, 3, 4):
            current = torch.tensor([[2.0, -1.0, -2.0, -3.0]], requires_grad=True)
            mask = torch.zeros(1, 4, dtype=torch.bool)
            mask[:, :effective_k] = True
            result = induced_js_from_cached_effects(
                current_seq_logp=current,
                old_seq_logp=old,
                doc_probs=documents,
                probe_mask=mask,
            )
            self.assertTrue(torch.isfinite(result["js"]).all())
            result["js"].sum().backward()
            self.assertTrue(torch.isfinite(current.grad).all())


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

        # The second uid group has only three eligible search actions. Phase 2
        # keeps it because K_eff>=2 is sufficient.
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
        self.assertEqual(tensors["eitr_state_valid"].nonzero().flatten().tolist(), [0, 5])
        self.assertEqual(int(tensors["eitr_probe_valid"][0].sum()), 4)
        self.assertEqual(int(tensors["eitr_probe_valid"][5].sum()), 3)
        self.assertAlmostEqual(metrics["eitr/probe_state_coverage"], 1.0)
        self.assertTrue(torch.allclose(tensors["eitr_probe_doc_probs"][0].sum(dim=-1), torch.ones(4)))

    def test_layout_and_config_guards(self):
        validate_sibling_group_layout(["a"] * 5 + ["b"] * 5, n_agent=5, world_size=2)
        with self.assertRaises(ValueError):
            validate_sibling_group_layout(["a", "b"] * 5, n_agent=5, world_size=2)
        validate_eitr_config(
            {"max_probe_prompt_tokens": 16, "max_query_tokens": 4},
            n_agent=5,
            max_queries_per_turn=1,
            rollout_n=1,
            rollout_response_length=4,
            max_prompt_length=16,
            rollout_max_model_len=20,
            rollout_top_p=1.0,
            rollout_top_k=-1,
        )
        validate_eitr_config({}, n_agent=3, max_queries_per_turn=1, rollout_n=1)
        with self.assertRaisesRegex(ValueError, "probe_probability"):
            validate_eitr_config(
                {"probe_probability": 1.1},
                n_agent=3,
                max_queries_per_turn=1,
                rollout_n=1,
            )
        with self.assertRaisesRegex(ValueError, "exact same state"):
            validate_eitr_config(
                {"max_probe_prompt_tokens": 15},
                n_agent=5,
                max_queries_per_turn=1,
                rollout_n=1,
                max_prompt_length=16,
            )
        with self.assertRaisesRegex(ValueError, "max_model_len"):
            validate_eitr_config(
                {"max_probe_prompt_tokens": 16, "max_query_tokens": 4},
                n_agent=5,
                max_queries_per_turn=1,
                rollout_n=1,
                max_prompt_length=16,
                rollout_max_model_len=19,
            )
        with self.assertRaisesRegex(ValueError, "max_query_tokens must equal"):
            validate_eitr_config(
                {"max_query_tokens": 3, "max_action_tokens": 3},
                n_agent=5,
                max_queries_per_turn=1,
                rollout_n=1,
                rollout_response_length=4,
            )
        with self.assertRaisesRegex(ValueError, "max_action_tokens"):
            validate_eitr_config(
                {"max_query_tokens": 4, "max_action_tokens": 3},
                n_agent=5,
                max_queries_per_turn=1,
                rollout_n=1,
                rollout_response_length=4,
            )
        with self.assertRaisesRegex(ValueError, "top_p=1.0"):
            validate_eitr_config(
                {},
                n_agent=5,
                max_queries_per_turn=1,
                rollout_n=1,
                rollout_top_p=0.95,
                rollout_top_k=-1,
            )
        with self.assertRaisesRegex(ValueError, "top_k=-1"):
            validate_eitr_config(
                {},
                n_agent=5,
                max_queries_per_turn=1,
                rollout_n=1,
                rollout_top_p=1.0,
                rollout_top_k=20,
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
        self.assertEqual(metrics["eitr/probe_state_coverage"], 0.2)
        self.assertEqual(metrics["eitr/probe_retrieval_call_count"], 3.0)
        self.assertEqual(metrics["eitr/informative_probe_state_rate"], 1.0)
        self.assertEqual(metrics["eitr/probe_effect_top1_disagreement_rate"], 1.0)
        self.assertGreater(metrics["eitr/probe_effect_pairwise_js_max"], 0.0)
        self.assertEqual(tensors["eitr_state_slot"].nonzero().flatten().tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(tensors["eitr_state_valid"].nonzero().flatten().tolist(), [0])
        state_prefixes = tensors["eitr_probe_input_ids"][0, :, :4]
        self.assertTrue(torch.equal(state_prefixes, state_prefixes[0].expand_as(state_prefixes)))

    def test_online_variable_k_and_zero_valid_states_do_not_abort(self):
        probes = [
            {
                "query": f"q{index}",
                "action_token_ids": [60 + index, 70],
                "retrieval_effect": [{"doc_id": f"doc-{index}", "score": 1.0}],
            }
            for index in range(2)
        ]
        groups = [{
            "state_prompt_token_ids": [11, 12, 13, 50],
            "extra_retrieval_calls": 1,
            "probes": probes,
        }] + [None] * 4
        tensors, metrics = build_online_probe_tensors(
            prompts=torch.tensor([[0, 11, 12, 13]] * 5),
            attention_mask=torch.ones(5, 10, dtype=torch.long),
            responses=torch.tensor([[51, 52, 0, 0, 0, 0]] * 5),
            uids=["q0"] * 5,
            probe_groups=groups,
            pad_token_id=0,
            config={
                "probe_count": 4,
                "min_valid_probe_count": 2,
                "max_action_tokens": 8,
                "max_doc_support": 8,
            },
        )
        self.assertEqual(int(tensors["eitr_state_valid"].sum()), 1)
        self.assertEqual(int(tensors["eitr_probe_valid"][0].sum()), 2)
        self.assertEqual(metrics["eitr/partial_probe_state_count"], 1.0)

        empty_tensors, empty_metrics = build_online_probe_tensors(
            prompts=torch.tensor([[0, 11, 12, 13]] * 5),
            attention_mask=torch.ones(5, 10, dtype=torch.long),
            responses=torch.tensor([[51, 52, 0, 0, 0, 0]] * 5),
            uids=["q0"] * 5,
            probe_groups=[None] * 5,
            pad_token_id=0,
            config={"probe_count": 4, "max_action_tokens": 8, "max_doc_support": 8},
        )
        self.assertEqual(int(empty_tensors["eitr_state_valid"].sum()), 0)
        self.assertEqual(int(empty_tensors["eitr_state_slot"].sum()), 0)
        self.assertEqual(empty_metrics["eitr/probe_state_count"], 0.0)

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

        tensors, metrics = build_online_probe_tensors(
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
        self.assertEqual(int(tensors["eitr_state_valid"].sum()), 0)
        self.assertEqual(int(tensors["eitr_state_slot"].sum()), 0)
        self.assertEqual(metrics["eitr/probe_state_coverage_below_threshold"], 1.0)

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
        _, metrics = build_online_probe_tensors(
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
        self.assertEqual(metrics["eitr/informative_probe_state_rate"], 0.0)
        self.assertEqual(metrics["eitr/informative_probe_state_rate_below_threshold"], 1.0)


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
            eitr_max_probe_prompt_tokens=16,
            max_prompt_length=16,
            max_response_length=6,
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
        self.assertEqual(group["query_token_budget"], 3)
        self.assertEqual(manager._eitr_probe_collection_stats["state_group_collected"], 1)

    def test_unusable_first_search_does_not_consume_primary_probe_state(self):
        manager = object.__new__(LLMGenerationManager)
        manager.tokenizer = self._ContextSensitiveTokenizer()
        manager.config = SimpleNamespace(
            eitr_probe_count=1,
            eitr_probe_oversample=0,
            eitr_n_agent=1,
            eitr_max_probe_prompt_tokens=16,
            max_prompt_length=16,
            max_response_length=6,
        )
        manager._eitr_probe_groups = [None]
        manager._eitr_probe_state_groups = []
        manager._eitr_primary_search_seen = [False]
        manager._eitr_probe_collection_stats = Counter()
        rollings = SimpleNamespace(batch={
            "input_ids": torch.tensor([[0, 11, 12]]),
            "attention_mask": torch.tensor([[0, 1, 1]]),
        })
        base_record = {
            "queries": ["who wrote Hamlet"],
            "prefix_text": "<think>x</think>\n",
            "search_open_text": "<think>x</think>\n<search>",
            "action_text": "<think>x</think>\n<search>who wrote Hamlet</search>",
            "retrieval_effect": [],
        }
        manager._eitr_current_search_records = {0: base_record}
        manager._collect_eitr_same_state_probes(
            rollings=rollings,
            raw_responses_ids=torch.tensor([[101, 102, 103, 104, 105, 0]]),
            turn_index=0,
        )
        self.assertFalse(manager._eitr_primary_search_seen[0])

        manager._eitr_current_search_records = {0: {
            **base_record,
            "turn_index": 1,
            "retrieval_effect": [{"doc_id": "hamlet", "score": 1.0}],
        }}
        manager._collect_eitr_same_state_probes(
            rollings=rollings,
            raw_responses_ids=torch.tensor([[101, 102, 103, 104, 105, 0]]),
            turn_index=1,
        )
        group = manager._eitr_probe_groups[0]
        self.assertIsNotNone(group)
        self.assertFalse(group["deferred"])
        self.assertEqual(group["turn_index"], 1)

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

        final_observations, _, final_valid_actions, final_searches = manager.execute_predictions(
            ["<search>another query</search>"],
            pad_token="<pad>",
            active_mask=[True],
            do_search=False,
        )
        self.assertEqual(final_observations, ["\n\n<information></information>\n\n"])
        self.assertEqual(final_valid_actions, [0])
        self.assertEqual(final_searches, [0])

    def test_complete_trajectory_uses_its_own_length_limit(self):
        manager = object.__new__(LLMGenerationManager)
        manager.tokenizer = SimpleNamespace(pad_token_id=0)
        manager.config = SimpleNamespace(
            max_prompt_length=4,
            max_trajectory_length=8,
        )
        manager.tensor_fn = SimpleNamespace(
            create_attention_mask=lambda values: (values != 0).long()
        )
        right_side = {
            "responses": torch.tensor([[1, 2, 3, 4, 5]]),
            "responses_with_info_mask": torch.tensor([[1, 2, 3, 4, 5]]),
        }
        updated = manager._update_right_side(
            right_side,
            torch.tensor([[6, 7, 8, 9]]),
        )
        self.assertEqual(updated["responses"].tolist(), [[1, 2, 3, 4, 5, 6, 7, 8]])

    def test_empty_closed_search_is_not_a_valid_action(self):
        manager = object.__new__(LLMGenerationManager)
        actions, contents = manager.postprocess_predictions(
            ["<search></search>", "<search>   </search>", "<search>valid query</search>"]
        )
        self.assertEqual(actions, [None, None, "search"])
        self.assertEqual(contents, ["", "", "valid query"])

    def test_probe_close_tag_uses_progressive_decoding(self):
        class ProbeTokenizer:
            def decode(self, token_ids, skip_special_tokens=True):
                mapping = {
                    (201,): "who wrote Hamlet",
                    (201, 202): "who wrote Hamlet</search>",
                    (202,): "</search>",
                    (203,): "symbols || <inside>",
                    (203, 202): "symbols || <inside></search>",
                }
                return mapping.get(tuple(token_ids), "")

        manager = object.__new__(LLMGenerationManager)
        manager.tokenizer = ProbeTokenizer()
        query, action_ids, reason = manager._parse_eitr_probe_response([201, 202, 0])
        self.assertEqual(query, "who wrote Hamlet")
        self.assertEqual(action_ids, [201, 202])
        self.assertIsNone(reason)

        query, action_ids, reason = manager._parse_eitr_probe_response([201, 0])
        self.assertIsNone(query)
        self.assertIsNone(action_ids)
        self.assertEqual(reason, "missing_close_tag")

        query, action_ids, reason = manager._parse_eitr_probe_response([202])
        self.assertIsNone(query)
        self.assertIsNone(action_ids)
        self.assertEqual(reason, "empty_query")

        query, action_ids, reason = manager._parse_eitr_probe_response([203, 202])
        self.assertEqual(query, "symbols || <inside>")
        self.assertEqual(action_ids, [203, 202])
        self.assertIsNone(reason)

    def test_same_state_probe_rounds_use_distinct_seeds(self):
        class FakeDataProto:
            @staticmethod
            def from_dict(batch):
                return SimpleNamespace(batch=batch, meta_info={})

        class ProbeTokenizer:
            pad_token_id = 0

            def decode(self, token_ids, skip_special_tokens=True):
                if len(token_ids) >= 2 and int(token_ids[1]) == 999:
                    return f"query-{int(token_ids[0])}</search>"
                return ""

        manager = object.__new__(LLMGenerationManager)
        manager.tokenizer = ProbeTokenizer()
        manager.tensor_fn = SimpleNamespace(
            create_position_ids=lambda mask: (torch.cumsum(mask, dim=-1) - 1).clamp_min(0)
        )
        manager.config = SimpleNamespace(
            eitr_probe_count=4,
            eitr_probe_oversample=0,
            eitr_max_query_tokens=3,
            max_response_length=3,
            eitr_probe_seed=123,
        )
        group = {
            "source_index": 0,
            "state_prompt_token_ids": [11, 12, 13],
            "query_token_budget": 3,
            "extra_retrieval_calls": 0,
            "probes": [{
                "query": "real-query",
                "action_token_ids": [101, 999],
                "retrieval_effect": [{"doc_id": "real-doc", "score": 1.0}],
            }],
        }
        manager._eitr_pending_probe_groups = [group]
        manager._eitr_probe_groups = [None]
        manager._eitr_probe_collection_stats = Counter()
        manager._eitr_probe_call_index = 0
        seen_seeds = []
        seen_max_tokens = []

        def fake_generate(prompts):
            seen_seeds.append(prompts.meta_info["sampling_params"]["seed"])
            seen_max_tokens.append(prompts.meta_info["sampling_params"]["max_tokens"])
            token = 301 + len(seen_seeds)
            return SimpleNamespace(batch={
                "responses": torch.tensor([[token, 999, 0]], dtype=torch.long)
            })

        manager._generate_with_gpu_padding = fake_generate
        manager.batch_search = lambda queries: [[{
            "document": {"id": query},
            "score": 1.0,
        }] for query in queries]

        original_data_proto = generation_module.DataProto
        generation_module.DataProto = FakeDataProto
        try:
            manager._finalize_eitr_same_state_probes()
        finally:
            generation_module.DataProto = original_data_proto

        self.assertEqual(seen_seeds, [123, 124, 125])
        self.assertEqual(seen_max_tokens, [3, 3, 3])
        self.assertEqual(manager._eitr_probe_collection_stats["probe_generation_call_count"], 3)
        self.assertEqual(group["effective_probe_count"], 4)
        self.assertEqual(len({probe["query"] for probe in group["probes"]}), 4)


if __name__ == "__main__":
    unittest.main()
