import ast
import unittest
import importlib.util
import sys
import types
from collections import Counter
from contextlib import contextmanager
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

CHECKPOINTING_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "verl"
    / "trainer"
    / "ppo"
    / "eitr_checkpointing.py"
)
CHECKPOINTING_SPEC = importlib.util.spec_from_file_location(
    "eitr_checkpointing_module", CHECKPOINTING_MODULE_PATH
)
CHECKPOINTING = importlib.util.module_from_spec(CHECKPOINTING_SPEC)
CHECKPOINTING_SPEC.loader.exec_module(CHECKPOINTING)

build_sibling_probe_tensors = EITR.build_sibling_probe_tensors
build_online_probe_tensors = EITR.build_online_probe_tensors
build_grpo_uids = EITR.build_grpo_uids
build_eitr_correction_optimizer = EITR.build_eitr_correction_optimizer
bounded_quantile_sample_stride = EITR.bounded_quantile_sample_stride
coverage_weighted_state_scale = EITR.coverage_weighted_state_scale
compacted_state_forward_plan = EITR.compacted_state_forward_plan
induced_js_from_cached_effects = EITR.induced_js_from_cached_effects
directional_parameter_candidates = EITR.directional_parameter_candidates
directional_descent_diagnostics = EITR.directional_descent_diagnostics
cached_probe_fingerprint = EITR.cached_probe_fingerprint
eitr_update_direction_diagnostic_enabled = EITR.eitr_update_direction_diagnostic_enabled
eitr_score_path_noop_direction_audit_enabled = EITR.eitr_score_path_noop_direction_audit_enabled
score_path_audit_event_sequence = EITR.score_path_audit_event_sequence
same_batch_cache_signature = EITR.same_batch_cache_signature
same_batch_scale_diagnostic_lrs = EITR.same_batch_scale_diagnostic_lrs
same_batch_sgd_candidates = EITR.same_batch_sgd_candidates
probe_effect_diversity = EITR.probe_effect_diversity
parameter_correction_diagnostics = EITR.parameter_correction_diagnostics
proposal_signal_diagnostics = EITR.proposal_signal_diagnostics
normalized_correction_step = EITR.normalized_correction_step
rollout_averaged_env_drift = EITR.rollout_averaged_env_drift
rank_owned_global_additive_stats = EITR.rank_owned_global_additive_stats
resolved_query_direction_candidates = EITR.resolved_query_direction_candidates
should_run_post_diagnostic = EITR.should_run_post_diagnostic
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

    def test_both_ppo_entrypoints_finish_wandb(self):
        trainer_dir = Path(__file__).resolve().parents[1] / "verl" / "trainer"
        for entrypoint in ("main_ppo.py", "main_ppo_format.py"):
            source = (trainer_dir / entrypoint).read_text()
            self.assertIn("finally:", source, entrypoint)
            self.assertIn("trainer.logger.finish()", source, entrypoint)

    def test_smoke_logs_to_console_and_wandb(self):
        runner = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "train"
            / "train_eitr_nq_gate_c_smoke.sh"
        ).read_text()
        self.assertIn("trainer.logger=\"['console','wandb']\"", runner)
        self.assertIn('RAY_TMPDIR="${RAY_TMPDIR:-$STORAGE_ROOT/r}"', runner)
        self.assertNotIn('$STORAGE_ROOT/ray_tmp/$EXPERIMENT_NAME', runner)
        self.assertIn(
            "EITR score-path audit requires METRICS_LEVEL=debug",
            runner,
        )
        self.assertIn(
            'EITR_COMPACT_INVALID_STATES="${EITR_COMPACT_INVALID_STATES:-true}"',
            runner,
        )
        self.assertIn(
            'actor_rollout_ref.actor.eitr.compact_invalid_states="$EITR_COMPACT_INVALID_STATES"',
            runner,
        )
        self.assertIn('EITR_MAX_UPDATE_NORM="${EITR_MAX_UPDATE_NORM:-3e-6}"', runner)
        self.assertIn('EITR_MIN_UPDATE_NORM="${EITR_MIN_UPDATE_NORM:-3e-6}"', runner)
        self.assertIn(
            'actor_rollout_ref.actor.eitr.correction_max_update_norm="$EITR_MAX_UPDATE_NORM"',
            runner,
        )
        self.assertIn(
            'actor_rollout_ref.actor.eitr.correction_min_update_norm="$EITR_MIN_UPDATE_NORM"',
            runner,
        )

    def test_smoke_uses_shared_mandatory_search_reward(self):
        runner = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "train"
            / "train_eitr_nq_gate_c_smoke.sh"
        ).read_text()
        self.assertIn("python3 -m verl.trainer.main_ppo_format", runner)
        self.assertIn('REWARD_PROFILE="${REWARD_PROFILE:-mandatory_search}"', runner)
        self.assertIn('DEFAULT_STRUCTURE_FORMAT_SCORE=0.2', runner)
        self.assertIn('DEFAULT_STRUCTURE_FORMAT_SCORE=0', runner)
        self.assertIn('DEFAULT_RETRIEVAL_SCORE=0.1', runner)
        self.assertIn('RETRIEVAL_SCORE="${RETRIEVAL_SCORE:-$DEFAULT_RETRIEVAL_SCORE}"', runner)
        self.assertIn('THINK_FORMAT_SCORE="${THINK_FORMAT_SCORE:-0.05}"', runner)
        self.assertIn('ANSWER_FORMAT_SCORE="${ANSWER_FORMAT_SCORE:-0.05}"', runner)
        self.assertIn('EVIDENCE_SCORE="${EVIDENCE_SCORE:-0.2}"', runner)
        self.assertIn('ANSWER_EM_SCORE="${ANSWER_EM_SCORE:-0.7}"', runner)
        self.assertIn('JOINT_SUCCESS_BONUS="${JOINT_SUCCESS_BONUS:-0.5}"', runner)
        self.assertIn("MIN_HARD_REWARD_TRAJECTORY_LENGTH", runner)
        self.assertIn('reward_model.structure_format_score="$STRUCTURE_FORMAT_SCORE"', runner)
        self.assertIn('reward_model.final_format_score="$FINAL_FORMAT_SCORE"', runner)
        self.assertIn('reward_model.retrieval_score="$RETRIEVAL_SCORE"', runner)
        self.assertIn('reward_model.reward_profile="$REWARD_PROFILE"', runner)
        self.assertIn('reward_model.think_format_score="$THINK_FORMAT_SCORE"', runner)
        self.assertIn('reward_model.answer_format_score="$ANSWER_FORMAT_SCORE"', runner)
        self.assertIn('reward_model.evidence_score="$EVIDENCE_SCORE"', runner)
        self.assertIn('reward_model.answer_em_score="$ANSWER_EM_SCORE"', runner)
        self.assertIn('reward_model.joint_success_bonus="$JOINT_SUCCESS_BONUS"', runner)
        self.assertIn('ACTOR_LR="${ACTOR_LR:-5e-7}"', runner)
        self.assertIn(
            'ACTOR_FSDP_PARAM_OFFLOAD="${ACTOR_FSDP_PARAM_OFFLOAD:-false}"',
            runner,
        )
        self.assertIn(
            'ACTOR_FSDP_OPTIMIZER_OFFLOAD="${ACTOR_FSDP_OPTIMIZER_OFFLOAD:-false}"',
            runner,
        )
        self.assertIn(
            'ACTOR_BATCH_OFFLOAD="${ACTOR_BATCH_OFFLOAD:-false}"',
            runner,
        )
        self.assertIn(
            'REF_FSDP_PARAM_OFFLOAD="${REF_FSDP_PARAM_OFFLOAD:-false}"',
            runner,
        )
        self.assertIn(
            'actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_FSDP_PARAM_OFFLOAD"',
            runner,
        )
        self.assertIn(
            'actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_FSDP_OPTIMIZER_OFFLOAD"',
            runner,
        )
        self.assertIn(
            'actor_rollout_ref.actor.fsdp_config.batch_offload="$ACTOR_BATCH_OFFLOAD"',
            runner,
        )
        self.assertIn(
            'actor_rollout_ref.ref.fsdp_config.param_offload="$REF_FSDP_PARAM_OFFLOAD"',
            runner,
        )

    def test_official_v03_runners_select_exact_reward_profile(self):
        script_root = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "nq_hotpotqa"
            / "v0.3"
        )
        for script_name in ("train_grpo_format.sh", "train_ppo_format.sh"):
            runner = (script_root / script_name).read_text()
            self.assertIn(
                "reward_model.reward_profile=official_v03",
                runner,
                script_name,
            )

    def test_formal_run_uses_mixed_train_and_nq_validation(self):
        runner = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "train"
            / "train_eitr_nq_gate_c_full.sh"
        ).read_text()
        self.assertIn("data/nq_hotpotqa_train", runner)
        self.assertIn("data/nq_search", runner)
        self.assertIn('SHUFFLE_TRAIN_DATALOADER="${SHUFFLE_TRAIN_DATALOADER:-true}"', runner)
        self.assertIn('TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-null}"', runner)
        self.assertIn('EITR_LR="${EITR_LR:-3e-5}"', runner)
        self.assertIn(
            'EITR_MAX_UPDATE_NORM="${EITR_MAX_UPDATE_NORM:-3e-6}"',
            runner,
        )
        self.assertIn(
            'EITR_MIN_UPDATE_NORM="${EITR_MIN_UPDATE_NORM:-3e-6}"',
            runner,
        )
        self.assertIn('VAL_DATA_NUM="${VAL_DATA_NUM:-256}"', runner)
        self.assertIn('TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-8000}"', runner)
        self.assertIn('LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.03575}"', runner)
        self.assertIn('PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-4}"', runner)
        self.assertIn('REWARD_PROFILE="${REWARD_PROFILE:-mandatory_search}"', runner)
        self.assertIn('JOINT_SUCCESS_BONUS="${JOINT_SUCCESS_BONUS:-0.5}"', runner)
        self.assertIn('REWARD_PROFILE_SLUG="${REWARD_PROFILE//_/-}"', runner)
        self.assertIn('${EITR_MODE}-nq-hotpotqa-${REWARD_PROFILE_SLUG}-phase2-full', runner)
        self.assertIn(
            'ACTOR_FSDP_PARAM_OFFLOAD="${ACTOR_FSDP_PARAM_OFFLOAD:-false}"',
            runner,
        )
        self.assertIn(
            'ACTOR_FSDP_OPTIMIZER_OFFLOAD="${ACTOR_FSDP_OPTIMIZER_OFFLOAD:-true}"',
            runner,
        )
        self.assertIn(
            'ACTOR_BATCH_OFFLOAD="${ACTOR_BATCH_OFFLOAD:-true}"',
            runner,
        )
        self.assertIn(
            'REF_FSDP_PARAM_OFFLOAD="${REF_FSDP_PARAM_OFFLOAD:-true}"',
            runner,
        )
        self.assertIn(
            'EITR_PROBE_LOGPROB_MICRO_BATCH_SIZE="${EITR_PROBE_LOGPROB_MICRO_BATCH_SIZE:-4}"',
            runner,
        )
        self.assertIn(
            'EITR_PROBE_MICRO_BATCH_SIZE="${EITR_PROBE_MICRO_BATCH_SIZE:-4}"',
            runner,
        )

    def test_three_gpu_profile_uses_divisible_batches_and_conservative_offload(self):
        root = Path(__file__).resolve().parents[1]
        profile = (
            root / "scripts" / "train" / "train_eitr_nq_hotpotqa_3gpu.sh"
        ).read_text()
        smoke = (
            root / "scripts" / "train" / "train_eitr_nq_gate_c_smoke.sh"
        ).read_text()
        self.assertIn('CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"', profile)
        self.assertIn('NUM_GPUS="${NUM_GPUS:-3}"', profile)
        self.assertIn('TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-30}"', profile)
        self.assertIn('PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-30}"', profile)
        self.assertIn('PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-6}"', profile)
        self.assertIn(
            'EITR_PROBE_GRADIENT_CHECKPOINTING="${EITR_PROBE_GRADIENT_CHECKPOINTING:-true}"',
            profile,
        )
        self.assertIn(
            'EITR_COMPACT_INVALID_STATES="${EITR_COMPACT_INVALID_STATES:-true}"',
            profile,
        )
        self.assertIn(
            'ACTOR_FSDP_OPTIMIZER_OFFLOAD="${ACTOR_FSDP_OPTIMIZER_OFFLOAD:-true}"',
            profile,
        )
        self.assertIn('Physical GPU0 is reserved and must not be used', smoke)
        self.assertIn('divisible by NUM_GPUS=', smoke)
        self.assertIn(
            'actor_rollout_ref.rollout.log_prob_micro_batch_size="$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE"',
            smoke,
        )
        self.assertIn(
            'actor_rollout_ref.ref.log_prob_micro_batch_size="$REF_LOG_PROB_MICRO_BATCH_SIZE"',
            smoke,
        )

    def test_combined_worker_offloads_reference_without_offloading_actor(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "verl"
            / "workers"
            / "fsdp_workers.py"
        ).read_text()

        init_flags = source[
            source.index("self._is_ref_offload_param = bool("):
            source.index("# normalize config")
        ]
        self.assertIn("self.config.ref.fsdp_config.get('param_offload', False)", init_flags)
        self.assertIn("self.config.actor.fsdp_config.get('param_offload', False)", init_flags)

        ref_build = source[
            source.index("if self._is_ref:\n", source.index("def init_model")):
            source.index("if self._is_actor:\n", source.index("if self._is_ref:\n", source.index("def init_model")))
        ]
        self.assertIn("if self._is_ref_offload_param:", ref_build)
        self.assertIn("offload_fsdp_param_and_grad(", ref_build)

        ref_compute = source[
            source.index("def compute_ref_log_prob"):
            source.index("def save_checkpoint")
        ]
        self.assertIn("if self._is_ref_offload_param:", ref_compute)
        self.assertIn("load_fsdp_param_and_grad(", ref_compute)
        self.assertIn("offload_fsdp_param_and_grad(", ref_compute)
        self.assertNotIn("if self._is_offload_param:", ref_compute)

    def test_formal_actor_streams_batches_and_offloads_adamw_before_eitr(self):
        root = Path(__file__).resolve().parents[1]
        actor_source = (root / "verl" / "workers" / "actor" / "dp_actor.py").read_text()
        worker_source = (root / "verl" / "workers" / "fsdp_workers.py").read_text()

        update_policy = actor_source[
            actor_source.index("    def update_policy("):
            actor_source.index("        return metrics", actor_source.index("    def update_policy("))
        ]
        last_grpo_step = update_policy.index("grad_norm = self._optimizer_step()")
        optimizer_offload = update_policy.index(
            "offload_fsdp_optimizer(self.actor_optimizer)"
        )
        correction_start = update_policy.index(
            "# Conditional correction is deliberately not another GRPO epoch"
        )
        self.assertLess(last_grpo_step, optimizer_offload)
        self.assertLess(optimizer_offload, correction_start)
        self.assertIn(
            "micro_batch.select(*grpo_select_keys).cuda()",
            update_policy,
        )
        self.assertIn(
            "state_chunk = self._eitr_chunk_to_cuda(state_chunk)",
            update_policy,
        )

        chunk_helper = actor_source[
            actor_source.index("    def _eitr_chunk_to_cuda"):
            actor_source.index("    def _snapshot_eitr_local_state")
        ]
        self.assertIn("state_chunk.select(*EITR_BATCH_KEYS).cuda()", chunk_helper)

        worker_update = worker_source[
            worker_source.index("    def update_actor("):
            worker_source.index("    def compute_log_prob(")
        ]
        self.assertIn(
            "data.to('cpu' if self._is_offload_batch else 'cuda')",
            worker_update,
        )
        self.assertIn("if not self._is_offload_batch:", worker_update)
        self.assertIn("load_fsdp_optimizer(", worker_update)
        self.assertIn("timing_s/actor_optimizer_load_before_grpo", worker_update)

    def test_audit_failure_is_logged_before_raise_and_skips_side_effects(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "verl"
            / "trainer"
            / "ppo"
            / "ray_trainer.py"
        ).read_text()
        fit_start = source.index("    def fit(self):")
        pending_failure_start = source.index(
            "pending_audit_failure = {", fit_start
        )
        validation_call = source.index("val_metrics: dict = self._validate()", fit_start)
        checkpoint_call = source.index("self._save_checkpoint()", fit_start)
        logger_call = source.index("                logger.log(", fit_start)
        failure_guard = source.index(
            "                if pending_audit_failure is not None:", logger_call
        )
        finish_call = source.index("                    logger.finish()", failure_guard)
        failure_raise = source.index("                    raise RuntimeError(", finish_call)

        validation_guard = source[source.rfind("if ", fit_start, validation_call):validation_call]
        checkpoint_guard = source[source.rfind("if ", fit_start, checkpoint_call):checkpoint_call]
        self.assertIn("pending_audit_failure is None", validation_guard)
        self.assertIn("pending_audit_failure is None", checkpoint_guard)
        self.assertLess(pending_failure_start, validation_call)
        self.assertLess(pending_failure_start, checkpoint_call)
        self.assertLess(validation_call, logger_call)
        self.assertLess(checkpoint_call, logger_call)
        self.assertLess(logger_call, failure_guard)
        self.assertLess(failure_guard, finish_call)
        self.assertLess(finish_call, failure_raise)

    def test_audit_inconclusive_is_logged_before_report_and_skips_side_effects(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "verl"
            / "trainer"
            / "ppo"
            / "ray_trainer.py"
        ).read_text()
        fit_start = source.index("    def fit(self):")
        pending_start = source.index(
            "pending_audit_inconclusive = {", fit_start
        )
        validation_call = source.index("val_metrics: dict = self._validate()", fit_start)
        checkpoint_call = source.index("self._save_checkpoint()", fit_start)
        logger_call = source.index("                logger.log(", fit_start)
        inconclusive_guard = source.index(
            "                if pending_audit_inconclusive is not None:",
            logger_call,
        )
        finish_call = source.index(
            "                    logger.finish()", inconclusive_guard
        )
        report = source.index("INCONCLUSIVE_ZERO_PROPOSAL", finish_call)
        inconclusive_raise = source.rfind(
            "                    raise RuntimeError(", finish_call, report
        )

        validation_guard = source[
            source.rfind("if ", fit_start, validation_call):validation_call
        ]
        checkpoint_guard = source[
            source.rfind("if ", fit_start, checkpoint_call):checkpoint_call
        ]
        self.assertIn("pending_audit_inconclusive is None", validation_guard)
        self.assertIn("pending_audit_inconclusive is None", checkpoint_guard)
        self.assertLess(pending_start, validation_call)
        self.assertLess(pending_start, checkpoint_call)
        self.assertLess(validation_call, logger_call)
        self.assertLess(checkpoint_call, logger_call)
        self.assertLess(logger_call, inconclusive_guard)
        self.assertLess(inconclusive_guard, finish_call)
        self.assertLess(finish_call, inconclusive_raise)
        self.assertLess(inconclusive_raise, report)


class ObservationTruncationTest(unittest.TestCase):
    class CharacterTokenizer:
        pad_token_id = 0
        padding_side = "right"

        def __call__(self, text, add_special_tokens=False):
            del add_special_tokens
            return {"input_ids": [ord(character) for character in text]}

        def decode(self, token_ids):
            return "".join(chr(token_id) for token_id in token_ids if token_id)

    def test_rollout_substage_timings_accumulate(self):
        manager = LLMGenerationManager.__new__(LLMGenerationManager)
        manager.timing_raw = {}

        manager._record_timing("eitr_probe_generation", 1.25)
        manager._record_timing("eitr_probe_generation", 0.75)

        self.assertEqual(manager.timing_raw["eitr_probe_generation"], 2.0)

    def test_long_information_preserves_both_tags(self):
        manager = LLMGenerationManager.__new__(LLMGenerationManager)
        manager.tokenizer = self.CharacterTokenizer()
        manager.config = SimpleNamespace(max_obs_length=64)
        observation = "\n<information>" + ("document text " * 20) + "</information>\n"

        token_ids = manager._process_next_obs([observation])[0]
        decoded = manager.tokenizer.decode(token_ids.tolist())

        self.assertLessEqual(len(token_ids), 64)
        self.assertIn("<information>", decoded)
        self.assertTrue(decoded.endswith("</information>\n"))

    def test_non_information_observation_keeps_legacy_prefix_truncation(self):
        manager = LLMGenerationManager.__new__(LLMGenerationManager)
        manager.tokenizer = self.CharacterTokenizer()
        manager.config = SimpleNamespace(max_obs_length=8)

        token_ids = manager._process_next_obs(["abcdefghijk"])[0]

        self.assertEqual(manager.tokenizer.decode(token_ids.tolist()), "abcdefgh")


class EITRMathTest(unittest.TestCase):
    def test_compacted_state_plan_pads_only_to_busiest_rank(self):
        plan = compacted_state_forward_plan(
            local_active_count=30,
            local_physical_count=50,
            max_active_count=34,
            states_per_chunk=1,
        )
        self.assertEqual(plan["target_state_count"], 34.0)
        self.assertEqual(plan["dummy_state_count"], 4.0)
        self.assertEqual(plan["chunk_count"], 34.0)
        self.assertAlmostEqual(plan["forward_reduction_rate"], 0.32)

    def test_compacted_state_plan_rounds_to_complete_probe_chunks(self):
        plan = compacted_state_forward_plan(
            local_active_count=30,
            local_physical_count=50,
            max_active_count=33,
            states_per_chunk=2,
        )
        self.assertEqual(plan["target_state_count"], 34.0)
        self.assertEqual(plan["dummy_state_count"], 4.0)
        self.assertEqual(plan["chunk_count"], 17.0)

    def test_compacted_state_plan_handles_zero_active_batch(self):
        plan = compacted_state_forward_plan(
            local_active_count=0,
            local_physical_count=50,
            max_active_count=0,
            states_per_chunk=1,
        )
        self.assertEqual(plan["target_state_count"], 0.0)
        self.assertEqual(plan["chunk_count"], 0.0)

    def test_compacted_state_plan_rejects_inconsistent_counts(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            compacted_state_forward_plan(
                local_active_count=31,
                local_physical_count=30,
                max_active_count=31,
                states_per_chunk=1,
            )

    def test_coverage_weighted_scale_tracks_rollout_coverage(self):
        self.assertEqual(coverage_weighted_state_scale(0, 160), 0.0)
        self.assertEqual(coverage_weighted_state_scale(40, 160), 0.25)
        self.assertEqual(coverage_weighted_state_scale(160, 160), 1.0)
        # Two FSDP ranks each contribute a local active-state mean. Gradient
        # averaging recovers the same global 25% coverage scale.
        self.assertEqual(
            coverage_weighted_state_scale(20, 160, world_size=2),
            0.25,
        )
        self.assertEqual(coverage_weighted_state_scale(0, 0), 0.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            coverage_weighted_state_scale(-1, 160)

    def test_rollout_averaged_drift_uses_full_batch_denominator(self):
        self.assertAlmostEqual(rollout_averaged_env_drift(0.8, 160), 0.005)
        self.assertEqual(rollout_averaged_env_drift(0.0, 0), 0.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            rollout_averaged_env_drift(0.1, -1)

    def test_post_diagnostic_schedule(self):
        self.assertTrue(
            should_run_post_diagnostic(10, 10, correction_applied=True)
        )
        self.assertFalse(
            should_run_post_diagnostic(9, 10, correction_applied=True)
        )
        self.assertFalse(
            should_run_post_diagnostic(10, 10, correction_applied=False)
        )
        self.assertFalse(
            should_run_post_diagnostic(10, 0, correction_applied=True)
        )

    def test_coverage_weighting_scales_eitr_gradient(self):
        old = torch.zeros(1, 4)
        docs = torch.eye(4).unsqueeze(0)
        mask = torch.ones(1, 4, dtype=torch.bool)

        gradients = []
        for active_states in (40, 160):
            current = torch.tensor([[2.0, -1.0, -2.0, -3.0]], requires_grad=True)
            result = induced_js_from_cached_effects(
                current_seq_logp=current,
                old_seq_logp=old,
                doc_probs=docs,
                probe_mask=mask,
            )
            scale = coverage_weighted_state_scale(active_states, 160)
            (result["js"].mean() * scale).backward()
            gradients.append(current.grad.detach().clone())

        self.assertTrue(torch.allclose(gradients[0], gradients[1] * 0.25))

    def test_mixed_dataset_grpo_uids_do_not_collide(self):
        uids = build_grpo_uids(
            ["nq", "nq", "hotpotqa", "hotpotqa"],
            [7, 7, 7, 7],
        )
        self.assertEqual(uids.tolist(), ["nq::7", "nq::7", "hotpotqa::7", "hotpotqa::7"])
        self.assertNotEqual(uids[0], uids[2])
        with self.assertRaisesRegex(ValueError, "metadata length mismatch"):
            build_grpo_uids(["nq"], [1, 2])

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
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_eitr_optimization_schedule(
                {"mode": "eitr"},
                ppo_epochs=1,
                correction_passes=2,
            )

    def test_same_batch_scale_diagnostic_is_opt_in_and_non_accumulating(self):
        self.assertEqual(same_batch_scale_diagnostic_lrs({}), ())
        self.assertEqual(
            same_batch_scale_diagnostic_lrs({"same_batch_scale_diagnostic": True}),
            (1e-5, 3e-5, 1e-4),
        )
        parameter = torch.tensor([3.0, -2.0])
        gradient = torch.tensor([2.0, -4.0])
        candidates = same_batch_sgd_candidates([parameter], [gradient])
        self.assertTrue(torch.equal(parameter, torch.tensor([3.0, -2.0])))
        self.assertTrue(torch.equal(candidates[1e-5][0], parameter - 1e-5 * gradient))
        self.assertTrue(torch.equal(candidates[3e-5][0], parameter - 3e-5 * gradient))
        self.assertTrue(torch.equal(candidates[1e-4][0], parameter - 1e-4 * gradient))

        cached = {key: torch.ones(2, 2) for key in EITR.EITR_BATCH_KEYS}
        before = same_batch_cache_signature(cached)
        same_batch_sgd_candidates([parameter], [gradient])
        self.assertEqual(before, same_batch_cache_signature(cached))

    def test_direction_diagnostic_is_opt_in_and_has_opposite_non_accumulating_updates(self):
        self.assertFalse(eitr_update_direction_diagnostic_enabled({}))
        self.assertTrue(
            eitr_update_direction_diagnostic_enabled({"update_direction_diagnostic": "true"})
        )
        parameter = torch.tensor([2.0, -1.0])
        gradient = torch.tensor([4.0, 2.0])
        candidates = directional_parameter_candidates([parameter], [gradient], 0.25)
        self.assertTrue(torch.equal(parameter, torch.tensor([2.0, -1.0])))
        self.assertTrue(torch.equal(candidates["minus"][0], torch.tensor([1.0, -1.5])))
        self.assertTrue(torch.equal(candidates["plus"][0], torch.tensor([3.0, -0.5])))

    def test_eitr_sgd_does_not_advance_grpo_adamw_state(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        grpo_optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        parameter.grad = torch.tensor([0.5])
        grpo_optimizer.step()
        grpo_optimizer.zero_grad()
        adam_state_before = {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in grpo_optimizer.state[parameter].items()
        }

        eitr_optimizer = build_eitr_correction_optimizer(
            [parameter],
            {"mode": "eitr", "correction_lr": 1e-2},
            default_lr=1e-3,
        )
        self.assertIsInstance(eitr_optimizer, torch.optim.SGD)
        self.assertEqual(eitr_optimizer.param_groups[0]["momentum"], 0.0)
        self.assertEqual(eitr_optimizer.param_groups[0]["weight_decay"], 0.0)
        parameter.grad = torch.tensor([0.25])
        eitr_optimizer.step()

        self.assertEqual(eitr_optimizer.state, {})
        for key, expected in adam_state_before.items():
            actual = grpo_optimizer.state[parameter][key]
            if torch.is_tensor(expected):
                self.assertTrue(torch.equal(actual, expected))
            else:
                self.assertEqual(actual, expected)

        self.assertIsNone(
            build_eitr_correction_optimizer(
                [parameter],
                {"mode": "probe_only"},
                default_lr=1e-3,
            )
        )

    def test_normalized_correction_preserves_small_fixed_lr_step(self):
        step = normalized_correction_step(
            grad_norm=0.05,
            correction_lr=3e-5,
            grad_clip=1.0,
            max_update_norm=3e-6,
        )
        self.assertEqual(step["normalization_scale"], 1.0)
        self.assertEqual(step["effective_lr"], 3e-5)
        self.assertAlmostEqual(step["predicted_update_norm"], 1.5e-6)

    def test_normalized_correction_caps_one_sgd_update_without_flipping_direction(self):
        step = normalized_correction_step(
            grad_norm=0.2,
            correction_lr=3e-5,
            grad_clip=1.0,
            max_update_norm=3e-6,
        )
        self.assertAlmostEqual(step["normalization_scale"], 0.5)
        self.assertAlmostEqual(step["effective_lr"], 1.5e-5)
        self.assertAlmostEqual(step["unnormalized_update_norm"], 6e-6)
        self.assertAlmostEqual(step["predicted_update_norm"], 3e-6)
        gradient = torch.tensor([3.0, -4.0])
        delta = -step["effective_lr"] * gradient
        self.assertLess(float(torch.dot(gradient, delta)), 0.0)

    def test_normalized_correction_respects_actor_clip_and_can_be_disabled(self):
        clipped = normalized_correction_step(
            grad_norm=4.0,
            correction_lr=3e-5,
            grad_clip=1.0,
            max_update_norm=3e-6,
        )
        self.assertEqual(clipped["clipped_grad_norm"], 1.0)
        self.assertAlmostEqual(clipped["normalization_scale"], 0.1)
        self.assertAlmostEqual(clipped["predicted_update_norm"], 3e-6)

        disabled = normalized_correction_step(
            grad_norm=0.2,
            correction_lr=3e-5,
            grad_clip=1.0,
            max_update_norm=None,
        )
        self.assertEqual(disabled["normalization_scale"], 1.0)
        self.assertAlmostEqual(disabled["effective_lr"], 3e-5)

    def test_weak_predicted_update_skips_the_sgd_commit(self):
        skipped = normalized_correction_step(
            grad_norm=0.08,
            correction_lr=3e-5,
            grad_clip=1.0,
            max_update_norm=3e-6,
            min_update_norm=3e-6,
        )
        self.assertEqual(skipped["should_apply"], 0.0)
        self.assertEqual(skipped["normalization_scale"], 0.0)
        self.assertEqual(skipped["effective_lr"], 0.0)
        self.assertEqual(skipped["predicted_update_norm"], 0.0)
        self.assertAlmostEqual(skipped["unnormalized_update_norm"], 2.4e-6)

        boundary = normalized_correction_step(
            grad_norm=0.1,
            correction_lr=3e-5,
            grad_clip=1.0,
            max_update_norm=3e-6,
            min_update_norm=3e-6,
        )
        self.assertEqual(boundary["should_apply"], 1.0)
        self.assertAlmostEqual(boundary["predicted_update_norm"], 3e-6)

    def test_normalized_correction_rejects_invalid_ceiling(self):
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "correction_max_update_norm"):
                normalized_correction_step(
                    grad_norm=0.2,
                    correction_lr=3e-5,
                    grad_clip=1.0,
                    max_update_norm=invalid,
                )

        with self.assertRaisesRegex(ValueError, "correction_max_update_norm"):
            validate_eitr_config(
                {"correction_max_update_norm": 0},
                n_agent=5,
                max_queries_per_turn=1,
                rollout_n=1,
            )
        with self.assertRaisesRegex(ValueError, "correction_min_update_norm"):
            validate_eitr_config(
                {
                    "correction_min_update_norm": 4e-6,
                    "correction_max_update_norm": 3e-6,
                },
                n_agent=5,
                max_queries_per_turn=1,
                rollout_n=1,
            )
        for invalid_lr in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "correction_lr"):
                validate_eitr_config(
                    {"correction_lr": invalid_lr},
                    n_agent=5,
                    max_queries_per_turn=1,
                    rollout_n=1,
                )

    def test_actor_uses_single_global_normalized_step(self):
        actor_source = (
            Path(__file__).resolve().parents[1]
            / "verl"
            / "workers"
            / "actor"
            / "dp_actor.py"
        ).read_text()
        method_start = actor_source.index(
            "    def _normalized_eitr_optimizer_step(self):"
        )
        method_end = actor_source.index("    @contextmanager", method_start)
        method = actor_source[method_start:method_end]
        self.assertIn("self.actor_module.clip_grad_norm_", method)
        self.assertIn("normalized_correction_step(", method)
        self.assertEqual(method.count("self.eitr_optimizer.step()"), 1)
        self.assertIn("if step['should_apply'] > 0.5:", method)
        self.assertNotIn("backward()", method)
        self.assertIn("group['lr'] = base_lr", method)

    def test_weak_update_noop_drift_is_explicit_and_not_counted_as_correction(self):
        actor_source = (
            Path(__file__).resolve().parents[1]
            / "verl"
            / "workers"
            / "actor"
            / "dp_actor.py"
        ).read_text()
        self.assertIn("post_drift_stat_tensor = pre_drift_stat_tensor.clone()", actor_source)
        self.assertIn("eitr_post_diagnostic_noop = True", actor_source)
        self.assertIn(
            "if correction_applied:\n                                    eitr_correction_optimizer_step_count += 1",
            actor_source,
        )
        self.assertIn(
            "'actor/eitr_correction_skipped_weak_update': float(", actor_source
        )

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
        with self.assertRaisesRegex(ValueError, "correction_optimizer"):
            validate_eitr_config(
                {"correction_optimizer": "adamw"},
                n_agent=5,
                max_queries_per_turn=1,
                rollout_n=1,
            )
        with self.assertRaisesRegex(ValueError, "correction_lr"):
            validate_eitr_config(
                {"correction_lr": 0},
                n_agent=5,
                max_queries_per_turn=1,
                rollout_n=1,
            )
        with self.assertRaisesRegex(ValueError, "post_diagnostic_freq"):
            validate_eitr_config(
                {"post_diagnostic_freq": -1},
                n_agent=5,
                max_queries_per_turn=1,
                rollout_n=1,
            )
        with self.assertRaisesRegex(ValueError, "must equal"):
            validate_eitr_config(
                {
                    "probe_micro_batch_size": 4,
                    "probe_logprob_micro_batch_size": 8,
                },
                n_agent=5,
                max_queries_per_turn=1,
                rollout_n=1,
            )
        with self.assertRaisesRegex(ValueError, "divisible by probe_count"):
            validate_eitr_config(
                {
                    "probe_count": 4,
                    "probe_micro_batch_size": 6,
                    "probe_logprob_micro_batch_size": 6,
                },
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
        self.assertEqual(tensors["eitr_probe_old_seq_logp"].dtype, torch.float64)
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
    def test_executed_search_count_is_materialized_as_batch_aligned_tensor(self):
        captured = {}

        class FakeDataProto:
            @staticmethod
            def from_dict(batch):
                captured.update(batch)
                return SimpleNamespace(batch=batch, meta_info={})

        manager = object.__new__(LLMGenerationManager)
        manager.tokenizer = SimpleNamespace(pad_token_id=0)
        manager.tensor_fn = SimpleNamespace(
            create_attention_mask=lambda values: values.ne(0).long(),
            create_position_ids=lambda mask: (
                torch.cumsum(mask, dim=-1) - 1
            ).clamp_min(0),
        )
        left_side = {"input_ids": torch.tensor([[11, 12], [21, 22]])}
        right_side = {
            "responses": torch.tensor([[31, 32], [41, 0]]),
            "responses_with_info_mask": torch.tensor([[31, 0], [41, 0]]),
        }
        original_data_proto = generation_module.DataProto
        generation_module.DataProto = FakeDataProto
        try:
            output = manager._compose_final_output(
                left_side,
                right_side,
                {"valid_search_stats": [0, 2]},
            )
        finally:
            generation_module.DataProto = original_data_proto

        self.assertEqual(
            output.batch["executed_search_count"].tolist(),
            [0, 2],
        )
        self.assertEqual(captured["executed_search_count"].dtype, torch.long)

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

    def test_same_state_probe_candidates_use_one_multi_sample_call(self):
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
        seen_sampling_params = []

        def fake_generate(prompts):
            seen_sampling_params.append(dict(prompts.meta_info["sampling_params"]))
            return SimpleNamespace(batch={
                "responses": torch.tensor([
                    [302, 999, 0],
                    [303, 999, 0],
                    [304, 999, 0],
                ], dtype=torch.long)
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

        self.assertEqual(seen_sampling_params, [{
            "max_tokens": 3,
            "n": 3,
            "seed": 123,
        }])
        self.assertEqual(manager._eitr_probe_collection_stats["probe_generation_call_count"], 1)
        self.assertEqual(manager._eitr_probe_collection_stats["probe_candidate_generated"], 3)
        self.assertEqual(group["effective_probe_count"], 4)
        self.assertEqual(len({probe["query"] for probe in group["probes"]}), 4)

    def test_multi_sample_gpu_padding_discards_all_padded_prompt_outputs(self):
        class FakeDataProto:
            @staticmethod
            def from_dict(batch):
                return SimpleNamespace(batch=batch, meta_info={})

        class FakeRolloutGroup:
            def generate_sequences(self, prompts):
                self.prompt_rows = int(prompts.batch["input_ids"].shape[0])
                return SimpleNamespace(
                    batch={"responses": torch.arange(9).reshape(9, 1)},
                    meta_info={},
                )

        manager = object.__new__(LLMGenerationManager)
        manager.config = SimpleNamespace(num_gpus=3, collect_eitr_probes=True)
        manager.actor_rollout_wg = FakeRolloutGroup()
        active_batch = SimpleNamespace(
            batch={"input_ids": torch.tensor([[1], [2]])},
            meta_info={"sampling_params": {"n": 3}},
        )
        original_data_proto = generation_module.DataProto
        generation_module.DataProto = FakeDataProto
        try:
            output = manager._generate_with_gpu_padding(active_batch)
        finally:
            generation_module.DataProto = original_data_proto

        self.assertEqual(manager.actor_rollout_wg.prompt_rows, 3)
        self.assertEqual(output.batch["responses"].flatten().tolist(), list(range(6)))

    def test_vllm_rollout_shapes_temporary_multi_sample_requests_from_request_n(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "verl"
            / "workers"
            / "rollout"
            / "vllm_rollout"
            / "vllm_rollout.py"
        ).read_text()
        self.assertIn("request_n = int(kwargs.get('n', self.config.n))", source)
        self.assertIn("idx = idx.repeat_interleave(request_n, dim=0)", source)
        self.assertNotIn("if self.config.n > 1 and do_sample:", source)


class ScorePathNoopDirectionAuditTest(unittest.TestCase):
    def _cached_batch(self):
        return {
            key: torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
            for key in EITR.EITR_BATCH_KEYS
        }

    def test_audit_defaults_to_disabled(self):
        self.assertFalse(eitr_score_path_noop_direction_audit_enabled({}))
        self.assertFalse(eitr_score_path_noop_direction_audit_enabled({
            "score_path_noop_direction_audit": "false"
        }))

    def test_large_flat_shard_quantile_sample_is_bounded(self):
        total_values = 1_600_000_000
        per_rank_budget = 131_072
        stride = bounded_quantile_sample_stride(total_values, per_rank_budget)
        sample_count = (total_values + stride - 1) // stride
        self.assertLessEqual(sample_count, per_rank_budget)
        self.assertEqual(bounded_quantile_sample_stride(0, per_rank_budget), 1)
        with self.assertRaisesRegex(ValueError, "sample_budget"):
            bounded_quantile_sample_stride(total_values, 0)

    def test_event_order_finishes_grpo_before_one_eitr_step(self):
        sequence = score_path_audit_event_sequence(5)
        self.assertEqual(sequence[0], "probe_old_logp@v0")
        self.assertEqual(sequence[1:6], tuple(f"GRPO_STEP_{i}" for i in range(1, 6)))
        self.assertEqual(sequence[6:], (
            "D_PRE_ALL_STATES@v5", "EITR_SGD_STEP_1", "D_POST_ALL_STATES@v6"
        ))

    def test_fingerprint_detects_cached_probe_mismatch(self):
        first = self._cached_batch()
        second = {key: value.clone() for key, value in first.items()}
        self.assertEqual(cached_probe_fingerprint([first]), cached_probe_fingerprint([second]))
        second["eitr_probe_old_seq_logp"][0, 0, 0] += 1
        self.assertNotEqual(cached_probe_fingerprint([first]), cached_probe_fingerprint([second]))

    def test_probe_hash_failures_are_synchronized_before_raising(self):
        actor_source = (
            Path(__file__).resolve().parents[1]
            / "verl"
            / "workers"
            / "actor"
            / "dp_actor.py"
        ).read_text()
        self.assertEqual(
            actor_source.count(
                "probe_hash_mismatch = self._distributed_scalar("
            ),
            2,
        )
        self.assertEqual(
            actor_source.count("if probe_hash_mismatch > 0:"),
            2,
        )

    def test_noop_does_not_change_parameters_and_candidates_do_not_accumulate(self):
        parameter = torch.tensor([2.0], dtype=torch.float32)
        gradient = torch.tensor([4.0], dtype=torch.float32)
        before = parameter.clone()
        noop = parameter.clone()
        candidates = directional_parameter_candidates([parameter], [gradient], 0.1)
        self.assertTrue(torch.equal(parameter, before))
        self.assertTrue(torch.equal(noop, before))
        self.assertAlmostEqual(candidates["minus"][0].item(), 1.6, places=6)
        self.assertAlmostEqual(candidates["plus"][0].item(), 2.4, places=6)

    def test_tiny_fp32_convex_target_decreases_along_negative_gradient(self):
        parameter = torch.tensor([2.0], dtype=torch.float32)
        gradient = 2 * parameter
        candidate = directional_parameter_candidates([parameter], [gradient], 0.1)["minus"][0]
        self.assertLess((candidate.square()).item(), (parameter.square()).item())

    def test_resolved_query_candidates_follow_the_js_descent_direction(self):
        old = torch.zeros(1, 4)
        current = torch.tensor(
            [[0.4, -0.2, 0.1, -0.3]], dtype=torch.float32, requires_grad=True
        )
        documents = torch.eye(4).unsqueeze(0)
        probe_mask = torch.ones(1, 4, dtype=torch.bool)
        zero_result = induced_js_from_cached_effects(
            current_seq_logp=current,
            old_seq_logp=old,
            doc_probs=documents,
            probe_mask=probe_mask,
        )
        gradient = torch.autograd.grad(zero_result["js"].mean(), current)[0]
        candidates = resolved_query_direction_candidates(current, gradient)

        values = {}
        for direction in ("minus", "zero", "plus"):
            values[direction] = float(
                induced_js_from_cached_effects(
                    current_seq_logp=candidates[direction],
                    old_seq_logp=old,
                    doc_probs=documents,
                    probe_mask=probe_mask,
                )["js"].mean().item()
            )
        diagnostic = directional_descent_diagnostics(**values, atol=1e-10)

        self.assertTrue(candidates["resolved"])
        self.assertGreaterEqual(candidates["resolved_grad_energy"], 0.99)
        self.assertLess(values["minus"], values["zero"])
        self.assertLess(values["zero"], values["plus"])
        self.assertEqual(diagnostic["pass"], 1.0)

    def test_direction_diagnostic_rejects_reversed_flat_and_nonfinite_values(self):
        resolved = directional_descent_diagnostics(minus=0.9, zero=1.0, plus=1.1)
        self.assertEqual(resolved["minus_pass"], 1.0)
        self.assertEqual(resolved["plus_pass"], 1.0)
        self.assertEqual(resolved["pass"], 1.0)
        self.assertEqual(
            directional_descent_diagnostics(minus=1.1, zero=1.0, plus=0.9)["pass"],
            0.0,
        )
        self.assertEqual(
            directional_descent_diagnostics(
                minus=1.0 - 1e-9,
                zero=1.0,
                plus=1.0 + 1e-9,
                jitter=1e-8,
            )["pass"],
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            directional_descent_diagnostics(
                minus=float("nan"), zero=1.0, plus=1.1
            )

    def test_parameter_gate_accepts_real_minus_descent_and_reports_nonsmooth_plus(self):
        diagnostic = parameter_correction_diagnostics(
            minus=9.326941658561e-4,
            zero=1.053403202048e-3,
            plus=8.439664564913e-4,
            minus_g_dot_delta=-2.856884e-7,
            minus_cosine=0.767159,
            jitter=0.0,
        )

        self.assertEqual(diagnostic["minus_pass"], 1.0)
        self.assertEqual(diagnostic["plus_pass"], 0.0)
        self.assertEqual(diagnostic["bidirectional_pass"], 0.0)
        self.assertEqual(diagnostic["update_direction_pass"], 1.0)
        self.assertEqual(diagnostic["correction_pass"], 1.0)
        self.assertEqual(diagnostic["nonsmooth_warning"], 1.0)

    def test_parameter_gate_rejects_wrong_update_sign_or_no_minus_descent(self):
        wrong_sign = parameter_correction_diagnostics(
            minus=0.9,
            zero=1.0,
            plus=1.1,
            minus_g_dot_delta=1e-3,
            minus_cosine=-0.5,
        )
        no_descent = parameter_correction_diagnostics(
            minus=1.1,
            zero=1.0,
            plus=0.9,
            minus_g_dot_delta=-1e-3,
            minus_cosine=0.5,
        )
        self.assertEqual(wrong_sign["correction_pass"], 0.0)
        self.assertEqual(no_descent["correction_pass"], 0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            parameter_correction_diagnostics(
                minus=0.9,
                zero=1.0,
                plus=1.1,
                minus_g_dot_delta=float("nan"),
                minus_cosine=0.5,
            )

    def test_actor_gate_uses_correction_pass_and_keeps_bidirectional_diagnostic(self):
        actor_source = (
            Path(__file__).resolve().parents[1]
            / "verl"
            / "workers"
            / "actor"
            / "dp_actor.py"
        ).read_text()
        gate_start = actor_source.index(
            "        parameter_direction = parameter_correction_diagnostics("
        )
        gate_end = actor_source.index(
            "        metrics['actor/eitr_score_path_final_restore_max_abs']",
            gate_start,
        )
        gate_block = actor_source[gate_start:gate_end]
        self.assertIn("and parameter_correction_pass", gate_block)
        self.assertNotIn("and parameter_bidirectional_pass", gate_block)
        self.assertIn(
            "'actor/eitr_score_path_parameter_bidirectional_pass'",
            gate_block,
        )
        self.assertIn(
            "'actor/eitr_score_path_parameter_nonsmooth_warning'",
            gate_block,
        )

    def test_two_rank_global_additive_stats_are_owned_only_once(self):
        global_stats = {
            "js_sum": 0.125,
            "state_count": 93.0,
            "probe_count": 358.0,
            "ess_sum": 360.5,
            "clipfrac_sum": 0.25,
            "active_micro_batch_count": 5.0,
        }
        per_rank = [
            rank_owned_global_additive_stats(
                global_stats, distributed=True, rank=rank
            )
            for rank in range(2)
        ]
        reduced = {
            key: sum(rank_stats[key] for rank_stats in per_rank)
            for key in global_stats
        }

        self.assertEqual(per_rank[0], global_stats)
        self.assertTrue(all(value == 0.0 for value in per_rank[1].values()))
        self.assertEqual(reduced, global_stats)
        self.assertEqual(
            rank_owned_global_additive_stats(
                global_stats, distributed=False, rank=7
            ),
            global_stats,
        )

    def test_temporary_probe_eval_restores_mode_on_success_and_exception(self):
        actor_source_path = (
            Path(__file__).resolve().parents[1]
            / "verl"
            / "workers"
            / "actor"
            / "dp_actor.py"
        )
        actor_source = actor_source_path.read_text()
        actor_tree = ast.parse(actor_source)
        actor_class = next(
            node
            for node in actor_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DataParallelPPOActor"
        )
        method = next(
            node
            for node in actor_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "_temporary_probe_eval"
        )
        namespace = {"contextmanager": contextmanager}
        method_module = ast.fix_missing_locations(
            ast.Module(body=[method], type_ignores=[])
        )
        exec(compile(method_module, str(actor_source_path), "exec"), namespace)
        temporary_probe_eval = namespace["_temporary_probe_eval"]

        class FakeModule:
            def __init__(self, training):
                self.training = training

            def eval(self):
                self.training = False
                return self

            def train(self, mode=True):
                self.training = bool(mode)
                return self

        for initial_mode in (True, False):
            with self.subTest(initial_mode=initial_mode):
                fake_actor = SimpleNamespace(actor_module=FakeModule(initial_mode))
                with temporary_probe_eval(fake_actor):
                    self.assertFalse(fake_actor.actor_module.training)
                self.assertEqual(fake_actor.actor_module.training, initial_mode)

        fake_actor = SimpleNamespace(actor_module=FakeModule(True))
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            with temporary_probe_eval(fake_actor):
                self.assertFalse(fake_actor.actor_module.training)
                raise RuntimeError("forward failed")
        self.assertTrue(fake_actor.actor_module.training)

        compute_log_prob_start = actor_source.index("    def compute_log_prob(")
        compute_eitr_start = actor_source.index("    def _compute_eitr_micro_batch(")
        iter_chunks_start = actor_source.index("    def _iter_eitr_state_chunks(")
        self.assertIn(
            "else self._temporary_probe_eval",
            actor_source[compute_log_prob_start:compute_eitr_start],
        )
        self.assertIn(
            "with self._temporary_eitr_probe_checkpoint_train():",
            actor_source[compute_eitr_start:iter_chunks_start],
        )

    def test_deterministic_probe_checkpointing_enables_and_restores_train_mode(self):
        class CheckpointBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gradient_checkpointing = True
                self.attention_dropout = 0.0
                self.dropout = torch.nn.Dropout(p=0.0)

        root = torch.nn.Module()
        root.block = CheckpointBlock()
        root.eval()
        original_states = {
            name: child.training for name, child in root.named_modules()
        }

        CHECKPOINTING.validate_deterministic_probe_checkpointing(root)
        with CHECKPOINTING.deterministic_probe_checkpointing_mode(root):
            self.assertTrue(root.training)
            self.assertTrue(root.block.training)
        self.assertEqual(
            {name: child.training for name, child in root.named_modules()},
            original_states,
        )

        root.train(True)
        root.block.dropout.eval()
        original_mixed_states = {
            name: child.training for name, child in root.named_modules()
        }
        with self.assertRaisesRegex(RuntimeError, "probe failed"):
            with CHECKPOINTING.deterministic_probe_checkpointing_mode(root):
                self.assertTrue(root.block.dropout.training)
                raise RuntimeError("probe failed")
        self.assertEqual(
            {name: child.training for name, child in root.named_modules()},
            original_mixed_states,
        )

    def test_probe_checkpointing_rejects_missing_checkpoint_or_dropout(self):
        plain = torch.nn.Sequential(torch.nn.Linear(2, 2))
        with self.assertRaisesRegex(ValueError, "gradient_checkpointing=True"):
            CHECKPOINTING.validate_deterministic_probe_checkpointing(plain)

        class StochasticBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gradient_checkpointing = True
                self.attention_dropout = 0.1

        stochastic = StochasticBlock()
        with self.assertRaisesRegex(ValueError, "zero dropout"):
            CHECKPOINTING.validate_deterministic_probe_checkpointing(stochastic)

        stochastic.attention_dropout = 0.0
        stochastic.dropout_layer = torch.nn.Dropout(p=0.2)
        with self.assertRaisesRegex(ValueError, "dropout_layer.p=0.2"):
            CHECKPOINTING.validate_deterministic_probe_checkpointing(stochastic)

    def test_old_and_current_eitr_scores_share_checkpoint_train_mode(self):
        root = Path(__file__).resolve().parents[1]
        actor_source = (
            root / "verl" / "workers" / "actor" / "dp_actor.py"
        ).read_text()
        trainer_source = (
            root / "verl" / "trainer" / "ppo" / "ray_trainer.py"
        ).read_text()
        runner = (
            root / "scripts" / "train" / "train_eitr_nq_gate_c_smoke.sh"
        ).read_text()

        compute_log_prob = actor_source[
            actor_source.index("    def compute_log_prob("):
            actor_source.index("    def _compute_log_prob_eval(")
        ]
        compute_eitr = actor_source[
            actor_source.index("    def _compute_eitr_micro_batch("):
            actor_source.index("    def _iter_eitr_state_chunks(")
        ]
        self.assertIn("data.meta_info.get('eitr_probe_score', False)", compute_log_prob)
        self.assertIn("self._temporary_eitr_probe_checkpoint_train", compute_log_prob)
        self.assertIn("self._temporary_eitr_probe_checkpoint_train()", compute_eitr)
        self.assertIn(
            "probe_logprob_batch.meta_info['eitr_probe_score'] = True",
            trainer_source,
        )
        self.assertIn(
            'EITR_PROBE_GRADIENT_CHECKPOINTING="${EITR_PROBE_GRADIENT_CHECKPOINTING:-true}"',
            runner,
        )
        self.assertIn(
            'actor_rollout_ref.actor.eitr.probe_gradient_checkpointing="$EITR_PROBE_GRADIENT_CHECKPOINTING"',
            runner,
        )

    def test_zero_warmup_uses_base_lr_from_the_first_optimizer_step(self):
        scheduler_source_path = (
            Path(__file__).resolve().parents[1]
            / "verl"
            / "utils"
            / "torch_functional.py"
        )
        scheduler_tree = ast.parse(scheduler_source_path.read_text())
        scheduler_method = next(
            node
            for node in scheduler_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "get_constant_schedule_with_warmup"
        )
        namespace = {
            "Optimizer": torch.optim.Optimizer,
            "LambdaLR": torch.optim.lr_scheduler.LambdaLR,
        }
        scheduler_module = ast.fix_missing_locations(
            ast.Module(body=[scheduler_method], type_ignores=[])
        )
        exec(
            compile(scheduler_module, str(scheduler_source_path), "exec"),
            namespace,
        )
        make_scheduler = namespace["get_constant_schedule_with_warmup"]

        base_lr = 5e-7
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=base_lr)
        scheduler = make_scheduler(optimizer, num_warmup_steps=0)

        self.assertEqual(optimizer.param_groups[0]["lr"], base_lr)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], base_lr)
        self.assertEqual(scheduler.get_last_lr(), [base_lr])

    def test_tiny_same_policy_js_is_float64_nonnegative_and_near_zero(self):
        old = torch.tensor(
            [
                [-500.0, -700.0, -900.0, -1100.0],
                [-120.0, -320.0, -520.0, -720.0],
            ],
            dtype=torch.float64,
        )
        documents = torch.tensor(
            [
                [
                    [0.8, 0.2, 0.0],
                    [0.1, 0.7, 0.2],
                    [0.0, 0.3, 0.7],
                    [0.4, 0.1, 0.5],
                ],
                [
                    [0.6, 0.4, 0.0],
                    [0.2, 0.3, 0.5],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ],
            dtype=torch.float32,
        )
        probe_mask = torch.tensor(
            [[True, True, True, True], [True, True, False, False]]
        )

        same_policy = induced_js_from_cached_effects(
            current_seq_logp=old.clone(),
            old_seq_logp=old,
            doc_probs=documents,
            probe_mask=probe_mask,
        )["js"]
        self.assertEqual(same_policy.dtype, torch.float64)
        self.assertTrue(torch.isfinite(same_policy).all())
        self.assertTrue((same_policy >= 0).all())
        self.assertLessEqual(float(same_policy.max().item()), 1e-12)

        perturbation = torch.tensor(
            [[1e-8, -1e-8, 2e-8, -2e-8], [1e-8, -1e-8, 0.0, 0.0]],
            dtype=torch.float64,
        )
        current = (old + perturbation).detach().requires_grad_(True)
        tiny_result = induced_js_from_cached_effects(
            current_seq_logp=current,
            old_seq_logp=old,
            doc_probs=documents,
            probe_mask=probe_mask,
        )
        tiny_gradient = torch.autograd.grad(tiny_result["js"].sum(), current)[0]
        self.assertEqual(tiny_result["js"].dtype, torch.float64)
        self.assertTrue(torch.isfinite(tiny_result["js"]).all())
        self.assertTrue((tiny_result["js"] >= 0).all())
        self.assertTrue(torch.isfinite(tiny_gradient).all())

    def test_proposal_signal_real_noise_fixture_is_inconclusive(self):
        diagnostic = proposal_signal_diagnostics(
            d_old=-1e-8,
            d_pre=-4.6e-10,
            noop_jitter=0.0,
            zero_noop_abs_error=0.0,
            actor_lr=5e-7,
        )

        self.assertEqual(diagnostic["signal"], 0.0)
        self.assertEqual(diagnostic["numerical_floor"], 1e-8)
        self.assertEqual(diagnostic["required_signal"], 1e-7)
        self.assertEqual(diagnostic["score_mode_pass"], 1.0)
        self.assertEqual(diagnostic["pass"], 0.0)
        self.assertEqual(diagnostic["no_signal"], 1.0)

    def test_proposal_signal_accepts_a_resolved_positive_drift(self):
        diagnostic = proposal_signal_diagnostics(
            d_old=0.0,
            d_pre=1e-4,
            noop_jitter=1e-9,
            zero_noop_abs_error=1e-9,
            actor_lr=5e-7,
        )

        self.assertEqual(diagnostic["score_mode_pass"], 1.0)
        self.assertGreater(diagnostic["signal"], diagnostic["required_signal"])
        self.assertGreater(diagnostic["signal_to_floor_ratio"], 10.0)
        self.assertEqual(diagnostic["pass"], 1.0)
        self.assertEqual(diagnostic["no_signal"], 0.0)

    def test_no_signal_skips_candidate_sgd_and_correction_count(self):
        actor_source = (
            Path(__file__).resolve().parents[1]
            / "verl"
            / "workers"
            / "actor"
            / "dp_actor.py"
        ).read_text()
        terminal_start = actor_source.index("        if integrity_fail or no_signal:")
        terminal_return = actor_source.index("            return metrics", terminal_start)
        terminal_block = actor_source[terminal_start:terminal_return]
        self.assertIn("'actor/eitr_score_path_direction_evaluated': 0.0", terminal_block)
        self.assertIn("'actor/eitr_score_path_candidate_sgd_ran': 0.0", terminal_block)
        self.assertIn("'actor/eitr_score_path_audit_inconclusive': float(no_signal)", terminal_block)
        self.assertNotIn("self.eitr_optimizer.step()", terminal_block)
        self.assertNotIn("self._optimizer_step(", terminal_block)

        count_guard = actor_source.index(
            "                if diagnostic_metrics.get(\n"
            "                    'actor/eitr_score_path_candidate_sgd_ran', 0.0"
        )
        count_increment = actor_source.index(
            "                    eitr_correction_optimizer_step_count += 1",
            count_guard,
        )
        self.assertLess(count_guard, count_increment)


if __name__ == "__main__":
    unittest.main()
