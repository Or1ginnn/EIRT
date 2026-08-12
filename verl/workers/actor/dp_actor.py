# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import time
from contextlib import contextmanager
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.eitr import (
    EITR_BATCH_KEYS,
    bounded_quantile_sample_stride,
    cached_probe_fingerprint,
    compacted_state_forward_plan,
    directional_descent_diagnostics,
    eitr_loss_enabled_for_pass,
    eitr_score_path_noop_direction_audit_enabled,
    eitr_update_direction_diagnostic_enabled,
    coverage_weighted_state_scale,
    eitr_probe_enabled_for_pass,
    induced_js_from_cached_effects,
    parameter_correction_diagnostics,
    normalized_correction_step,
    proposal_signal_diagnostics,
    rank_owned_global_additive_stats,
    resolved_query_direction_candidates,
    rollout_averaged_env_drift,
    same_batch_cache_signature,
    same_batch_scale_diagnostic_lrs,
    score_path_audit_event_sequence,
    should_run_post_diagnostic,
    resolve_eitr_mode,
    validate_eitr_optimization_schedule,
)
from verl.trainer.ppo.eitr_checkpointing import (
    deterministic_probe_checkpointing_mode,
    validate_deterministic_probe_checkpointing,
)
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.fsdp_utils import offload_fsdp_optimizer
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        eitr_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.eitr_optimizer = eitr_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.eitr_config = self.config.get('eitr', {})
        self.eitr_mode = resolve_eitr_mode(self.eitr_config)
        self.eitr_uses_probes = self.eitr_mode != 'off'
        self.eitr_lambda_env = float(self.eitr_config.get('lambda_env', 0.1))
        configured_max_update_norm = self.eitr_config.get(
            'correction_max_update_norm', None
        )
        self.eitr_correction_max_update_norm = (
            None
            if configured_max_update_norm is None
            else float(configured_max_update_norm)
        )
        self.eitr_post_diagnostic_freq = int(
            self.eitr_config.get('post_diagnostic_freq', 0)
        )
        self.eitr_same_batch_scale_lrs = same_batch_scale_diagnostic_lrs(
            self.eitr_config
        )
        self.eitr_update_direction_diagnostic = eitr_update_direction_diagnostic_enabled(
            self.eitr_config
        )
        self.eitr_score_path_noop_direction_audit = (
            eitr_score_path_noop_direction_audit_enabled(self.eitr_config)
        )
        self.eitr_score_path_anchor_max_abs_logprob_diff = float(
            self.eitr_config.get(
                'score_path_anchor_max_abs_logprob_diff', 1e-3
            )
        )
        self.ppo_epochs = int(self.config.get('ppo_epochs', 1))
        self.eitr_correction_passes = int(self.eitr_config.get('correction_passes', 1))
        compact_invalid_states = self.eitr_config.get('compact_invalid_states', True)
        if isinstance(compact_invalid_states, str):
            compact_invalid_states = compact_invalid_states.strip().lower() == 'true'
        self.eitr_compact_invalid_states = bool(compact_invalid_states)
        probe_gradient_checkpointing = self.eitr_config.get(
            'probe_gradient_checkpointing', False
        )
        if isinstance(probe_gradient_checkpointing, str):
            probe_gradient_checkpointing = (
                probe_gradient_checkpointing.strip().lower() == 'true'
            )
        self.eitr_probe_gradient_checkpointing_active = bool(
            probe_gradient_checkpointing and self.eitr_mode == 'eitr'
        )
        if self.eitr_probe_gradient_checkpointing_active:
            validate_deterministic_probe_checkpointing(self.actor_module)
        self.offload_actor_optimizer_after_grpo = bool(
            self.config.fsdp_config.get('optimizer_offload', False)
        )
        self.grpo_optimizer_steps_completed = 0
        self.eitr_optimizer_steps_completed = 0
        validate_eitr_optimization_schedule(
            self.eitr_config,
            self.ppo_epochs,
            self.eitr_correction_passes,
        )
        if self.eitr_mode == 'eitr' and self.eitr_optimizer is None:
            raise ValueError('EITR mode requires an independent correction optimizer')
        if self.eitr_mode != 'eitr' and self.eitr_optimizer is not None:
            raise ValueError('Only EITR mode may own a correction optimizer')
        if self.eitr_same_batch_scale_lrs and self.eitr_mode != 'eitr':
            raise ValueError('Same-batch scale diagnostic requires EITR mode')
        if self.eitr_update_direction_diagnostic and self.eitr_mode != 'eitr':
            raise ValueError('Update-direction diagnostic requires EITR mode')
        if self.eitr_score_path_noop_direction_audit and self.eitr_mode != 'eitr':
            raise ValueError('Score-path audit requires EITR mode')
        if sum((
            bool(self.eitr_same_batch_scale_lrs),
            self.eitr_update_direction_diagnostic,
            self.eitr_score_path_noop_direction_audit,
        )) > 1:
            raise ValueError('Only one EITR diagnostic may be enabled at once')

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature, compute_entropy=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = (
                    self.compute_entropy_from_logits(logits_rmpad)
                    if compute_entropy
                    else None
                )

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if compute_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                                gather_dim=0,
                                                                unpad_dim=0,
                                                                padding_size=pad_size)
                # pad back to (bsz, seqlen)
                if compute_entropy:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                             indices=indices,
                                             batch=batch_size,
                                             seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = (
                    full_entropy.squeeze(-1)[:, -response_length - 1:-1]
                    if compute_entropy
                    else None
                )
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits) if compute_entropy else None

            return entropy, log_probs

    def _optimizer_step(self, optimizer=None):
        assert self.config.grad_clip is not None

        optimizer = optimizer or self.actor_optimizer

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        optimizer.step()
        return grad_norm

    def _normalized_eitr_optimizer_step(self):
        """Apply one stateless EITR SGD step with a global norm ceiling."""
        if self.eitr_optimizer is None:
            raise RuntimeError('Normalized EITR step requires its SGD optimizer')
        base_lrs = [float(group['lr']) for group in self.eitr_optimizer.param_groups]
        if not base_lrs or any(abs(lr - base_lrs[0]) > 1e-15 for lr in base_lrs[1:]):
            raise RuntimeError('Normalized EITR requires one shared correction LR')

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(
                max_norm=self.config.grad_clip
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module.parameters(), max_norm=self.config.grad_clip
            )
        step = normalized_correction_step(
            grad_norm=float(grad_norm.detach().item()),
            correction_lr=base_lrs[0],
            grad_clip=float(self.config.grad_clip),
            max_update_norm=self.eitr_correction_max_update_norm,
        )
        try:
            for group, base_lr in zip(self.eitr_optimizer.param_groups, base_lrs):
                group['lr'] = base_lr * step['normalization_scale']
            self.eitr_optimizer.step()
        finally:
            # The configured LR remains the fixed upper bound for the next
            # batch; normalization is recomputed independently per correction.
            for group, base_lr in zip(self.eitr_optimizer.param_groups, base_lrs):
                group['lr'] = base_lr
        return grad_norm, step

    @contextmanager
    def _temporary_probe_eval(self):
        """Use one deterministic probe-scoring mode and always restore it."""
        actor_was_training = self.actor_module.training
        self.actor_module.eval()
        try:
            yield
        finally:
            self.actor_module.train(actor_was_training)

    @contextmanager
    def _temporary_eitr_probe_checkpoint_train(self):
        """Enable deterministic train-mode scoring so HF checkpoints activate."""
        if not self.eitr_probe_gradient_checkpointing_active:
            with self._temporary_probe_eval():
                yield
            return
        with deterministic_probe_checkpointing_mode(self.actor_module):
            if not self.actor_module.training:
                raise RuntimeError(
                    'EITR probe gradient checkpointing failed to enter train mode'
                )
            yield

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute old log-probs in the matching ordinary/probe score mode."""
        is_eitr_probe_score = bool(data.meta_info.get('eitr_probe_score', False))
        score_context = (
            self._temporary_eitr_probe_checkpoint_train
            if is_eitr_probe_score
            else self._temporary_probe_eval
        )
        with score_context():
            return self._compute_log_prob_eval(data)

    def _compute_log_prob_eval(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def _compute_eitr_micro_batch(self, data, temperature, *, track_model_grad=True):
        """Run cached same-state probes and return a differentiable induced-JS mean.

        Invalid states still execute the same dummy probe forward on every FSDP
        rank, but their response masks make the returned loss exactly zero.
        """
        state_slot = data['eitr_state_slot'].bool()
        slot_count = int(state_slot.sum().item())
        if slot_count == 0:
            return None

        probe_valid = data['eitr_probe_valid'][state_slot].bool()
        state_valid = data['eitr_state_valid'][state_slot].bool()
        probe_count = probe_valid.size(1)

        probe_batch = {
            'input_ids': data['eitr_probe_input_ids'][state_slot].flatten(0, 1),
            'attention_mask': data['eitr_probe_attention_mask'][state_slot].flatten(0, 1),
            'position_ids': data['eitr_probe_position_ids'][state_slot].flatten(0, 1),
            'responses': data['eitr_probe_responses'][state_slot].flatten(0, 1),
        }
        probe_micro_batch_size = int(self.eitr_config.get('probe_micro_batch_size', 4))
        token_log_prob_chunks = []
        flat_probe_count = int(probe_batch['input_ids'].size(0))
        # Every valid global probe batch gives each FSDP rank the same number of
        # physical row slots. Fixed-size chunking therefore preserves identical
        # forward/collective counts while avoiding one K*actor_microbatch prefill.
        # Cached-old, gradient, no-op, and candidate probe scores share this
        # deterministic mode. For Qwen2.5, zero-dropout train mode activates
        # Hugging Face layer checkpointing and avoids retaining every long-state
        # activation until the EITR correction backward.
        with self._temporary_eitr_probe_checkpoint_train():
            for chunk_start in range(0, flat_probe_count, probe_micro_batch_size):
                chunk_end = min(chunk_start + probe_micro_batch_size, flat_probe_count)
                probe_chunk = {
                    key: value[chunk_start:chunk_end]
                    for key, value in probe_batch.items()
                }
                with torch.set_grad_enabled(track_model_grad):
                    _, chunk_log_probs = self._forward_micro_batch(
                        micro_batch=probe_chunk,
                        temperature=temperature,
                        compute_entropy=False,
                    )
                token_log_prob_chunks.append(chunk_log_probs)
        token_log_probs = torch.cat(token_log_prob_chunks, dim=0)
        response_mask = data['eitr_probe_response_mask'][state_slot].flatten(0, 1).float()
        current_seq_logp = (
            token_log_probs.double() * response_mask.double()
        ).sum(dim=-1).view(
            slot_count, probe_count
        )
        if not track_model_grad:
            # Probe-GRPO needs the same candidate re-scoring forward, but no
            # model backward or optimizer update. A detached leaf still lets us
            # report the estimator's query-logprob gradient proxy.
            current_seq_logp = current_seq_logp.detach().requires_grad_(True)

        if not state_valid.any():
            # Preserve identical FSDP forward/backward call structure while an
            # ineligible group contributes exactly zero gradient.
            zero = current_seq_logp.sum() * 0.0
            return {
                'loss': zero,
                'current_seq_logp': current_seq_logp,
                'valid_state_count': 0,
                'valid_probe_count': 0,
                'js_sum': 0.0,
                'js_mean': 0.0,
                'ess_mean': 0.0,
                'log_ratio_abs_max': 0.0,
                'log_ratio_clipfrac': 0.0,
            }

        estimates = induced_js_from_cached_effects(
            current_seq_logp=current_seq_logp[state_valid],
            old_seq_logp=data['eitr_probe_old_seq_logp'][state_slot][state_valid],
            doc_probs=data['eitr_probe_doc_probs'][state_slot][state_valid],
            probe_mask=probe_valid[state_valid],
            log_ratio_clip=float(self.eitr_config.get('log_ratio_clip', 10.0)),
        )
        js = estimates['js']
        return {
            'loss': js.mean(),
            'current_seq_logp': current_seq_logp,
            'valid_state_count': int(js.numel()),
            'valid_probe_count': int(probe_valid[state_valid].sum().item()),
            'js_sum': float(js.detach().sum().item()),
            'js_mean': float(js.detach().mean().item()),
            'ess_mean': float(estimates['ess'].detach().mean().item()),
            'log_ratio_abs_max': float(estimates['log_ratio_abs_max'].detach().max().item()),
            'log_ratio_clipfrac': float(estimates['log_ratio_clipfrac'].detach().mean().item()),
        }

    def _iter_eitr_state_chunks(self, mini_batch):
        """Yield fixed-state chunks while keeping all K probes of a state together."""
        if self.config.use_dynamic_bsz:
            max_token_len = (
                self.config.ppo_max_token_len_per_gpu
                * self.ulysses_sequence_parallel_size
            )
            rollout_micro_batches, _ = rearrange_micro_batches(
                batch=mini_batch,
                max_token_len=max_token_len,
            )
        else:
            rollout_micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

        for rollout_micro_batch in rollout_micro_batches:
            physical_probe_count = int(
                rollout_micro_batch['eitr_probe_valid'].size(1)
            )
            probe_micro_batch_size = int(
                self.eitr_config.get('probe_micro_batch_size', 4)
            )
            states_per_probe_chunk = max(
                probe_micro_batch_size // physical_probe_count,
                1,
            )
            yield from rollout_micro_batch.split(states_per_probe_chunk)

    def _prepare_eitr_correction_state_chunks(self, dataloader, *, distributed):
        """Drop invalid online states while preserving rank-symmetric FSDP calls."""

        probe_source = str(self.eitr_config.get('probe_source', 'online_same_state'))
        if not self.eitr_compact_invalid_states or probe_source != 'online_same_state':
            state_chunks = [
                state_chunk
                for mini_batch in dataloader
                for state_chunk in self._iter_eitr_state_chunks(mini_batch)
            ]
            physical_count = sum(
                int(mini_batch['eitr_state_valid'].numel()) for mini_batch in dataloader
            )
            return state_chunks, {
                'actor/eitr_state_compaction_enabled': 0.0,
                'actor/eitr_physical_state_count_before_compaction': float(physical_count),
                'actor/eitr_compacted_state_forward_count': float(physical_count),
                'actor/eitr_dummy_state_forward_count': float(
                    physical_count
                    - sum(
                        int(mini_batch['eitr_state_valid'].sum().item())
                        for mini_batch in dataloader
                    )
                ),
                'actor/eitr_state_forward_reduction_rate': 0.0,
            }

        active_rows = []
        dummy_row = None
        local_physical_count = 0
        for mini_batch in dataloader:
            state_valid = mini_batch['eitr_state_valid'].bool()
            state_slot = mini_batch['eitr_state_slot'].bool()
            if bool((state_valid & ~state_slot).any().item()):
                raise RuntimeError('EITR valid state is missing its physical slot')
            local_physical_count += int(state_valid.numel())
            for row_index in torch.nonzero(state_valid, as_tuple=False).flatten().tolist():
                active_rows.append(mini_batch[row_index:row_index + 1])
            if dummy_row is None:
                dummy_indices = torch.nonzero(
                    state_slot & ~state_valid, as_tuple=False
                ).flatten()
                if int(dummy_indices.numel()) > 0:
                    row_index = int(dummy_indices[0].item())
                    dummy_row = mini_batch[row_index:row_index + 1]

        local_active_count = len(active_rows)
        count_tensor = torch.tensor(
            [float(local_active_count), float(local_physical_count)],
            dtype=torch.float64,
            device=torch.cuda.current_device(),
        )
        max_counts = count_tensor.clone()
        min_physical_count = count_tensor[1].clone()
        if distributed:
            torch.distributed.all_reduce(max_counts, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(
                min_physical_count, op=torch.distributed.ReduceOp.MIN
            )
        max_active_count = int(max_counts[0].item())
        max_physical_count = int(max_counts[1].item())
        if int(min_physical_count.item()) != max_physical_count:
            raise RuntimeError(
                'EITR compaction requires equal physical rollout rows on every FSDP rank'
            )

        probe_count = int(dataloader[0]['eitr_probe_valid'].size(1))
        probe_micro_batch_size = int(
            self.eitr_config.get('probe_micro_batch_size', 4)
        )
        states_per_chunk = max(probe_micro_batch_size // probe_count, 1)
        plan = compacted_state_forward_plan(
            local_active_count=local_active_count,
            local_physical_count=local_physical_count,
            max_active_count=max_active_count,
            states_per_chunk=states_per_chunk,
        )
        target_state_count = int(plan['target_state_count'])

        if target_state_count > local_active_count:
            if dummy_row is None:
                if not active_rows:
                    raise RuntimeError(
                        'EITR compaction could not construct a rank-symmetric dummy state'
                    )
                # This only covers the generic all-active, odd-tail case. Keep
                # the real cached tensors but zero every applicability mask.
                dummy_row = active_rows[0].clone()
                dummy_row['eitr_state_valid'].zero_()
                dummy_row['eitr_probe_valid'].zero_()
                dummy_row['eitr_probe_response_mask'].zero_()
            active_rows.extend(
                [dummy_row] * (target_state_count - local_active_count)
            )

        state_chunks = []
        for start in range(0, target_state_count, states_per_chunk):
            rows = active_rows[start:start + states_per_chunk]
            state_chunks.append(rows[0] if len(rows) == 1 else torch.cat(rows, dim=0))
        if len(state_chunks) != int(plan['chunk_count']):
            raise RuntimeError('EITR compaction produced an inconsistent chunk count')

        return state_chunks, {
            'actor/eitr_state_compaction_enabled': 1.0,
            'actor/eitr_physical_state_count_before_compaction': float(
                local_physical_count
            ),
            'actor/eitr_compacted_state_forward_count': float(target_state_count),
            'actor/eitr_dummy_state_forward_count': float(plan['dummy_state_count']),
            'actor/eitr_state_forward_reduction_rate': float(
                plan['forward_reduction_rate']
            ),
        }

    @staticmethod
    def _eitr_chunk_to_cuda(state_chunk):
        """Move only cached-probe fields for the current state chunk to GPU."""
        return state_chunk.select(*EITR_BATCH_KEYS).cuda()

    def _snapshot_eitr_local_state(self):
        """Save only each rank's FSDP-local shard and correction gradient on CPU."""
        snapshot = []
        for parameter in self.actor_module.parameters():
            if not parameter.requires_grad:
                continue
            snapshot.append((
                parameter,
                parameter.detach().to(device='cpu', copy=True),
                None if parameter.grad is None else parameter.grad.detach().to(
                    device='cpu', copy=True
                ),
            ))
        if not snapshot:
            raise RuntimeError('Same-batch diagnostic found no trainable actor shards')
        return snapshot

    @staticmethod
    def _restore_eitr_local_state(snapshot, *, restore_gradients):
        with torch.no_grad():
            for parameter, saved_parameter, saved_gradient in snapshot:
                parameter.copy_(saved_parameter.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                    non_blocking=True,
                ))
                if restore_gradients:
                    parameter.grad = (
                        None if saved_gradient is None else saved_gradient.to(
                            device=parameter.device,
                            dtype=parameter.dtype,
                            non_blocking=True,
                        )
                    )

    @staticmethod
    def _local_snapshot_max_abs_error(snapshot):
        maximum = 0.0
        for parameter, saved_parameter, _ in snapshot:
            maximum = max(
                maximum,
                float((parameter.detach().float().cpu() - saved_parameter.float()).abs().max().item()),
            )
        return maximum

    @staticmethod
    def _local_snapshot_delta_sq(snapshot):
        total = 0.0
        for parameter, saved_parameter, _ in snapshot:
            total += float((
                parameter.detach().float().cpu() - saved_parameter.float()
            ).square().sum().item())
        return total

    @staticmethod
    def _distributed_scalar(value, *, distributed, op):
        tensor = torch.tensor(
            float(value), dtype=torch.float64, device=torch.cuda.current_device()
        )
        if distributed:
            torch.distributed.all_reduce(tensor, op=op)
        return float(tensor.item())

    def _score_cached_eitr_drift(
        self,
        dataloader,
        temperature,
        *,
        distributed,
        state_chunks=None,
    ):
        """Re-score exactly the existing cached probes without gradient or retrieval."""
        js_sum = 0.0
        state_count = 0.0
        if state_chunks is not None:
            for state_chunk in state_chunks:
                result = self._compute_eitr_micro_batch(
                    self._eitr_chunk_to_cuda(state_chunk),
                    temperature,
                    track_model_grad=False,
                )
                if result is not None:
                    js_sum += result['js_sum']
                    state_count += result['valid_state_count']
                del result
            result_tensor = torch.tensor(
                [js_sum, state_count],
                dtype=torch.float64,
                device=torch.cuda.current_device(),
            )
            if distributed:
                torch.distributed.all_reduce(
                    result_tensor, op=torch.distributed.ReduceOp.SUM
                )
            return result_tensor

        for mini_batch in dataloader:
            mini_global_state_count = torch.tensor(
                float(mini_batch['eitr_state_valid'].sum().item()),
                dtype=torch.float64,
                device=torch.cuda.current_device(),
            )
            if distributed:
                torch.distributed.all_reduce(
                    mini_global_state_count, op=torch.distributed.ReduceOp.SUM
                )
            if mini_global_state_count.item() <= 0:
                continue
            for state_chunk in self._iter_eitr_state_chunks(mini_batch):
                result = self._compute_eitr_micro_batch(
                    self._eitr_chunk_to_cuda(state_chunk),
                    temperature,
                    track_model_grad=False,
                )
                if result is not None:
                    js_sum += result['js_sum']
                    state_count += result['valid_state_count']
                del result
        result_tensor = torch.tensor(
            [js_sum, state_count], dtype=torch.float64, device=torch.cuda.current_device()
        )
        if distributed:
            torch.distributed.all_reduce(result_tensor, op=torch.distributed.ReduceOp.SUM)
        return result_tensor

    def _audit_score_cached_eitr_drift(
        self,
        dataloader,
        temperature,
        *,
        distributed,
        global_rollout_state_count,
        eitr_world_size,
        build_grad,
        capture_query_sample=False,
    ):
        """One numerical score path for every D value in the audit.

        ``build_grad`` changes only autograd retention. The forwards, state
        chunking, dtype/autocast path, cached tensors, and distributed reduction
        are intentionally identical for D_old, no-op, and +/- candidates.
        """
        local = {
            'js_sum': 0.0,
            'state_count': 0.0,
            'probe_count': 0.0,
            'ess_sum': 0.0,
            'clipfrac_sum': 0.0,
            'log_ratio_abs_max': 0.0,
        }
        query_sample = None
        query_sample_grad_energy = -1.0
        for mini_batch in dataloader:
            for state_chunk in self._iter_eitr_state_chunks(mini_batch):
                state_chunk = self._eitr_chunk_to_cuda(state_chunk)
                result = self._compute_eitr_micro_batch(
                    state_chunk, temperature, track_model_grad=build_grad
                )
                if result is None:
                    continue
                valid_state_count = result['valid_state_count']
                if valid_state_count > 0:
                    local['js_sum'] += result['js_sum']
                    local['state_count'] += valid_state_count
                    local['probe_count'] += result['valid_probe_count']
                    local['ess_sum'] += result['ess_mean'] * valid_state_count
                    local['clipfrac_sum'] += result['log_ratio_clipfrac'] * valid_state_count
                    local['log_ratio_abs_max'] = max(
                        local['log_ratio_abs_max'], result['log_ratio_abs_max']
                    )
                if build_grad:
                    raw_logprob_grad = torch.autograd.grad(
                        result['loss'], result['current_seq_logp'], retain_graph=True,
                        allow_unused=True,
                    )[0]
                    if (
                        capture_query_sample
                        and valid_state_count > 0
                        and raw_logprob_grad is not None
                    ):
                        state_slot = state_chunk['eitr_state_slot'].bool()
                        state_valid = state_chunk['eitr_state_valid'][state_slot].bool()
                        if state_valid.any():
                            valid_gradient = raw_logprob_grad[state_valid]
                            gradient_energy = (
                                valid_gradient.detach().float().square().sum(dim=-1)
                            )
                            finite_positive = torch.isfinite(gradient_energy) & (
                                gradient_energy > 0
                            )
                            if finite_positive.any():
                                ranked_energy = gradient_energy.masked_fill(
                                    ~finite_positive, float('-inf')
                                )
                                best_index = int(ranked_energy.argmax().item())
                                best_energy = float(ranked_energy[best_index].item())
                            else:
                                best_index = -1
                                best_energy = -1.0
                            if best_index >= 0 and best_energy > query_sample_grad_energy:
                                query_sample_grad_energy = best_energy
                                query_sample = tuple(
                                    value[best_index:best_index + 1].detach().to(
                                        device='cpu', copy=True
                                    )
                                    for value in (
                                        result['current_seq_logp'][state_valid],
                                        valid_gradient,
                                        state_chunk['eitr_probe_old_seq_logp'][state_slot][state_valid],
                                        state_chunk['eitr_probe_doc_probs'][state_slot][state_valid],
                                        state_chunk['eitr_probe_valid'][state_slot][state_valid],
                                    )
                                )
                    state_weight = coverage_weighted_state_scale(
                        valid_state_count,
                        float(global_rollout_state_count.item()),
                        world_size=eitr_world_size,
                    )
                    # Every FSDP rank must execute the matching backward. For
                    # an inactive local chunk ``loss`` and ``state_weight`` are
                    # both zero, so this preserves collectives without changing
                    # the full-batch gradient.
                    (result['loss'] * self.eitr_lambda_env * state_weight).backward()
                    del raw_logprob_grad
                del result

        tensor = torch.tensor(
            [
                local['js_sum'], local['state_count'], local['probe_count'],
                local['ess_sum'], local['clipfrac_sum'],
            ],
            dtype=torch.float64,
            device=torch.cuda.current_device(),
        )
        if distributed:
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        log_ratio_abs_max = torch.tensor(
            local['log_ratio_abs_max'],
            dtype=torch.float64,
            device=torch.cuda.current_device(),
        )
        if distributed:
            torch.distributed.all_reduce(
                log_ratio_abs_max, op=torch.distributed.ReduceOp.MAX
            )
        if int(tensor[1].item()) <= 0:
            raise RuntimeError('Score-path audit found no valid cached EITR states')
        if not torch.isfinite(tensor).all():
            raise RuntimeError('Score-path audit produced a non-finite cached-probe score')
        return {
            'drift': rollout_averaged_env_drift(
                tensor[0].item(), global_rollout_state_count.item()
            ),
            'js_sum': float(tensor[0].item()),
            'state_count': int(tensor[1].item()),
            'probe_count': int(tensor[2].item()),
            'ess': float(tensor[3].item() / max(tensor[1].item(), 1.0)),
            'clipfrac': float(tensor[4].item() / max(tensor[1].item(), 1.0)),
            'log_ratio_abs_max': float(log_ratio_abs_max.item()),
            'query_sample': query_sample,
        }

    def _snapshot_checksum(self, snapshot, *, distributed):
        """Checksum FSDP-local CPU shards without triggering a full all-gather."""
        local = [0.0, 0.0, 0.0]
        for _, saved_parameter, _ in snapshot:
            value = saved_parameter.float()
            local[0] += float(value.sum(dtype=torch.float64).item())
            local[1] += float(value.square().sum(dtype=torch.float64).item())
            local[2] += float(value.abs().sum(dtype=torch.float64).item())
        return tuple(
            self._distributed_scalar(
                value, distributed=distributed, op=torch.distributed.ReduceOp.SUM
            )
            for value in local
        )

    def _audit_direction_statistics(self, snapshot, *, direction, distributed):
        # Exact norms/dot/counts are accumulated a bounded CPU chunk at a time.
        # Quantiles are diagnostic only and use a deterministic, candidate-
        # independent strided sample. Never concatenate a 3B-parameter delta or
        # pass an FSDP flat shard directly to torch.quantile().
        chunk_size = 1_048_576
        global_sample_budget = 262_144
        world_size = torch.distributed.get_world_size() if distributed else 1
        local_sample_budget = max(1, global_sample_budget // world_size)
        local_total_values = sum(
            int(saved_parameter.numel())
            for _, saved_parameter, saved_gradient in snapshot
            if saved_gradient is not None
        )
        sample_stride = bounded_quantile_sample_stride(
            local_total_values, local_sample_budget
        )
        local_values = [0.0] * 6
        local_max_abs = 0.0
        local_offset = 0
        sampled_absolute = []
        for parameter, saved_parameter, saved_gradient in snapshot:
            if saved_gradient is None:
                continue
            parameter_flat = parameter.detach().reshape(-1)
            saved_flat = saved_parameter.reshape(-1)
            gradient_flat = saved_gradient.reshape(-1)
            if not (
                parameter_flat.numel()
                == saved_flat.numel()
                == gradient_flat.numel()
            ):
                raise RuntimeError(
                    'Score-path audit parameter/snapshot/gradient shapes changed'
                )
            for chunk_start in range(0, parameter_flat.numel(), chunk_size):
                chunk_end = min(chunk_start + chunk_size, parameter_flat.numel())
                current_chunk = parameter_flat[chunk_start:chunk_end].to(
                    device='cpu', dtype=torch.float32, copy=True
                )
                saved_chunk = saved_flat[chunk_start:chunk_end].float()
                gradient_chunk = gradient_flat[chunk_start:chunk_end].float()
                delta = current_chunk - saved_chunk
                absolute = delta.abs()
                local_values[0] += float(
                    gradient_chunk.square().sum(dtype=torch.float64).item()
                )
                local_values[1] += float(
                    delta.square().sum(dtype=torch.float64).item()
                )
                local_values[2] += float(
                    (gradient_chunk * delta).sum(dtype=torch.float64).item()
                )
                local_values[3] += float((absolute > 0).sum().item())
                local_values[4] += float(absolute.numel())
                if absolute.numel() > 0:
                    local_max_abs = max(local_max_abs, float(absolute.max().item()))
                    first_sample = (-local_offset) % sample_stride
                    if first_sample < absolute.numel():
                        sampled_absolute.append(
                            absolute[first_sample::sample_stride].clone()
                        )
                local_offset += absolute.numel()

        if sampled_absolute:
            absolute_sample = torch.cat(sampled_absolute)[:local_sample_budget]
            local_values[5] = float(absolute_sample.numel())
            local_quantiles_cpu = torch.quantile(
                absolute_sample,
                torch.tensor([0.5, 0.95, 0.99], dtype=torch.float32),
            ).double()
        else:
            local_quantiles_cpu = torch.zeros(3, dtype=torch.float64)

        local_sums = torch.tensor(
            local_values, dtype=torch.float64, device=torch.cuda.current_device()
        )
        audit_device = torch.device('cuda', torch.cuda.current_device())
        local_quantiles = local_quantiles_cpu.to(device=audit_device)
        local_maxima = torch.tensor(
            [local_max_abs, float(sample_stride)],
            dtype=torch.float64,
            device=torch.cuda.current_device(),
        )
        if distributed:
            torch.distributed.all_reduce(local_sums, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_quantiles, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(local_maxima, op=torch.distributed.ReduceOp.MAX)
        grad_norm = float(local_sums[0].clamp_min(0).sqrt().item())
        delta_norm = float(local_sums[1].clamp_min(0).sqrt().item())
        g_dot_delta = float(local_sums[2].item())
        signed_dot = -g_dot_delta if direction == 'minus' else g_dot_delta
        cosine = signed_dot / max(grad_norm * delta_norm, 1e-30)
        return {
            'grad_norm': grad_norm,
            'delta_norm': delta_norm,
            'g_dot_delta': g_dot_delta,
            'cosine': cosine,
            'changed_fraction': float(local_sums[3].item() / max(local_sums[4].item(), 1.0)),
            'delta_abs_shard_p50_max': float(local_quantiles[0].item()),
            'delta_abs_shard_p95_max': float(local_quantiles[1].item()),
            'delta_abs_shard_p99_max': float(local_quantiles[2].item()),
            'delta_abs_max': float(local_maxima[0].item()),
            'quantile_sample_count': int(local_sums[5].item()),
            'quantile_sample_stride_max': int(local_maxima[1].item()),
            'saw_delta': bool(local_sums[3].item() > 0),
        }

    def _run_same_batch_scale_diagnostic(
        self,
        dataloader,
        temperature,
        *,
        distributed,
        global_active_state_count,
        global_rollout_state_count,
        d_zero,
    ):
        """Evaluate three stateless SGD scales from one exact theta_GRPO snapshot."""
        if self.eitr_optimizer.state:
            raise RuntimeError('Same-batch diagnostic requires stateless correction SGD')

        cache_signature = tuple(
            same_batch_cache_signature(mini_batch) for mini_batch in dataloader
        )
        snapshot = self._snapshot_eitr_local_state()
        original_lrs = [group['lr'] for group in self.eitr_optimizer.param_groups]
        metrics = {
            'actor/eitr_same_batch_scale_diagnostic': 1.0,
            'actor/eitr_same_batch_cache_reused': 1.0,
            'actor/eitr_same_batch_d_zero': float(d_zero),
        }
        candidate_report = []
        try:
            for learning_rate in self.eitr_same_batch_scale_lrs:
                self._restore_eitr_local_state(snapshot, restore_gradients=True)
                start_error = self._distributed_scalar(
                    self._local_snapshot_max_abs_error(snapshot),
                    distributed=distributed,
                    op=torch.distributed.ReduceOp.MAX,
                )
                if start_error != 0.0:
                    raise RuntimeError(
                        'Same-batch diagnostic could not restore the common theta_GRPO start'
                    )
                for group in self.eitr_optimizer.param_groups:
                    group['lr'] = float(learning_rate)
                correction_grad_norm = self._optimizer_step(self.eitr_optimizer)
                update_norm = self._distributed_scalar(
                    self._local_snapshot_delta_sq(snapshot),
                    distributed=distributed,
                    op=torch.distributed.ReduceOp.SUM,
                ) ** 0.5
                if tuple(
                    same_batch_cache_signature(mini_batch) for mini_batch in dataloader
                ) != cache_signature:
                    raise RuntimeError('Same-batch diagnostic mutated cached probe tensors')
                post_stats = self._score_cached_eitr_drift(
                    dataloader, temperature, distributed=distributed
                )
                if int(post_stats[1].item()) != int(global_active_state_count.item()):
                    raise RuntimeError(
                        'Same-batch diagnostic re-score did not cover every active state'
                    )
                d_post = rollout_averaged_env_drift(
                    post_stats[0].item(), global_rollout_state_count.item()
                )
                label = f'{learning_rate:.0e}'.replace('e-', 'e')
                metrics.update({
                    f'actor/eitr_same_batch_d_post_lr_{label}': float(d_post),
                    f'actor/eitr_same_batch_d_delta_lr_{label}': float(d_post - d_zero),
                    f'actor/eitr_same_batch_update_norm_lr_{label}': float(update_norm),
                    f'actor/eitr_same_batch_start_max_abs_lr_{label}': float(start_error),
                    f'actor/eitr_same_batch_grad_norm_lr_{label}': float(
                        correction_grad_norm.detach().item()
                    ),
                })
                candidate_report.append((learning_rate, d_post, update_norm, start_error))
                self.eitr_optimizer.zero_grad()
        finally:
            self._restore_eitr_local_state(snapshot, restore_gradients=False)
            self.eitr_optimizer.zero_grad()
            for group, original_lr in zip(self.eitr_optimizer.param_groups, original_lrs):
                group['lr'] = original_lr

        final_restore_error = self._distributed_scalar(
            self._local_snapshot_max_abs_error(snapshot),
            distributed=distributed,
            op=torch.distributed.ReduceOp.MAX,
        )
        if final_restore_error != 0.0:
            raise RuntimeError('Same-batch diagnostic failed to restore theta_GRPO')
        metrics['actor/eitr_same_batch_final_restore_max_abs'] = final_restore_error
        if not distributed or torch.distributed.get_rank() == 0:
            values = ' '.join(
                f'lr={lr:.0e} D_post={d_post:.12e} D_delta={d_post - d_zero:.12e} '
                f'update_norm={update_norm:.12e} start_max_abs={start_error:.12e}'
                for lr, d_post, update_norm, start_error in candidate_report
            )
            print(f'EITR_SAME_BATCH_SCALE_DIAGNOSTIC D_zero={d_zero:.12e} {values}')
        return metrics

    @staticmethod
    def _apply_snapshot_gradient(snapshot, scale):
        """Apply one explicit +/- epsilon*g update to FSDP-local parameter shards."""
        with torch.no_grad():
            for parameter, _, saved_gradient in snapshot:
                if saved_gradient is not None:
                    parameter.add_(saved_gradient.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                        non_blocking=True,
                    ), alpha=float(scale))

    def _directional_parameter_statistics(self, snapshot, *, direction, distributed):
        local_grad_sq = 0.0
        local_delta_sq = 0.0
        local_g_dot_delta = 0.0
        for parameter, saved_parameter, saved_gradient in snapshot:
            if saved_gradient is None:
                continue
            delta = parameter.detach().float().cpu() - saved_parameter.float()
            gradient = saved_gradient.float()
            local_grad_sq += float(gradient.square().sum().item())
            local_delta_sq += float(delta.square().sum().item())
            local_g_dot_delta += float((gradient * delta).sum().item())
        stats = torch.tensor(
            [local_grad_sq, local_delta_sq, local_g_dot_delta],
            dtype=torch.float64,
            device=torch.cuda.current_device(),
        )
        if distributed:
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        grad_norm = float(stats[0].clamp_min(0).sqrt().item())
        delta_norm = float(stats[1].clamp_min(0).sqrt().item())
        g_dot_delta = float(stats[2].item())
        signed_dot = -g_dot_delta if direction == 'minus' else g_dot_delta
        cosine = signed_dot / max(grad_norm * delta_norm, 1e-30)
        return grad_norm, delta_norm, g_dot_delta, cosine

    def _query_logprob_direction_metrics(
        self,
        query_sample,
        *,
        distributed,
    ):
        """Check the JS derivative in resolved query-logprob space."""
        device = torch.cuda.current_device()
        local_sums = torch.zeros(5, dtype=torch.float64, device=device)
        local_bounds = torch.tensor(
            [float('inf'), float('inf'), float('inf'), 0.0, 0.0],
            dtype=torch.float64,
            device=device,
        )
        local_invalid = torch.zeros(1, dtype=torch.float64, device=device)
        if query_sample is not None:
            try:
                current, gradient, old, doc_probs, probe_mask = (
                    value.cuda() for value in query_sample
                )
                candidates = resolved_query_direction_candidates(current, gradient)
                with torch.no_grad():
                    values = []
                    for candidate in (
                        candidates['minus'],
                        candidates['zero'],
                        candidates['plus'],
                    ):
                        result = induced_js_from_cached_effects(
                            current_seq_logp=candidate,
                            old_seq_logp=old,
                            doc_probs=doc_probs,
                            probe_mask=probe_mask,
                            log_ratio_clip=float(
                                self.eitr_config.get('log_ratio_clip', 10.0)
                            ),
                        )
                        values.append(float(result['js'].mean().item()))
                diagnostic = directional_descent_diagnostics(
                    minus=values[0], zero=values[1], plus=values[2], atol=1e-8
                )
                sample_count = float(current.size(0))
                local_sums = torch.tensor(
                    [
                        values[0] * sample_count,
                        values[1] * sample_count,
                        values[2] * sample_count,
                        candidates['grad_norm'],
                        sample_count,
                    ],
                    dtype=torch.float64,
                    device=device,
                )
                local_bounds = torch.tensor(
                    [
                        diagnostic['minus_margin'],
                        diagnostic['plus_margin'],
                        candidates['resolved_grad_energy'],
                        diagnostic['noise'],
                        candidates['target_logp_delta'],
                    ],
                    dtype=torch.float64,
                    device=device,
                )
            except ValueError:
                # A data-dependent failure on only one rank must not leave the
                # other ranks blocked in the reductions below.  Convert it to
                # a synchronized integrity failure after every rank arrives.
                local_invalid.fill_(1.0)
        if distributed:
            torch.distributed.all_reduce(
                local_invalid, op=torch.distributed.ReduceOp.MAX
            )
            torch.distributed.all_reduce(
                local_sums, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                local_bounds[:3], op=torch.distributed.ReduceOp.MIN
            )
            torch.distributed.all_reduce(
                local_bounds[3:], op=torch.distributed.ReduceOp.MAX
            )
        if local_invalid.item() > 0:
            raise RuntimeError(
                'Query-logprob direction diagnostic received a non-finite or '
                'numerically unresolved local sample'
            )
        if local_sums[4].item() <= 0:
            raise RuntimeError('Update-direction diagnostic captured no query-logprob sample')
        averages = local_sums[:3] / local_sums[4]
        if not torch.isfinite(averages).all() or not torch.isfinite(local_bounds).all():
            raise RuntimeError('Query-logprob direction diagnostic produced a non-finite JS')
        query_pass = bool(
            local_bounds[0].item() > local_bounds[3].item()
            and local_bounds[1].item() > local_bounds[3].item()
            and local_bounds[2].item() >= 0.99
        )
        return {
            'minus': float(averages[0].item()),
            'zero': float(averages[1].item()),
            'plus': float(averages[2].item()),
            'minus_margin_min': float(local_bounds[0].item()),
            'plus_margin_min': float(local_bounds[1].item()),
            'resolved_grad_energy_min': float(local_bounds[2].item()),
            'noise_max': float(local_bounds[3].item()),
            'target_logp_delta_max': float(local_bounds[4].item()),
            'grad_norm_sum': float(local_sums[3].item()),
            'sample_count': int(local_sums[4].item()),
            'pass': float(query_pass),
        }

    def _run_update_direction_diagnostic(
        self,
        dataloader,
        temperature,
        *,
        distributed,
        global_active_state_count,
        global_rollout_state_count,
        d_zero,
        query_sample,
        epsilon=3e-5,
    ):
        """Audit +/- epsilon*g from the exact same theta_GRPO and cached probes."""
        cache_signature = tuple(
            same_batch_cache_signature(mini_batch) for mini_batch in dataloader
        )
        snapshot = self._snapshot_eitr_local_state()
        metrics = {
            'actor/eitr_update_direction_diagnostic': 1.0,
            'actor/eitr_update_direction_cache_reused': 1.0,
            'actor/eitr_update_direction_d_zero': float(d_zero),
            'actor/eitr_update_direction_epsilon': float(epsilon),
        }
        query_direction = self._query_logprob_direction_metrics(
            query_sample, distributed=distributed
        )
        metrics.update({
            'actor/eitr_update_direction_query_js_minus': query_direction['minus'],
            'actor/eitr_update_direction_query_js_zero': query_direction['zero'],
            'actor/eitr_update_direction_query_js_plus': query_direction['plus'],
            'actor/eitr_update_direction_query_minus_margin_min': query_direction['minus_margin_min'],
            'actor/eitr_update_direction_query_plus_margin_min': query_direction['plus_margin_min'],
            'actor/eitr_update_direction_query_noise_max': query_direction['noise_max'],
            'actor/eitr_update_direction_query_resolved_energy_min': query_direction['resolved_grad_energy_min'],
            'actor/eitr_update_direction_query_target_delta_max': query_direction['target_logp_delta_max'],
            'actor/eitr_update_direction_query_sample_count': query_direction['sample_count'],
            'actor/eitr_update_direction_query_pass': query_direction['pass'],
        })
        reports = {}
        try:
            for direction, scale in (('minus', -epsilon), ('plus', epsilon)):
                self._restore_eitr_local_state(snapshot, restore_gradients=True)
                start_error = self._distributed_scalar(
                    self._local_snapshot_max_abs_error(snapshot),
                    distributed=distributed,
                    op=torch.distributed.ReduceOp.MAX,
                )
                if start_error != 0.0:
                    raise RuntimeError(
                        'Update-direction diagnostic could not restore theta_GRPO'
                    )
                self._apply_snapshot_gradient(snapshot, scale)
                grad_norm, update_norm, g_dot_delta, cosine = (
                    self._directional_parameter_statistics(
                        snapshot, direction=direction, distributed=distributed
                    )
                )
                if update_norm == 0.0:
                    raise RuntimeError('Update-direction diagnostic produced a zero parameter update')
                if tuple(
                    same_batch_cache_signature(mini_batch) for mini_batch in dataloader
                ) != cache_signature:
                    raise RuntimeError('Update-direction diagnostic mutated cached probe tensors')
                post_stats = self._score_cached_eitr_drift(
                    dataloader, temperature, distributed=distributed
                )
                if int(post_stats[1].item()) != int(global_active_state_count.item()):
                    raise RuntimeError(
                        'Update-direction diagnostic re-score did not cover every active state'
                    )
                d_post = rollout_averaged_env_drift(
                    post_stats[0].item(), global_rollout_state_count.item()
                )
                if not torch.isfinite(torch.tensor(d_post)):
                    raise RuntimeError('Update-direction diagnostic produced a non-finite drift')
                metrics.update({
                    f'actor/eitr_update_direction_d_{direction}': float(d_post),
                    f'actor/eitr_update_direction_d_delta_{direction}': float(d_post - d_zero),
                    f'actor/eitr_update_direction_start_max_abs_{direction}': float(start_error),
                    f'actor/eitr_update_direction_update_norm_{direction}': float(update_norm),
                    f'actor/eitr_update_direction_g_dot_delta_{direction}': float(g_dot_delta),
                    f'actor/eitr_update_direction_cos_{direction}': float(cosine),
                    f'actor/eitr_update_direction_grad_norm_{direction}': float(grad_norm),
                })
                reports[direction] = (d_post, update_norm, g_dot_delta, cosine, start_error)
        finally:
            self._restore_eitr_local_state(snapshot, restore_gradients=False)
            self.eitr_optimizer.zero_grad()

        restore_error = self._distributed_scalar(
            self._local_snapshot_max_abs_error(snapshot),
            distributed=distributed,
            op=torch.distributed.ReduceOp.MAX,
        )
        if restore_error != 0.0:
            raise RuntimeError('Update-direction diagnostic failed to restore theta_GRPO')
        metrics['actor/eitr_update_direction_final_restore_max_abs'] = restore_error
        if not distributed or torch.distributed.get_rank() == 0:
            print(
                'EITR_UPDATE_DIRECTION_DIAGNOSTIC '
                f'D_zero={d_zero:.12e} '
                f'D_minus={reports["minus"][0]:.12e} '
                f'D_plus={reports["plus"][0]:.12e} '
                f'g_dot_delta_minus={reports["minus"][2]:.12e} '
                f'g_dot_delta_plus={reports["plus"][2]:.12e} '
                f'cos_minus={reports["minus"][3]:.12e} '
                f'cos_plus={reports["plus"][3]:.12e} '
                f'query_minus={query_direction["minus"]:.12e} '
                f'query_zero={query_direction["zero"]:.12e} '
                f'query_plus={query_direction["plus"]:.12e} '
                f'query_pass={int(query_direction["pass"])}'
            )
        return metrics

    def _run_score_path_noop_direction_audit(
        self,
        dataloader,
        temperature,
        *,
        distributed,
        eitr_world_size,
        global_active_state_count,
        global_rollout_state_count,
        d_old,
        old_log_ratio_abs_max,
        theta_old_checksum,
        grpo_step_count,
        actor_lr,
        epsilon=3e-5,
    ):
        """Audit score consistency and the local +/- EITR update direction.

        The persistent policy is restored to theta_GRPO at exit.  The single
        SGD call is diagnostic-only: it proves the exact correction placement
        after all GRPO updates without creating a checkpoint or altering the
        normal training path.
        """
        cache_hash = cached_probe_fingerprint(dataloader)
        hashes = {'old': cache_hash}
        event_sequence = score_path_audit_event_sequence(grpo_step_count)
        if not distributed or torch.distributed.get_rank() == 0:
            print(f'EITR_SCORE_AUDIT_EVENT {event_sequence[-3]}')

        self.eitr_optimizer.zero_grad()
        zero_stats = self._audit_score_cached_eitr_drift(
            dataloader,
            temperature,
            distributed=distributed,
            global_rollout_state_count=global_rollout_state_count,
            eitr_world_size=eitr_world_size,
            build_grad=True,
            capture_query_sample=True,
        )
        if zero_stats['state_count'] != int(global_active_state_count.item()):
            raise RuntimeError('Score-path audit D_zero did not cover every active state')
        hashes['zero_grad'] = cached_probe_fingerprint(dataloader)
        snapshot = self._snapshot_eitr_local_state()
        theta_grpo_checksum = self._snapshot_checksum(snapshot, distributed=distributed)
        metrics = {
            'actor/eitr_score_path_noop_direction_audit': 1.0,
            'actor/eitr_score_path_d_old': float(d_old),
            'actor/eitr_score_path_old_log_ratio_abs_max': float(
                old_log_ratio_abs_max
            ),
            'actor/eitr_score_path_anchor_threshold': float(
                self.eitr_score_path_anchor_max_abs_logprob_diff
            ),
            'actor/eitr_score_path_anchor_pass': float(
                old_log_ratio_abs_max
                <= self.eitr_score_path_anchor_max_abs_logprob_diff
            ),
            'actor/eitr_score_path_d_zero_grad': float(zero_stats['drift']),
            'actor/eitr_score_path_epsilon': float(epsilon),
            'actor/eitr_score_path_state_count': float(zero_stats['state_count']),
            'actor/eitr_score_path_probe_count': float(zero_stats['probe_count']),
            'actor/eitr_score_path_coverage': float(
                global_active_state_count.item()
                / max(global_rollout_state_count.item(), 1.0)
            ),
            'actor/eitr_score_path_ess': float(zero_stats['ess']),
            'actor/eitr_score_path_log_ratio_clipfrac': float(zero_stats['clipfrac']),
            'actor/eitr_score_path_log_ratio_abs_max': float(
                zero_stats['log_ratio_abs_max']
            ),
            'actor/eitr_score_path_theta_old_checksum_sum': theta_old_checksum[0],
            'actor/eitr_score_path_theta_grpo_checksum_sum': theta_grpo_checksum[0],
            'actor/eitr_score_path_actor_lr': float(actor_lr),
        }
        noop_values = []
        noop_param_error = 0.0
        for index in range(3):
            self._restore_eitr_local_state(snapshot, restore_gradients=True)
            noop_stats = self._audit_score_cached_eitr_drift(
                dataloader,
                temperature,
                distributed=distributed,
                global_rollout_state_count=global_rollout_state_count,
                eitr_world_size=eitr_world_size,
                build_grad=False,
            )
            if noop_stats['state_count'] != int(global_active_state_count.item()):
                raise RuntimeError('Score-path audit no-op did not cover every active state')
            noop_values.append(float(noop_stats['drift']))
            hashes[f'noop_{index + 1}'] = cached_probe_fingerprint(dataloader)
            noop_param_error = max(
                noop_param_error,
                self._distributed_scalar(
                    self._local_snapshot_max_abs_error(snapshot),
                    distributed=distributed,
                    op=torch.distributed.ReduceOp.MAX,
                ),
            )

        noop_mean = sum(noop_values) / len(noop_values)
        noop_jitter = max(noop_values) - min(noop_values)
        metrics.update({
            'actor/eitr_score_path_d_noop_1': noop_values[0],
            'actor/eitr_score_path_d_noop_2': noop_values[1],
            'actor/eitr_score_path_d_noop_3': noop_values[2],
            'actor/eitr_score_path_noop_mean': noop_mean,
            'actor/eitr_score_path_noop_jitter': noop_jitter,
            'actor/eitr_score_path_zero_grad_minus_noop_mean': zero_stats['drift'] - noop_mean,
            'actor/eitr_score_path_noop_param_max_abs': noop_param_error,
        })
        if noop_param_error != 0.0:
            raise RuntimeError('Score-path audit no-op changed theta_GRPO')

        zero_noop_error = abs(zero_stats['drift'] - noop_mean)
        anchor_pass = bool(
            old_log_ratio_abs_max
            <= self.eitr_score_path_anchor_max_abs_logprob_diff
        )
        proposal_signal = proposal_signal_diagnostics(
            d_old=d_old,
            d_pre=noop_mean,
            noop_jitter=noop_jitter,
            zero_noop_abs_error=zero_noop_error,
            actor_lr=actor_lr,
        )
        metrics.update({
            'actor/eitr_score_path_zero_noop_abs_error': zero_noop_error,
            'actor/eitr_score_path_score_mode_tolerance': proposal_signal[
                'score_mode_tolerance'
            ],
            'actor/eitr_score_path_score_mode_pass': proposal_signal[
                'score_mode_pass'
            ],
            'actor/eitr_score_path_proposal_signal': proposal_signal['signal'],
            'actor/eitr_score_path_proposal_floor': proposal_signal[
                'numerical_floor'
            ],
            'actor/eitr_score_path_proposal_required_signal': proposal_signal[
                'required_signal'
            ],
            'actor/eitr_score_path_proposal_signal_to_floor': proposal_signal[
                'signal_to_floor_ratio'
            ],
            'actor/eitr_score_path_actor_lr_positive': proposal_signal[
                'actor_lr_positive'
            ],
            'actor/eitr_score_path_proposal_signal_pass': proposal_signal['pass'],
            'actor/eitr_score_path_no_signal': proposal_signal['no_signal'],
        })
        integrity_fail = bool(
            not anchor_pass or proposal_signal['score_mode_pass'] < 0.5
        )
        no_signal = bool(not integrity_fail and proposal_signal['no_signal'] > 0.5)
        metrics['actor/eitr_score_path_no_signal'] = float(no_signal)
        if integrity_fail or no_signal:
            # Do not manufacture a descent direction at the exact JS minimum.
            # Return a fully auditable terminal outcome without calling the
            # diagnostic optimizer or counting a persistent correction step.
            self._restore_eitr_local_state(snapshot, restore_gradients=False)
            self.eitr_optimizer.zero_grad()
            restore_error = self._distributed_scalar(
                self._local_snapshot_max_abs_error(snapshot),
                distributed=distributed,
                op=torch.distributed.ReduceOp.MAX,
            )
            theta_after_checksum = self._snapshot_checksum(
                snapshot, distributed=distributed
            )
            if restore_error != 0.0:
                raise RuntimeError('Score-path terminal audit failed to restore theta_GRPO')
            if theta_after_checksum != theta_grpo_checksum:
                raise RuntimeError('Score-path terminal audit checksum changed')
            probe_hash_mismatch = self._distributed_scalar(
                float(len(set(hashes.values())) != 1),
                distributed=distributed,
                op=torch.distributed.ReduceOp.MAX,
            )
            if probe_hash_mismatch > 0:
                raise RuntimeError(
                    f'Score-path terminal cached probe fingerprint mismatch: {hashes}'
                )
            metrics.update({
                'actor/eitr_score_path_direction_evaluated': 0.0,
                'actor/eitr_score_path_candidate_sgd_ran': 0.0,
                'actor/eitr_score_path_query_pass': 0.0,
                'actor/eitr_score_path_parameter_pass': 0.0,
                'actor/eitr_score_path_parameter_minus_descent_pass': 0.0,
                'actor/eitr_score_path_parameter_update_direction_pass': 0.0,
                'actor/eitr_score_path_parameter_bidirectional_pass': 0.0,
                'actor/eitr_score_path_parameter_plus_ascent_pass': 0.0,
                'actor/eitr_score_path_parameter_nonsmooth_warning': 0.0,
                'actor/eitr_score_path_audit_inconclusive': float(no_signal),
                'actor/eitr_score_path_audit_fail': float(integrity_fail),
                'actor/eitr_score_path_audit_pass': 0.0,
                'actor/eitr_score_path_final_restore_max_abs': restore_error,
                'actor/eitr_score_path_theta_after_checksum_sum': (
                    theta_after_checksum[0]
                ),
                'actor/eitr_score_path_probe_hash_match': 1.0,
            })
            outcome = 'NO_SIGNAL' if no_signal else 'INTEGRITY_FAIL'
            if not distributed or torch.distributed.get_rank() == 0:
                print(
                    f'EITR_SCORE_PATH_AUDIT_{outcome} '
                    f'actor_lr={actor_lr:.12e} '
                    f'D_old={d_old:.12e} '
                    f'D_pre={noop_mean:.12e} '
                    f'floor={proposal_signal["numerical_floor"]:.12e} '
                    f'signal={proposal_signal["signal"]:.12e} '
                    f'snr={proposal_signal["signal_to_floor_ratio"]:.12e} '
                    f'anchor_log_ratio_abs_max={old_log_ratio_abs_max:.12e} '
                    f'post_grpo_log_ratio_abs_max={zero_stats["log_ratio_abs_max"]:.12e} '
                    f'anchor_pass={int(anchor_pass)} '
                    f'score_mode_pass={int(proposal_signal["score_mode_pass"])} '
                    f'probe_hash={cache_hash} '
                    f'final_restore={restore_error:.12e}',
                    flush=True,
                )
                print(
                    f'EITR_SCORE_AUDIT_EVENT {outcome}@v{grpo_step_count}',
                    flush=True,
                )
            return metrics

        query_direction = self._query_logprob_direction_metrics(
            zero_stats['query_sample'], distributed=distributed
        )
        metrics.update({
            'actor/eitr_score_path_query_js_minus': query_direction['minus'],
            'actor/eitr_score_path_query_js_zero': query_direction['zero'],
            'actor/eitr_score_path_query_js_plus': query_direction['plus'],
            'actor/eitr_score_path_query_minus_margin_min': query_direction['minus_margin_min'],
            'actor/eitr_score_path_query_plus_margin_min': query_direction['plus_margin_min'],
            'actor/eitr_score_path_query_noise_max': query_direction['noise_max'],
            'actor/eitr_score_path_query_resolved_energy_min': query_direction['resolved_grad_energy_min'],
            'actor/eitr_score_path_query_target_delta_max': query_direction['target_logp_delta_max'],
            'actor/eitr_score_path_query_grad_norm_sum': query_direction['grad_norm_sum'],
            'actor/eitr_score_path_query_sample_count': query_direction['sample_count'],
            'actor/eitr_score_path_query_pass': query_direction['pass'],
        })

        original_lrs = [group['lr'] for group in self.eitr_optimizer.param_groups]
        reports = {}
        try:
            # The minus candidate is the one true independent SGD step. It is
            # performed once, after full-batch D_pre/gradient accumulation, and
            # then restored so this audit cannot perturb training.
            self._restore_eitr_local_state(snapshot, restore_gradients=True)
            start_minus = self._distributed_scalar(
                self._local_snapshot_max_abs_error(snapshot),
                distributed=distributed,
                op=torch.distributed.ReduceOp.MAX,
            )
            if start_minus != 0.0:
                raise RuntimeError('Score-path audit minus did not start from theta_GRPO')
            for group in self.eitr_optimizer.param_groups:
                group['lr'] = float(epsilon)
            self.eitr_optimizer.step()
            minus_stats = self._audit_direction_statistics(
                snapshot, direction='minus', distributed=distributed
            )
            if minus_stats['delta_norm'] == 0.0:
                raise RuntimeError('Score-path audit minus update was zero')
            hashes['minus'] = cached_probe_fingerprint(dataloader)
            if not distributed or torch.distributed.get_rank() == 0:
                print(f'EITR_SCORE_AUDIT_EVENT {event_sequence[-2]}')
            minus_score = self._audit_score_cached_eitr_drift(
                dataloader,
                temperature,
                distributed=distributed,
                global_rollout_state_count=global_rollout_state_count,
                eitr_world_size=eitr_world_size,
                build_grad=False,
            )
            if minus_score['state_count'] != int(global_active_state_count.item()):
                raise RuntimeError('Score-path audit minus did not cover every active state')
            reports['minus'] = (minus_score['drift'], minus_stats, start_minus)

            self._restore_eitr_local_state(snapshot, restore_gradients=True)
            start_plus = self._distributed_scalar(
                self._local_snapshot_max_abs_error(snapshot),
                distributed=distributed,
                op=torch.distributed.ReduceOp.MAX,
            )
            if start_plus != 0.0:
                raise RuntimeError('Score-path audit plus did not start from theta_GRPO')
            self._apply_snapshot_gradient(snapshot, epsilon)
            plus_stats = self._audit_direction_statistics(
                snapshot, direction='plus', distributed=distributed
            )
            if plus_stats['delta_norm'] == 0.0:
                raise RuntimeError('Score-path audit plus update was zero')
            hashes['plus'] = cached_probe_fingerprint(dataloader)
            plus_score = self._audit_score_cached_eitr_drift(
                dataloader,
                temperature,
                distributed=distributed,
                global_rollout_state_count=global_rollout_state_count,
                eitr_world_size=eitr_world_size,
                build_grad=False,
            )
            if plus_score['state_count'] != int(global_active_state_count.item()):
                raise RuntimeError('Score-path audit plus did not cover every active state')
            reports['plus'] = (plus_score['drift'], plus_stats, start_plus)
        finally:
            self._restore_eitr_local_state(snapshot, restore_gradients=False)
            self.eitr_optimizer.zero_grad()
            for group, original_lr in zip(self.eitr_optimizer.param_groups, original_lrs):
                group['lr'] = original_lr

        restore_error = self._distributed_scalar(
            self._local_snapshot_max_abs_error(snapshot),
            distributed=distributed,
            op=torch.distributed.ReduceOp.MAX,
        )
        theta_after_checksum = self._snapshot_checksum(snapshot, distributed=distributed)
        if restore_error != 0.0:
            raise RuntimeError('Score-path audit failed to restore theta_GRPO')
        if theta_after_checksum != theta_grpo_checksum:
            raise RuntimeError('Score-path audit checksum changed after final restoration')
        probe_hash_mismatch = self._distributed_scalar(
            float(len(set(hashes.values())) != 1),
            distributed=distributed,
            op=torch.distributed.ReduceOp.MAX,
        )
        if probe_hash_mismatch > 0:
            raise RuntimeError(f'Score-path audit cached probe fingerprint mismatch: {hashes}')

        for direction, (drift, stats, start_error) in reports.items():
            metrics.update({
                f'actor/eitr_score_path_d_{direction}': float(drift),
                f'actor/eitr_score_path_d_delta_{direction}_vs_noop': float(drift - noop_mean),
                f'actor/eitr_score_path_start_max_abs_{direction}': float(start_error),
                f'actor/eitr_score_path_update_norm_{direction}': stats['delta_norm'],
                f'actor/eitr_score_path_g_dot_delta_{direction}': stats['g_dot_delta'],
                f'actor/eitr_score_path_cos_{direction}': stats['cosine'],
                f'actor/eitr_score_path_changed_fraction_{direction}': stats['changed_fraction'],
                f'actor/eitr_score_path_delta_abs_shard_p50_max_{direction}': (
                    stats['delta_abs_shard_p50_max']
                ),
                f'actor/eitr_score_path_delta_abs_shard_p95_max_{direction}': (
                    stats['delta_abs_shard_p95_max']
                ),
                f'actor/eitr_score_path_delta_abs_shard_p99_max_{direction}': (
                    stats['delta_abs_shard_p99_max']
                ),
                f'actor/eitr_score_path_delta_abs_max_{direction}': (
                    stats['delta_abs_max']
                ),
                f'actor/eitr_score_path_quantile_sample_count_{direction}': (
                    stats['quantile_sample_count']
                ),
                f'actor/eitr_score_path_quantile_sample_stride_max_{direction}': (
                    stats['quantile_sample_stride_max']
                ),
                f'actor/eitr_score_path_grad_norm_{direction}': stats['grad_norm'],
            })
        parameter_direction = parameter_correction_diagnostics(
            minus=reports['minus'][0],
            zero=noop_mean,
            plus=reports['plus'][0],
            minus_g_dot_delta=reports['minus'][1]['g_dot_delta'],
            minus_cosine=reports['minus'][1]['cosine'],
            jitter=noop_jitter,
        )
        # Gate C validates the correction the algorithm actually applies:
        # theta <- theta - lr * grad(D_env).  The opposite candidate remains a
        # useful smoothness diagnostic, but FSDP/BF16 parameter re-scoring is
        # piecewise quantized and need not be locally symmetric at a finite
        # epsilon.  Query-logprob space remains the strict bidirectional sign
        # check; parameter space requires a real, correctly directed -g step
        # that lowers the same cached-batch drift beyond numerical noise.
        parameter_update_direction_pass = bool(
            parameter_direction['update_direction_pass'] > 0.5
        )
        parameter_correction_pass = bool(
            parameter_direction['correction_pass'] > 0.5
        )
        parameter_bidirectional_pass = bool(
            parameter_direction['bidirectional_pass'] > 0.5
        )
        score_path_pass = bool(proposal_signal['score_mode_pass'] > 0.5)
        audit_pass = bool(
            anchor_pass
            and score_path_pass
            and query_direction['pass'] > 0.5
            and parameter_correction_pass
        )
        metrics.update({
            'actor/eitr_score_path_parameter_minus_margin': parameter_direction['minus_margin'],
            'actor/eitr_score_path_parameter_plus_margin': parameter_direction['plus_margin'],
            'actor/eitr_score_path_parameter_noise': parameter_direction['noise'],
            'actor/eitr_score_path_parameter_minus_descent_pass': (
                parameter_direction['minus_pass']
            ),
            'actor/eitr_score_path_parameter_update_direction_pass': float(
                parameter_update_direction_pass
            ),
            'actor/eitr_score_path_parameter_bidirectional_pass': float(
                parameter_bidirectional_pass
            ),
            'actor/eitr_score_path_parameter_plus_ascent_pass': (
                parameter_direction['plus_pass']
            ),
            'actor/eitr_score_path_parameter_nonsmooth_warning': float(
                parameter_direction['nonsmooth_warning']
            ),
            'actor/eitr_score_path_parameter_pass': float(
                parameter_correction_pass
            ),
            'actor/eitr_score_path_zero_noop_abs_error': zero_noop_error,
            'actor/eitr_score_path_score_mode_pass': float(score_path_pass),
            'actor/eitr_score_path_direction_evaluated': 1.0,
            'actor/eitr_score_path_candidate_sgd_ran': 1.0,
            'actor/eitr_score_path_audit_inconclusive': 0.0,
            'actor/eitr_score_path_audit_fail': float(not audit_pass),
            'actor/eitr_score_path_audit_pass': float(audit_pass),
        })
        metrics['actor/eitr_score_path_final_restore_max_abs'] = restore_error
        metrics['actor/eitr_score_path_theta_after_checksum_sum'] = theta_after_checksum[0]
        metrics['actor/eitr_score_path_probe_hash_match'] = 1.0
        if not distributed or torch.distributed.get_rank() == 0:
            print(
                'EITR_SCORE_PATH_AUDIT '
                f'actor_lr={actor_lr:.12e} '
                f'D_old={d_old:.12e} '
                f'D_zero_grad={zero_stats["drift"]:.12e} '
                f'D_noop_1={noop_values[0]:.12e} '
                f'D_noop_2={noop_values[1]:.12e} '
                f'D_noop_3={noop_values[2]:.12e} '
                f'D_minus={reports["minus"][0]:.12e} '
                f'D_plus={reports["plus"][0]:.12e} '
                f'proposal_signal={proposal_signal["signal"]:.12e} '
                f'proposal_snr={proposal_signal["signal_to_floor_ratio"]:.12e} '
                f'anchor_log_ratio_abs_max={old_log_ratio_abs_max:.12e} '
                f'post_grpo_log_ratio_abs_max={zero_stats["log_ratio_abs_max"]:.12e} '
                f'query_minus={query_direction["minus"]:.12e} '
                f'query_zero={query_direction["zero"]:.12e} '
                f'query_plus={query_direction["plus"]:.12e} '
                f'query_pass={int(query_direction["pass"])} '
                f'parameter_pass={int(parameter_correction_pass)} '
                f'parameter_bidirectional_pass={int(parameter_bidirectional_pass)} '
                f'parameter_plus_ascent_pass={int(parameter_direction["plus_pass"])} '
                f'anchor_pass={int(anchor_pass)} '
                f'audit_pass={int(audit_pass)} '
                f'probe_hash={cache_hash} '
                f'final_restore={restore_error:.12e}',
                flush=True,
            )
            print(f'EITR_SCORE_AUDIT_EVENT {event_sequence[-1]}', flush=True)
            print(
                'EITR_SCORE_AUDIT_SEQUENCE ' + ' -> '.join(event_sequence),
                flush=True,
            )
        return metrics

    def update_policy(self, data: DataProto):
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # avoid silently training at the wrong temperature
        actor_lr_used = float(self.actor_optimizer.param_groups[0]['lr'])

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.state_masking:
            select_keys.append('loss_mask')
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        grpo_select_keys = tuple(select_keys)
        if self.eitr_uses_probes:
            select_keys.extend(EITR_BATCH_KEYS)
        batch = data.select(batch_keys=select_keys).batch

        # PPO epochs retain their ordinary meaning: complete GRPO passes over
        # the rollout batch. Conditional EITR corrections run only after every
        # configured GRPO epoch has finished.
        dataloader = list(batch.split(self.config.ppo_mini_batch_size))
        metrics = {}
        if self.eitr_uses_probes:
            append_to_dict(metrics, {
                'actor/eitr_probe_gradient_checkpointing_active': float(
                    self.eitr_probe_gradient_checkpointing_active
                ),
            })
        total_pass_count = self.ppo_epochs + (
            self.eitr_correction_passes if self.eitr_uses_probes else 0
        )
        pass_stats = [
            {
                'js_sum': 0.0,
                'state_count': 0.0,
                'probe_count': 0.0,
                'ess_sum': 0.0,
                'clipfrac_sum': 0.0,
                'raw_grad_sq_sum': 0.0,
                'applied_grad_sq_sum': 0.0,
                'active_micro_batch_count': 0.0,
                'log_ratio_abs_max': 0.0,
            }
            for _ in range(total_pass_count)
        ]
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        eitr_world_size = torch.distributed.get_world_size() if distributed else 1
        grpo_optimizer_step_count = 0
        eitr_correction_optimizer_step_count = 0
        audit_d_old = None
        audit_theta_old_checksum = None
        if self.eitr_score_path_noop_direction_audit:
            audit_global_active_state_count = torch.tensor(
                float(batch['eitr_state_valid'].sum().item()),
                dtype=torch.float64,
                device=torch.cuda.current_device(),
            )
            audit_global_rollout_state_count = torch.tensor(
                float(batch['eitr_state_valid'].numel()),
                dtype=torch.float64,
                device=torch.cuda.current_device(),
            )
            if distributed:
                torch.distributed.all_reduce(
                    audit_global_active_state_count, op=torch.distributed.ReduceOp.SUM
                )
                torch.distributed.all_reduce(
                    audit_global_rollout_state_count, op=torch.distributed.ReduceOp.SUM
                )
            if audit_global_active_state_count.item() <= 0:
                raise RuntimeError('Score-path audit found no active EITR states before GRPO')
            audit_old_snapshot = self._snapshot_eitr_local_state()
            audit_theta_old_checksum = self._snapshot_checksum(
                audit_old_snapshot, distributed=distributed
            )
            del audit_old_snapshot
            if not distributed or torch.distributed.get_rank() == 0:
                print('EITR_SCORE_AUDIT_EVENT probe_old_logp@v0')
            audit_old_stats = self._audit_score_cached_eitr_drift(
                dataloader,
                temperature,
                distributed=distributed,
                global_rollout_state_count=audit_global_rollout_state_count,
                eitr_world_size=eitr_world_size,
                build_grad=False,
            )
            if audit_old_stats['state_count'] != int(audit_global_active_state_count.item()):
                raise RuntimeError('Score-path audit D_old did not cover every active state')
            audit_d_old = audit_old_stats['drift']

        torch.cuda.synchronize()
        grpo_update_start = time.perf_counter()
        for ppo_epoch in range(self.ppo_epochs):
            for mini_batch in dataloader:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    # Formal runs keep the full rollout/probe cache on CPU.
                    # Stream only tensors consumed by the current GRPO forward;
                    # cached K-probe tensors are loaded later, one state chunk
                    # at a time, by ``_eitr_chunk_to_cuda``.
                    micro_batch = micro_batch.select(*grpo_select_keys).cuda()
                    responses = micro_batch['responses']
                    response_length = responses.size(1)
                    attention_mask = micro_batch['attention_mask']
                    response_mask = attention_mask[:, -response_length:]
                    if self.config.state_masking:
                        response_mask = micro_batch['loss_mask']
                    old_log_prob = micro_batch['old_log_probs']
                    advantages = micro_batch['advantages']

                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=micro_batch,
                        temperature=temperature,
                    )
                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        eos_mask=response_mask,
                        cliprange=self.config.clip_ratio,
                    )
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)
                    policy_loss = pg_loss - entropy_loss * self.config.entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = micro_batch['ref_log_prob']
                        kld = core_algos.kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = masked_mean(kld, response_mask)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                    loss = policy_loss / self.gradient_accumulation
                    loss.backward()
                    append_to_dict(metrics, {
                        'actor/entropy_loss': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                    })

                grad_norm = self._optimizer_step()
                grpo_optimizer_step_count += 1
                if self.eitr_score_path_noop_direction_audit and (
                    not distributed or torch.distributed.get_rank() == 0
                ):
                    print(f'EITR_SCORE_AUDIT_EVENT GRPO_STEP_{grpo_optimizer_step_count}')
                append_to_dict(metrics, {
                    'actor/grad_norm': grad_norm.detach().item(),
                    'actor/ppo_epoch': float(ppo_epoch),
                    f'actor/grad_norm_pass_{ppo_epoch}': grad_norm.detach().item(),
                })

        torch.cuda.synchronize()
        append_to_dict(metrics, {
            'timing_s/grpo_update': float(
                time.perf_counter() - grpo_update_start
            ),
        })

        optimizer_offload_seconds = 0.0
        optimizer_offloaded_for_eitr = False
        if self.eitr_uses_probes:
            # GRPO has already consumed these gradients. Release them before
            # moving AdamW moments so the transition itself has maximum room.
            self.actor_optimizer.zero_grad()
        if self.eitr_uses_probes and self.offload_actor_optimizer_after_grpo:
            # AdamW moments are not consumed by the independent stateless EITR
            # SGD.  Keep them on CPU from the end of all GRPO passes until the
            # next outer update loads them again in the FSDP worker.
            torch.cuda.synchronize()
            offload_start = time.perf_counter()
            offload_fsdp_optimizer(self.actor_optimizer)
            torch.cuda.synchronize()
            optimizer_offload_seconds = time.perf_counter() - offload_start
            optimizer_offloaded_for_eitr = True
        append_to_dict(metrics, {
            'actor/optimizer_state_offloaded_for_eitr': float(
                optimizer_offloaded_for_eitr
            ),
            'timing_s/actor_optimizer_offload_after_grpo': float(
                optimizer_offload_seconds
            ),
        })

        # Conditional correction is deliberately not another GRPO epoch. This
        # avoids giving EITR/Probe-GRPO an extra reward-bearing policy update.
        # With zero active states the whole section is skipped, so parameters
        # and optimizer state exactly match ordinary GRPO.
        post_drift_stat_tensor = None
        post_diagnostic_ran = False
        same_batch_scale_diagnostic_ran = False
        update_direction_diagnostic_ran = False
        score_path_noop_direction_audit_ran = False
        query_logprob_direction_sample = None
        eitr_effective_correction_lr = 0.0
        if self.eitr_uses_probes:
            torch.cuda.synchronize()
            eitr_correction_start = time.perf_counter()
            self.actor_optimizer.zero_grad()
            global_active_state_count = torch.tensor(
                float(batch['eitr_state_valid'].sum().item()),
                dtype=torch.float64,
                device=torch.cuda.current_device(),
            )
            if distributed:
                torch.distributed.all_reduce(
                    global_active_state_count,
                    op=torch.distributed.ReduceOp.SUM,
                )
            global_rollout_state_count = torch.tensor(
                float(batch['eitr_state_valid'].numel()),
                dtype=torch.float64,
                device=torch.cuda.current_device(),
            )
            if distributed:
                torch.distributed.all_reduce(
                    global_rollout_state_count,
                    op=torch.distributed.ReduceOp.SUM,
                )

            if self.eitr_score_path_noop_direction_audit:
                if global_active_state_count.item() <= 0:
                    raise RuntimeError('Score-path audit found no valid EITR states')
                if int(global_active_state_count.item()) != int(
                    audit_global_active_state_count.item()
                ):
                    raise RuntimeError('Score-path audit active-state count changed during GRPO')
                diagnostic_metrics = self._run_score_path_noop_direction_audit(
                    dataloader,
                    temperature,
                    distributed=distributed,
                    eitr_world_size=eitr_world_size,
                    global_active_state_count=global_active_state_count,
                    global_rollout_state_count=global_rollout_state_count,
                    d_old=audit_d_old,
                    old_log_ratio_abs_max=audit_old_stats['log_ratio_abs_max'],
                    theta_old_checksum=audit_theta_old_checksum,
                    grpo_step_count=grpo_optimizer_step_count,
                    actor_lr=actor_lr_used,
                )
                append_to_dict(metrics, diagnostic_metrics)
                pass_index = self.ppo_epochs
                audit_additive_stats = rank_owned_global_additive_stats({
                    'js_sum': diagnostic_metrics['actor/eitr_score_path_d_zero_grad']
                    * global_rollout_state_count.item(),
                    'state_count': global_active_state_count.item(),
                    'probe_count': diagnostic_metrics['actor/eitr_score_path_probe_count'],
                    'ess_sum': diagnostic_metrics['actor/eitr_score_path_ess']
                    * global_active_state_count.item(),
                    'clipfrac_sum': diagnostic_metrics[
                        'actor/eitr_score_path_log_ratio_clipfrac'
                    ] * global_active_state_count.item(),
                    'active_micro_batch_count': float(len(dataloader)),
                }, distributed=distributed, rank=(
                    torch.distributed.get_rank() if distributed else 0
                ))
                pass_stats[pass_index].update(audit_additive_stats)
                # Count only a real candidate SGD invocation. A zero-proposal
                # audit exits before the optimizer and must remain loss_applied=0.
                if diagnostic_metrics.get(
                    'actor/eitr_score_path_candidate_sgd_ran', 0.0
                ) > 0.5:
                    eitr_correction_optimizer_step_count += 1
                score_path_noop_direction_audit_ran = True
            elif global_active_state_count.item() > 0:
                correction_state_chunks, compaction_metrics = (
                    self._prepare_eitr_correction_state_chunks(
                        dataloader, distributed=distributed
                    )
                )
                append_to_dict(metrics, compaction_metrics)
                for correction_index in range(self.eitr_correction_passes):
                    pass_index = self.ppo_epochs + correction_index
                    probe_forward_enabled = eitr_probe_enabled_for_pass(
                        self.eitr_config,
                        pass_index,
                        grpo_passes=self.ppo_epochs,
                    )
                    eitr_loss_enabled = eitr_loss_enabled_for_pass(
                        self.eitr_config,
                        pass_index,
                        grpo_passes=self.ppo_epochs,
                    )
                    if not probe_forward_enabled:
                        continue

                    if eitr_loss_enabled:
                        self.eitr_optimizer.zero_grad()
                    local_applied_grad_sq = 0.0

                    for state_chunk in correction_state_chunks:
                        state_chunk = self._eitr_chunk_to_cuda(state_chunk)
                        eitr_result = self._compute_eitr_micro_batch(
                            state_chunk,
                            temperature,
                            track_model_grad=eitr_loss_enabled,
                        )
                        if eitr_result is None:
                            continue

                        valid_state_count = eitr_result['valid_state_count']
                        # Every chunk is normalized by the same full rollout
                        # denominator. Gradients accumulate over all state
                        # chunks before the single V6 correction step.
                        state_weight = coverage_weighted_state_scale(
                            valid_state_count,
                            float(global_rollout_state_count.item()),
                            world_size=eitr_world_size,
                        )
                        raw_logprob_grad = torch.autograd.grad(
                            eitr_result['loss'],
                            eitr_result['current_seq_logp'],
                            retain_graph=eitr_loss_enabled,
                            allow_unused=True,
                        )[0]
                        raw_grad_sq = (
                            float(raw_logprob_grad.detach().float().square().sum().item())
                            if raw_logprob_grad is not None
                            else 0.0
                        )
                        if (
                            self.eitr_update_direction_diagnostic
                            and query_logprob_direction_sample is None
                            and raw_logprob_grad is not None
                            and valid_state_count > 0
                        ):
                            state_slot = state_chunk['eitr_state_slot'].bool()
                            state_valid = state_chunk['eitr_state_valid'][state_slot].bool()
                            if state_valid.any():
                                query_logprob_direction_sample = tuple(
                                    value.detach().to(device='cpu', copy=True)
                                    for value in (
                                        eitr_result['current_seq_logp'][state_valid],
                                        raw_logprob_grad[state_valid],
                                        state_chunk['eitr_probe_old_seq_logp'][state_slot][state_valid],
                                        state_chunk['eitr_probe_doc_probs'][state_slot][state_valid],
                                        state_chunk['eitr_probe_valid'][state_slot][state_valid],
                                    )
                                )
                        applied_scale = (
                            self.eitr_lambda_env * state_weight
                            if eitr_loss_enabled and valid_state_count > 0
                            else 0.0
                        )
                        applied_grad_sq = raw_grad_sq * applied_scale * applied_scale
                        local_applied_grad_sq += applied_grad_sq

                        stats = pass_stats[pass_index]
                        stats['raw_grad_sq_sum'] += raw_grad_sq
                        stats['applied_grad_sq_sum'] += applied_grad_sq
                        if valid_state_count > 0:
                            stats['js_sum'] += eitr_result['js_sum']
                            stats['state_count'] += valid_state_count
                            stats['probe_count'] += eitr_result['valid_probe_count']
                            stats['ess_sum'] += eitr_result['ess_mean'] * valid_state_count
                            stats['clipfrac_sum'] += (
                                eitr_result['log_ratio_clipfrac'] * valid_state_count
                            )
                            stats['active_micro_batch_count'] += 1
                            stats['log_ratio_abs_max'] = max(
                                stats['log_ratio_abs_max'],
                                eitr_result['log_ratio_abs_max'],
                            )
                        append_to_dict(metrics, {
                            'actor/eitr_induced_js': eitr_result['js_mean'],
                            'actor/eitr_probe_ess': eitr_result['ess_mean'],
                            'actor/eitr_log_ratio_abs_max': eitr_result['log_ratio_abs_max'],
                            'actor/eitr_log_ratio_clipfrac': eitr_result['log_ratio_clipfrac'],
                            'actor/eitr_correction_pass': float(correction_index),
                            'actor/eitr_state_chunk_size': float(
                                state_chunk['eitr_state_slot'].size(0)
                            ),
                        })

                        if eitr_loss_enabled:
                            correction_loss = eitr_result['loss'] * applied_scale
                            correction_loss.backward()
                            del correction_loss
                        del raw_logprob_grad, eitr_result

                    if eitr_loss_enabled:
                        global_applied_grad_sq = torch.tensor(
                            local_applied_grad_sq,
                            dtype=torch.float64,
                            device=torch.cuda.current_device(),
                        )
                        if distributed:
                            torch.distributed.all_reduce(
                                global_applied_grad_sq,
                                op=torch.distributed.ReduceOp.SUM,
                            )
                        if global_applied_grad_sq.item() > 0:
                            pre_drift_stat_tensor = torch.tensor(
                                [
                                    pass_stats[pass_index]['js_sum'],
                                    pass_stats[pass_index]['state_count'],
                                ],
                                dtype=torch.float64,
                                device=torch.cuda.current_device(),
                            )
                            if distributed:
                                torch.distributed.all_reduce(
                                    pre_drift_stat_tensor,
                                    op=torch.distributed.ReduceOp.SUM,
                                )
                            if int(pre_drift_stat_tensor[1].item()) != int(
                                global_active_state_count.item()
                            ):
                                raise RuntimeError(
                                    'Same-batch diagnostic pre-score did not cover every active state'
                                )
                            if self.eitr_same_batch_scale_lrs:
                                diagnostic_metrics = self._run_same_batch_scale_diagnostic(
                                    dataloader,
                                    temperature,
                                    distributed=distributed,
                                    global_active_state_count=global_active_state_count,
                                    global_rollout_state_count=global_rollout_state_count,
                                    d_zero=rollout_averaged_env_drift(
                                        pre_drift_stat_tensor[0].item(),
                                        global_rollout_state_count.item(),
                                    ),
                                )
                                append_to_dict(metrics, diagnostic_metrics)
                                same_batch_scale_diagnostic_ran = True
                            elif self.eitr_update_direction_diagnostic:
                                diagnostic_metrics = self._run_update_direction_diagnostic(
                                    dataloader,
                                    temperature,
                                    distributed=distributed,
                                    global_active_state_count=global_active_state_count,
                                    global_rollout_state_count=global_rollout_state_count,
                                    d_zero=rollout_averaged_env_drift(
                                        pre_drift_stat_tensor[0].item(),
                                        global_rollout_state_count.item(),
                                    ),
                                    query_sample=query_logprob_direction_sample,
                                )
                                append_to_dict(metrics, diagnostic_metrics)
                                update_direction_diagnostic_ran = True
                            else:
                                correction_grad_norm, normalized_step = (
                                    self._normalized_eitr_optimizer_step()
                                )
                                eitr_effective_correction_lr = normalized_step[
                                    'effective_lr'
                                ]
                                eitr_correction_optimizer_step_count += 1
                                append_to_dict(metrics, {
                                    'actor/eitr_correction_grad_norm': correction_grad_norm.detach().item(),
                                    'actor/eitr_correction_normalization_scale': normalized_step['normalization_scale'],
                                    'actor/eitr_correction_effective_lr': normalized_step['effective_lr'],
                                    'actor/eitr_correction_unnormalized_update_norm': normalized_step['unnormalized_update_norm'],
                                    'actor/eitr_correction_predicted_update_norm': normalized_step['predicted_update_norm'],
                                    'actor/eitr_correction_max_update_norm': float(
                                        self.eitr_correction_max_update_norm or 0.0
                                    ),
                                    f'actor/grad_norm_pass_{pass_index}': correction_grad_norm.detach().item(),
                                })
                        elif (
                            self.eitr_same_batch_scale_lrs
                            or self.eitr_update_direction_diagnostic
                        ):
                            raise RuntimeError(
                                'Same-batch diagnostic found zero EITR gradient on the shared batch'
                            )
                        self.eitr_optimizer.zero_grad()
                        self.actor_optimizer.zero_grad()

                        outer_update_step = int(data.meta_info.get('outer_update_step', 0))
                        post_diagnostic_ran = should_run_post_diagnostic(
                            outer_update_step,
                            self.eitr_post_diagnostic_freq,
                            correction_applied=(
                                eitr_correction_optimizer_step_count > 0
                            ),
                        )
                        if post_diagnostic_ran:
                            post_drift_stat_tensor = self._score_cached_eitr_drift(
                                dataloader,
                                temperature,
                                distributed=distributed,
                                state_chunks=correction_state_chunks,
                            )
                            if int(post_drift_stat_tensor[1].item()) != int(
                                global_active_state_count.item()
                            ):
                                raise RuntimeError(
                                    'Post-correction EITR re-score did not cover the '
                                    'same active states as the pre-correction pass'
                                )
            elif (
                self.eitr_same_batch_scale_lrs
                or self.eitr_update_direction_diagnostic
            ):
                raise RuntimeError('Same-batch diagnostic found no valid EITR states')

            torch.cuda.synchronize()
            append_to_dict(metrics, {
                'timing_s/eitr_correction': float(
                    time.perf_counter() - eitr_correction_start
                ),
            })

        self.actor_optimizer.zero_grad()

        if self.eitr_uses_probes:
            # Sum additive statistics and separately max-reduce the ratio peak.
            pass_stat_tensor = torch.tensor(
                [
                    [
                        item['js_sum'],
                        item['state_count'],
                        item['probe_count'],
                        item['ess_sum'],
                        item['clipfrac_sum'],
                        item['raw_grad_sq_sum'],
                        item['applied_grad_sq_sum'],
                        item['active_micro_batch_count'],
                    ]
                    for item in pass_stats
                ],
                dtype=torch.float64,
                device=torch.cuda.current_device(),
            )
            ratio_max_tensor = torch.tensor(
                [item['log_ratio_abs_max'] for item in pass_stats],
                dtype=torch.float64,
                device=torch.cuda.current_device(),
            )
            if distributed:
                torch.distributed.all_reduce(pass_stat_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(ratio_max_tensor, op=torch.distributed.ReduceOp.MAX)

            for pass_index in range(total_pass_count):
                state_count = float(pass_stat_tensor[pass_index, 1].item())
                denominator = max(state_count, 1.0)
                append_to_dict(metrics, {
                    f'actor/eitr_pass_{pass_index}_active_state_count': state_count,
                    f'actor/eitr_pass_{pass_index}_valid_probe_count': float(
                        pass_stat_tensor[pass_index, 2].item()
                    ),
                    f'actor/eitr_pass_{pass_index}_induced_js': float(
                        pass_stat_tensor[pass_index, 0].item() / denominator
                    ),
                    f'actor/eitr_pass_{pass_index}_probe_ess': float(
                        pass_stat_tensor[pass_index, 3].item() / denominator
                    ),
                    f'actor/eitr_pass_{pass_index}_log_ratio_clipfrac': float(
                        pass_stat_tensor[pass_index, 4].item() / denominator
                    ),
                    f'actor/eitr_pass_{pass_index}_raw_logprob_grad_norm': float(
                        pass_stat_tensor[pass_index, 5].clamp_min(0).sqrt().item()
                    ),
                    f'actor/eitr_pass_{pass_index}_applied_logprob_grad_norm': float(
                        pass_stat_tensor[pass_index, 6].clamp_min(0).sqrt().item()
                    ),
                    f'actor/eitr_pass_{pass_index}_active_micro_batch_count': float(
                        pass_stat_tensor[pass_index, 7].item()
                    ),
                    f'actor/eitr_pass_{pass_index}_log_ratio_abs_max': float(
                        ratio_max_tensor[pass_index].item()
                    ),
                })

            final_probe_pass = total_pass_count - 1
            if int(pass_stat_tensor[final_probe_pass, 1].item()) != int(
                global_active_state_count.item()
            ):
                raise RuntimeError(
                    'Pre-correction EITR pass did not cover every active state'
                )
            final_state_count = max(float(pass_stat_tensor[final_probe_pass, 1].item()), 1.0)
            rollout_denominator = max(global_rollout_state_count.item(), 1.0)
            env_drift_pre = rollout_averaged_env_drift(
                pass_stat_tensor[final_probe_pass, 0].item(),
                rollout_denominator,
            )
            correction_lr = float(
                self.eitr_optimizer.param_groups[0]['lr']
                if self.eitr_optimizer is not None
                else 0.0
            )
            append_to_dict(metrics, {
                'actor/eitr_global_induced_js': float(
                    pass_stat_tensor[final_probe_pass, 0].item() / final_state_count
                ),
                'actor/eitr_global_probe_ess': float(
                    pass_stat_tensor[final_probe_pass, 3].item() / final_state_count
                ),
                'actor/eitr_active_state_count': float(
                    pass_stat_tensor[final_probe_pass, 1].item()
                ),
                'actor/eitr_coverage': float(
                    global_active_state_count.item()
                    / rollout_denominator
                ),
                'actor/eitr_lambda_env': self.eitr_lambda_env,
                'actor/eitr_effective_lambda': float(
                    self.eitr_lambda_env
                    * global_active_state_count.item()
                    / rollout_denominator
                    if self.eitr_mode == 'eitr'
                    else 0.0
                ),
                'actor/eitr_correction_lr': correction_lr,
                'actor/eitr_effective_step_scale': float(
                    eitr_effective_correction_lr
                    * self.eitr_lambda_env
                    * global_active_state_count.item()
                    / rollout_denominator
                    if self.eitr_mode == 'eitr'
                    else 0.0
                ),
                'actor/eitr_env_drift_pre': env_drift_pre,
                'actor/eitr_post_diagnostic_ran': float(post_diagnostic_ran),
                'actor/eitr_same_batch_scale_diagnostic_ran': float(
                    same_batch_scale_diagnostic_ran
                ),
                'actor/eitr_update_direction_diagnostic_ran': float(
                    update_direction_diagnostic_ran
                ),
                'actor/eitr_score_path_noop_direction_audit_ran': float(
                    score_path_noop_direction_audit_ran
                ),
                'actor/eitr_loss_applied': float(
                    eitr_correction_optimizer_step_count > 0
                ),
                'actor/eitr_probe_only': float(self.eitr_mode == 'probe_only'),
                'actor/eitr_grpo_pass_count': float(self.ppo_epochs),
                'actor/eitr_correction_pass_count': float(self.eitr_correction_passes),
            })
            if post_drift_stat_tensor is not None:
                env_drift_post = rollout_averaged_env_drift(
                    post_drift_stat_tensor[0].item(),
                    rollout_denominator,
                )
                env_drift_delta = env_drift_post - env_drift_pre
                # Positive reduction is good and is easier to read than the
                # opposite-signed delta.  Keep the raw delta as the canonical
                # quantity and report a scale-free diagnostic for smoke tests.
                env_drift_relative_reduction = (
                    (env_drift_pre - env_drift_post)
                    / max(abs(env_drift_pre), 1e-12)
                )
                append_to_dict(metrics, {
                    'actor/eitr_env_drift_post': env_drift_post,
                    'actor/eitr_env_drift_delta': env_drift_delta,
                    'actor/eitr_env_drift_relative_reduction': (
                        env_drift_relative_reduction
                    ),
                })

        # Report GRPO AdamW and the single full-batch EITR SGD step separately
        # so paired experiments share an unambiguous outer x-axis.
        self.grpo_optimizer_steps_completed += grpo_optimizer_step_count
        self.eitr_optimizer_steps_completed += eitr_correction_optimizer_step_count
        append_to_dict(metrics, {
            'actor/grpo_optimizer_step_count': float(grpo_optimizer_step_count),
            'actor/eitr_correction_optimizer_step_count': float(
                eitr_correction_optimizer_step_count
            ),
            'actor/optimizer_step_count': float(
                grpo_optimizer_step_count + eitr_correction_optimizer_step_count
            ),
            'actor/grpo_optimizer_step_count_cumulative': float(
                self.grpo_optimizer_steps_completed
            ),
            'actor/eitr_correction_optimizer_step_count_cumulative': float(
                self.eitr_optimizer_steps_completed
            ),
            'actor/optimizer_step_count_cumulative': float(
                self.grpo_optimizer_steps_completed + self.eitr_optimizer_steps_completed
            ),
        })
        return metrics
