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
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.eitr import (
    EITR_BATCH_KEYS,
    eitr_loss_enabled_for_pass,
    coverage_weighted_state_scale,
    eitr_probe_enabled_for_pass,
    induced_js_from_cached_effects,
    resolve_eitr_mode,
    validate_eitr_optimization_schedule,
)
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
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
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.eitr_config = self.config.get('eitr', {})
        self.eitr_mode = resolve_eitr_mode(self.eitr_config)
        self.eitr_uses_probes = self.eitr_mode != 'off'
        self.eitr_lambda_env = float(self.eitr_config.get('lambda_env', 0.1))
        self.ppo_epochs = int(self.config.get('ppo_epochs', 1))
        self.eitr_correction_passes = int(self.eitr_config.get('correction_passes', 1))
        self.grpo_optimizer_steps_completed = 0
        self.eitr_optimizer_steps_completed = 0
        validate_eitr_optimization_schedule(
            self.eitr_config,
            self.ppo_epochs,
            self.eitr_correction_passes,
        )

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

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
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
        # set to eval
        self.actor_module.eval()

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
        current_seq_logp = (token_log_probs.float() * response_mask).sum(dim=-1).view(
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

    def update_policy(self, data: DataProto):
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # avoid silently training at the wrong temperature

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.state_masking:
            select_keys.append('loss_mask')
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        if self.eitr_uses_probes:
            select_keys.extend(EITR_BATCH_KEYS)
        batch = data.select(batch_keys=select_keys).batch

        # PPO epochs retain their ordinary meaning: complete GRPO passes over
        # the rollout batch. Conditional EITR corrections run only after every
        # configured GRPO epoch has finished.
        dataloader = list(batch.split(self.config.ppo_mini_batch_size))
        metrics = {}
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

        for ppo_epoch in range(self.ppo_epochs):
            for mini_batch in dataloader:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.cuda()  # actor tensors may live on CPU under offload
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
                append_to_dict(metrics, {
                    'actor/grad_norm': grad_norm.detach().item(),
                    'actor/ppo_epoch': float(ppo_epoch),
                    f'actor/grad_norm_pass_{ppo_epoch}': grad_norm.detach().item(),
                })

        # Conditional correction is deliberately not another GRPO epoch. This
        # avoids giving EITR/Probe-GRPO an extra reward-bearing policy update.
        # With zero active states the whole section is skipped, so parameters
        # and optimizer state exactly match ordinary GRPO.
        if self.eitr_uses_probes:
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

            if global_active_state_count.item() > 0:
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

                    for mini_batch in dataloader:
                        mini_global_state_count = torch.tensor(
                            float(mini_batch['eitr_state_valid'].sum().item()),
                            dtype=torch.float64,
                            device=torch.cuda.current_device(),
                        )
                        if distributed:
                            torch.distributed.all_reduce(
                                mini_global_state_count,
                                op=torch.distributed.ReduceOp.SUM,
                            )
                        if mini_global_state_count.item() <= 0:
                            continue

                        mini_global_rollout_state_count = torch.tensor(
                            float(mini_batch['eitr_state_valid'].numel()),
                            dtype=torch.float64,
                            device=torch.cuda.current_device(),
                        )
                        if distributed:
                            torch.distributed.all_reduce(
                                mini_global_rollout_state_count,
                                op=torch.distributed.ReduceOp.SUM,
                            )

                        if self.config.use_dynamic_bsz:
                            max_token_len = (
                                self.config.ppo_max_token_len_per_gpu
                                * self.ulysses_sequence_parallel_size
                            )
                            micro_batches, _ = rearrange_micro_batches(
                                batch=mini_batch,
                                max_token_len=max_token_len,
                            )
                        else:
                            micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

                        if eitr_loss_enabled:
                            self.actor_optimizer.zero_grad()
                        local_applied_grad_sq = 0.0

                        for rollout_micro_batch in micro_batches:
                            # ``probe_micro_batch_size`` is expressed in flattened
                            # query sequences. Keep all K samples of one state
                            # together, then backward each state chunk immediately
                            # so earlier probe activations can be released.
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
                            state_chunks = rollout_micro_batch.split(states_per_probe_chunk)

                            for state_chunk in state_chunks:
                                state_chunk = state_chunk.cuda()
                                eitr_result = self._compute_eitr_micro_batch(
                                    state_chunk,
                                    temperature,
                                    track_model_grad=eitr_loss_enabled,
                                )
                                if eitr_result is None:
                                    # Online packing never reaches this branch.
                                    # The legacy sibling ablation has group-level
                                    # physical slots; its validated contiguous
                                    # layout makes the same chunks empty on every
                                    # rank, so they can be skipped symmetrically.
                                    continue

                                valid_state_count = eitr_result['valid_state_count']
                                state_weight = coverage_weighted_state_scale(
                                    valid_state_count,
                                    float(mini_global_rollout_state_count.item()),
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
                                    # Invalid local rows keep the same FSDP
                                    # backward structure through their zero-loss
                                    # dummy graph. Backward now releases this
                                    # chunk before the next state chunk forward.
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
                                correction_grad_norm = self._optimizer_step()
                                eitr_correction_optimizer_step_count += 1
                                append_to_dict(metrics, {
                                    'actor/eitr_correction_grad_norm': correction_grad_norm.detach().item(),
                                    f'actor/grad_norm_pass_{pass_index}': correction_grad_norm.detach().item(),
                                })
                            else:
                                # Do not advance AdamW or apply weight decay for
                                # a mathematically zero correction.
                                self.actor_optimizer.zero_grad()

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
            final_state_count = max(float(pass_stat_tensor[final_probe_pass, 1].item()), 1.0)
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
                    / max(global_rollout_state_count.item(), 1.0)
                ),
                'actor/eitr_lambda_env': self.eitr_lambda_env,
                'actor/eitr_effective_lambda': float(
                    self.eitr_lambda_env
                    * global_active_state_count.item()
                    / max(global_rollout_state_count.item(), 1.0)
                    if self.eitr_mode == 'eitr'
                    else 0.0
                ),
                'actor/eitr_loss_applied': float(
                    eitr_correction_optimizer_step_count > 0
                ),
                'actor/eitr_probe_only': float(self.eitr_mode == 'probe_only'),
                'actor/eitr_grpo_pass_count': float(self.ppo_epochs),
                'actor/eitr_correction_pass_count': float(self.eitr_correction_passes),
            })

        # A trainer outer update can contain several synchronized AdamW steps.
        # Report ordinary GRPO and EITR correction steps separately so paired
        # experiments share an unambiguous outer x-axis without hiding the
        # method's extra correction work.
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
