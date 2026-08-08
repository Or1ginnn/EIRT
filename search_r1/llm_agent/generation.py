import torch
import re
import random
from collections import Counter, defaultdict
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    max_trajectory_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    collect_eitr_probes: bool = False
    # Probability of probing a *valid* real search state. Phase 2 uses 1.0 so
    # scarce search states are never discarded. ``eitr_probe_count`` is the
    # target total K and includes the real query.
    eitr_probe_probability: float = 1.0
    eitr_probe_count: int = 4
    eitr_n_agent: int = 1
    # Default zero keeps the K samples unconditional: an invalid extra sample
    # lowers K_eff instead of being silently replaced by a later draw.
    eitr_probe_oversample: int = 0
    # Must match ``max_response_length``. Each same-state probe is cropped to
    # the real turn's remaining token budget after the fixed ``<search>``
    # prefix, so real and counterfactual queries share one action horizon.
    eitr_max_query_tokens: int = 500
    eitr_max_probe_prompt_tokens: int = 4096
    eitr_probe_seed: int = 20260805

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self._eitr_probe_call_index = 0
        self._eitr_rollout_call_index = 0
        self._eitr_current_rollout_index = 0

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_trajectory_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        if self.config.collect_eitr_probes:
            padded_active_batch.meta_info.update(active_batch.meta_info)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        final_generation_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        final_search_attempt_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        self._eitr_current_rollout_index = self._eitr_rollout_call_index
        self._eitr_rollout_call_index += 1
        self._eitr_first_search_records = (
            [None] * gen_batch.batch['input_ids'].shape[0]
            if self.config.collect_eitr_probes
            else None
        )
        self._eitr_probe_groups = (
            [None] * gen_batch.batch['input_ids'].shape[0]
            if self.config.collect_eitr_probes
            else None
        )
        # ``eitr_probe_groups`` remains a batch-aligned compatibility view
        # containing at most the first collected state for each real rollout.
        # The flat list is the lossless Phase-2 representation and can contain
        # every valid search turn from every real trajectory.
        self._eitr_probe_state_groups = (
            []
            if self.config.collect_eitr_probes
            else None
        )
        self._eitr_current_search_records = (
            {}
            if self.config.collect_eitr_probes
            else None
        )
        self._eitr_primary_search_seen = (
            [False] * gen_batch.batch['input_ids'].shape[0]
            if self.config.collect_eitr_probes
            else None
        )
        self._eitr_pending_probe_groups = (
            []
            if self.config.collect_eitr_probes
            else None
        )
        self._eitr_probe_collection_stats = (
            Counter()
            if self.config.collect_eitr_probes
            else None
        )

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            # Count actual LLM generations explicitly. ``max_turns`` bounds the
            # retriever-enabled interaction loop; the optional final generation
            # below is a separate answer opportunity.
            turns_stats[active_mask] += 1
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            active_indices = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info
            raw_responses_ids = (
                gen_output.batch['responses'].detach().cpu()
                if self.config.collect_eitr_probes
                else None
            )
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            if self.config.collect_eitr_probes:
                self._eitr_current_search_records = {}
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str,
                self.tokenizer.pad_token,
                active_mask,
                turn_index=step,
            )
            if self.config.collect_eitr_probes:
                self._collect_eitr_same_state_probes(
                    rollings=rollings,
                    raw_responses_ids=raw_responses_ids,
                    active_indices=active_indices,
                    turn_index=step,
                )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # final LLM rollout
        if active_mask.sum():
            turns_stats[active_mask] += 1
            final_generation_stats[active_mask] += 1
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            final_actions, _ = self.postprocess_predictions(responses_str)
            final_search_attempt_stats += torch.tensor([
                int(bool(active) and action == 'search')
                for action, active in zip(final_actions, active_mask)
            ], dtype=torch.int)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )

        # Counterfactual generation is deliberately delayed until every real
        # turn (including the final answer turn) has finished. Probe requests
        # therefore cannot consume RNG between two real trajectory turns.
        if self.config.collect_eitr_probes:
            self._finalize_eitr_same_state_probes()
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        meta_info['final_generation_stats'] = final_generation_stats.tolist()
        meta_info['final_search_attempt_stats'] = final_search_attempt_stats.tolist()
        if self.config.collect_eitr_probes:
            meta_info['eitr_first_search_records'] = self._eitr_first_search_records
            meta_info['eitr_probe_groups'] = self._eitr_probe_groups
            meta_info['eitr_probe_state_groups'] = self._eitr_probe_state_groups
            meta_info['eitr_probe_collection_stats'] = dict(
                self._eitr_probe_collection_stats or {}
            )
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(
        self,
        predictions: List[str],
        pad_token: str,
        active_mask=None,
        do_search=True,
        turn_index: Optional[int] = None,
    ) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    retrieval_result = search_results.pop(0)
                    if do_search:
                        next_obs.append(
                            f'\n\n<information>{self._passages2string(retrieval_result).strip()}</information>\n\n'
                        )
                        if self.config.collect_eitr_probes:
                            self._record_eitr_search(
                                index=i,
                                turn_index=turn_index,
                                prediction=predictions[i],
                                query=contents[i],
                                retrieval_result=retrieval_result,
                            )
                        dones.append(0)
                        valid_action.append(1)
                        is_search.append(1)
                    else:
                        # The final answer opportunity does not execute tools.
                        # A generated search here is an unfinished/invalid final
                        # action, not a real retriever call.
                        next_obs.append('\n\n<information></information>\n\n')
                        dones.append(0)
                        valid_action.append(0)
                        is_search.append(0)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

    def _record_eitr_search(
        self,
        index: int,
        turn_index: Optional[int],
        prediction: str,
        query: str,
        retrieval_result: List[Dict[str, Any]],
    ) -> None:
        """Cache one complete, non-empty real search action.

        The real environment path has already accepted this action. EITR adds a
        stricter conditional mask: an opening tag alone or an empty closed
        query is not a valid query-policy state and is never probed.
        """
        first_records = getattr(self, '_eitr_first_search_records', None)
        current_records = getattr(self, '_eitr_current_search_records', None)
        if first_records is None or current_records is None:
            return
        match = re.search(r'<search>(.*?)</search>', prediction, re.DOTALL)
        if match is None:
            self._eitr_probe_collection_stats['real_search_missing_close_tag'] += 1
            return
        normalized_query = self._normalize_eitr_query_text(query)
        matched_query = self._normalize_eitr_query_text(match.group(1))
        if not normalized_query or not matched_query:
            self._eitr_probe_collection_stats['real_search_empty_query'] += 1
            return
        action_text = prediction[:match.end()]
        action_token_ids = self.tokenizer(
            action_text,
            add_special_tokens=False,
        )['input_ids']
        record = {
            'queries': [normalized_query],
            'source_index': int(index),
            'turn_index': int(turn_index) if turn_index is not None else 0,
            'prefix_text': prediction[:match.start()],
            'search_open_text': prediction[:match.start(1)],
            'action_text': action_text,
            'action_token_ids': list(action_token_ids),
            'retrieval_effect': self._compact_retrieval_effect(retrieval_result),
        }
        current_records[index] = record
        if first_records[index] is None:
            first_records[index] = record
        stats = self._eitr_probe_collection_stats
        stats['real_search_valid'] += 1
        stats[f'real_search_valid_turn_{record["turn_index"]}'] += 1

    def _record_eitr_first_search(
        self,
        index: int,
        prediction: str,
        query: str,
        retrieval_result: List[Dict[str, Any]],
    ) -> None:
        """Backward-compatible wrapper used by early Gate-C integrations."""
        self._record_eitr_search(
            index=index,
            turn_index=0,
            prediction=prediction,
            query=query,
            retrieval_result=retrieval_result,
        )

    @staticmethod
    def _find_token_subsequence(values: List[int], pattern: List[int]) -> int:
        if not pattern or len(pattern) > len(values):
            return -1
        for offset in range(len(values) - len(pattern) + 1):
            if values[offset:offset + len(pattern)] == pattern:
                return offset
        return -1

    def _locate_search_token_boundaries(
        self,
        generated_ids: List[int],
        record: Dict[str, Any],
    ) -> Optional[Tuple[int, int]]:
        """Locate the generated token span after ``<search>`` through ``</search>``.

        Byte-level tokenizers can encode an isolated ``<search>`` differently
        from the same text after whitespace or reasoning tokens. Re-tokenizing
        the exact decoded prefixes preserves that context. The progressive
        decode is a compatibility fallback for rare non-idempotent round trips.
        """
        open_text = record.get('search_open_text')
        action_text = record.get('action_text')
        if open_text and action_text:
            open_ids = list(self.tokenizer(open_text, add_special_tokens=False)['input_ids'])
            action_ids = list(self.tokenizer(action_text, add_special_tokens=False)['input_ids'])
            if (
                0 < len(open_ids) < len(action_ids) <= len(generated_ids)
                and generated_ids[:len(action_ids)] == action_ids
                and action_ids[:len(open_ids)] == open_ids
            ):
                return len(open_ids), len(action_ids)

        open_end = None
        for token_end in range(1, len(generated_ids) + 1):
            prefix_text = self.tokenizer.decode(
                generated_ids[:token_end],
                skip_special_tokens=True,
            )
            if open_end is None and '<search>' in prefix_text:
                open_end = token_end
            if open_end is not None and '</search>' in prefix_text:
                return open_end, token_end
        return None

    def _should_probe_eitr_state(self, source_index: int, turn_index: int) -> bool:
        """Make a reproducible Bernoulli decision for one valid search state."""
        probability = float(getattr(self.config, 'eitr_probe_probability', 1.0))
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f'eitr_probe_probability must be in [0, 1], got {probability}'
            )
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        seed = int(getattr(self.config, 'eitr_probe_seed', 20260805))
        rollout_index = int(getattr(self, '_eitr_current_rollout_index', 0))
        state_seed = f'{seed}:{rollout_index}:{int(turn_index)}:{int(source_index)}'
        return random.Random(state_seed).random() < probability

    @staticmethod
    def _normalize_eitr_query_text(query: Any) -> str:
        """Apply the same non-empty query normalization to real and probe actions."""
        return ' '.join(str(query).strip().split())

    def _parse_eitr_probe_response(
        self,
        response_tokens: List[int],
    ) -> Tuple[Optional[str], Optional[List[int]], Optional[str]]:
        """Parse a query continuation using decoded closing-tag detection.

        Looking for tokenized ``</search>`` in isolation is unsafe for
        context-sensitive tokenizers. Progressive decoding also gives the exact
        generated token boundary needed by the later teacher-forced log-prob
        computation.
        """
        close_end = None
        decoded_action = None
        for token_end in range(1, len(response_tokens) + 1):
            decoded_prefix = self.tokenizer.decode(
                response_tokens[:token_end],
                skip_special_tokens=True,
            )
            if '</search>' in decoded_prefix:
                close_end = token_end
                decoded_action = decoded_prefix
                break
        if close_end is None or decoded_action is None:
            return None, None, 'missing_close_tag'

        query_text = decoded_action.split('</search>', 1)[0]
        query = self._normalize_eitr_query_text(query_text)
        if not query:
            return None, None, 'empty_query'
        return query, response_tokens[:close_end], None

    def _register_compat_eitr_group(self, group: Dict[str, Any]) -> None:
        """Expose one primary state per rollout through the legacy field."""
        compatibility_groups = getattr(self, '_eitr_probe_groups', None)
        if compatibility_groups is None:
            return
        source_index = int(group['source_index'])
        if not 0 <= source_index < len(compatibility_groups):
            return
        current = compatibility_groups[source_index]
        if (
            not isinstance(current, dict)
            or int(group.get('effective_probe_count', 0))
            > int(current.get('effective_probe_count', 0))
        ):
            compatibility_groups[source_index] = group

    def _collect_eitr_same_state_probes(
        self,
        rollings: DataProto,
        raw_responses_ids: torch.Tensor,
        active_indices: Optional[List[int]] = None,
        turn_index: int = 0,
    ) -> None:
        """Cache the selected same-state prefix without running a probe yet.

        The real rollout is never modified. A selected state always retains its
        real query. Counterfactual generation and retrieval happen only in the
        post-rollout finalizer, where malformed probes lower the effective K.
        """
        if self._eitr_probe_groups is None:
            return
        stats = self._eitr_probe_collection_stats
        probe_count = max(int(self.config.eitr_probe_count), 1)
        n_agent = int(self.config.eitr_n_agent)
        if n_agent <= 0:
            raise ValueError(
                f'EITR expected positive n_agent, got {n_agent}'
            )

        if active_indices is None:
            active_indices = list(range(raw_responses_ids.size(0)))
        if len(active_indices) != raw_responses_ids.size(0):
            raise ValueError(
                'EITR active-index mapping must align with generated responses: '
                f'{len(active_indices)} != {raw_responses_ids.size(0)}'
            )

        current_records = getattr(self, '_eitr_current_search_records', None)
        if not isinstance(current_records, dict):
            # Compatibility with the original first-search collector unit test.
            first_records = getattr(self, '_eitr_first_search_records', None) or []
            current_records = {
                index: first_records[index]
                for index in active_indices
                if index < len(first_records) and first_records[index]
            }

        flat_groups = getattr(self, '_eitr_probe_state_groups', None)
        if flat_groups is None:
            flat_groups = []
            self._eitr_probe_state_groups = flat_groups
        primary_seen = getattr(self, '_eitr_primary_search_seen', None)
        if primary_seen is None:
            primary_seen = [False] * len(self._eitr_probe_groups)
            self._eitr_primary_search_seen = primary_seen

        state_groups: List[Dict[str, Any]] = []
        response_by_source = {
            source_index: raw_responses_ids[local_index].tolist()
            for local_index, source_index in enumerate(active_indices)
        }
        for source_index in active_indices:
            record = current_records.get(source_index)
            if not record:
                continue
            stats['state_group_total'] += 1
            state_turn = int(record.get('turn_index', turn_index))
            is_additional_state = bool(primary_seen[source_index])
            if len(record.get('queries') or []) != 1:
                stats['candidate_not_single_query'] += 1
                stats['state_group_rejected'] += 1
                continue
            if not record.get('retrieval_effect'):
                stats['candidate_empty_retrieval_effect'] += 1
                stats['state_group_rejected'] += 1
                continue

            generated_ids = response_by_source[source_index]
            boundaries = self._locate_search_token_boundaries(generated_ids, record)
            if boundaries is None:
                stats['candidate_search_boundary_not_found'] += 1
                stats['state_group_rejected'] += 1
                continue
            open_end, close_end = boundaries
            normal_action_ids = generated_ids[open_end:close_end]
            if not normal_action_ids:
                stats['candidate_empty_query_action'] += 1
                stats['state_group_rejected'] += 1
                continue
            query_token_budget = int(self.config.max_response_length) - int(open_end)
            if query_token_budget <= 0 or len(normal_action_ids) > query_token_budget:
                # The real turn itself was sampled with max_response_length. A
                # probe must inherit the exact suffix budget left after the
                # fixed reasoning + <search> prefix.
                stats['candidate_query_budget_mismatch'] += 1
                stats['state_group_rejected'] += 1
                continue
            prompt_mask = rollings.batch['attention_mask'][source_index].bool()
            base_prompt_ids = rollings.batch['input_ids'][source_index][prompt_mask].tolist()
            fixed_prefix_ids = generated_ids[:open_end]
            state_prompt_ids = base_prompt_ids + fixed_prefix_ids
            max_probe_prompt_tokens = int(
                getattr(
                    self.config,
                    'eitr_max_probe_prompt_tokens',
                    self.config.max_prompt_length,
                )
            )
            if len(state_prompt_ids) > max_probe_prompt_tokens:
                # The real query was sampled under the full prefix. Silently
                # shortening it would make probe generation and later scoring
                # condition on a different state.
                stats['state_prompt_too_long'] += 1
                stats['state_group_rejected'] += 1
                continue
            if not is_additional_state:
                # Consume the rollout's one Phase-2 probe opportunity only
                # after a usable exact state has actually been constructed.
                # An empty retrieval or an unlocatable boundary on an earlier
                # turn must not prevent a later valid search from being used.
                primary_seen[source_index] = True
            group_start = source_index - (source_index % n_agent)
            group = {
                'state_id': f'{source_index}:{state_turn}',
                'state_prompt_token_ids': state_prompt_ids,
                'source_index': source_index,
                'turn_index': state_turn,
                'group_start': group_start,
                'target_probe_count': probe_count,
                'query_token_budget': query_token_budget,
                'selected_for_probe': False,
                'deferred': is_additional_state,
                'extra_retrieval_calls': 0,
                'probes': [{
                    'query': record['queries'][0],
                    'action_token_ids': normal_action_ids,
                    'retrieval_effect': record['retrieval_effect'],
                }],
            }
            flat_groups.append(group)
            if is_additional_state:
                group['effective_probe_count'] = 1
                group['eitr_eligible'] = False
                stats['additional_state_deferred'] += 1
                continue
            if not self._should_probe_eitr_state(source_index, state_turn):
                group['effective_probe_count'] = 1
                group['eitr_eligible'] = False
                stats['probe_state_not_selected'] += 1
                self._register_compat_eitr_group(group)
                continue
            group['selected_for_probe'] = True
            stats['probe_state_selected'] += 1
            state_groups.append(group)
            stats['state_group_collected'] += 1

        if not state_groups:
            return
        if probe_count <= 1:
            for group in state_groups:
                group['effective_probe_count'] = len(group['probes'])
                group['eitr_eligible'] = len(group['probes']) >= 2
                if group['eitr_eligible']:
                    stats['probe_state_effective'] += 1
                else:
                    stats['probe_state_insufficient'] += 1
                self._register_compat_eitr_group(group)
            return

        pending_groups = getattr(self, '_eitr_pending_probe_groups', None)
        if pending_groups is None:
            pending_groups = []
            self._eitr_pending_probe_groups = pending_groups
        pending_groups.extend(state_groups)

    def _finalize_eitr_same_state_probes(self) -> None:
        """Generate and retrieve all cached probes after the real rollout ends."""
        state_groups = list(getattr(self, '_eitr_pending_probe_groups', None) or [])
        self._eitr_pending_probe_groups = []
        if not state_groups:
            return

        stats = self._eitr_probe_collection_stats
        probe_count = max(int(self.config.eitr_probe_count), 1)
        candidates_per_state = max(
            probe_count - 1 + int(self.config.eitr_probe_oversample),
            probe_count - 1,
        )
        if candidates_per_state <= 0:
            for group in state_groups:
                group['effective_probe_count'] = len(group['probes'])
                group['eitr_eligible'] = len(group['probes']) >= 2
                stats['probe_effective_k_sum'] += len(group['probes'])
                if group['eitr_eligible']:
                    stats['probe_state_effective'] += 1
                else:
                    stats['probe_state_insufficient'] += 1
                self._register_compat_eitr_group(group)
            return

        # vLLM constructs one seeded generator per request. Repeating the same
        # state several rows in one call with one shared seed therefore produces
        # identical continuations. Sample one candidate per state per round and
        # advance the seed between rounds so the K-1 candidates for a state are
        # genuine independent draws from the frozen rollout policy.
        generated_candidates = []
        state_prompt_ids = [group['state_prompt_token_ids'] for group in state_groups]
        max_state_length = max(len(item) for item in state_prompt_ids)
        max_query_budget = max(int(group['query_token_budget']) for group in state_groups)
        configured_query_limit = int(self.config.eitr_max_query_tokens)
        if configured_query_limit != int(self.config.max_response_length):
            raise ValueError(
                'EITR max_query_tokens must equal the real rollout '
                f'max_response_length; got {configured_query_limit} != '
                f'{int(self.config.max_response_length)}'
            )
        for _ in range(candidates_per_state):
            probe_input_ids = torch.full(
                (len(state_prompt_ids), max_state_length),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
            )
            probe_attention_mask = torch.zeros_like(probe_input_ids)
            for index, token_ids in enumerate(state_prompt_ids):
                length = len(token_ids)
                probe_input_ids[index, -length:] = torch.tensor(token_ids, dtype=torch.long)
                probe_attention_mask[index, -length:] = 1
            probe_position_ids = self.tensor_fn.create_position_ids(probe_attention_mask)
            probe_prompts = DataProto.from_dict({
                'input_ids': probe_input_ids,
                'attention_mask': probe_attention_mask,
                'position_ids': probe_position_ids,
            })
            probe_seed = int(
                self.config.eitr_probe_seed + self._eitr_probe_call_index
            )
            probe_prompts.meta_info.update({
                'recompute_log_prob': False,
                'sampling_params': {
                    # One batched vLLM call needs one shared limit. Every result
                    # is cropped below to its owning real state's smaller
                    # residual budget before parsing.
                    'max_tokens': max_query_budget,
                    'n': 1,
                    'seed': probe_seed,
                },
            })
            self._eitr_probe_call_index += 1
            probe_outputs = self._generate_with_gpu_padding(probe_prompts)
            stats['probe_generation_call_count'] += 1
            stats['probe_candidate_generated'] += len(state_groups)
            generated_candidates.extend(
                zip(range(len(state_groups)), probe_outputs.batch['responses'])
            )

        valid_candidates = []
        flat_queries = []
        seen_queries_by_owner = {
            owner: {group['probes'][0]['query'].strip().lower()}
            for owner, group in enumerate(state_groups)
        }
        accepted_candidates_by_owner = defaultdict(int)
        for owner, response in generated_candidates:
            if accepted_candidates_by_owner[owner] >= probe_count - 1:
                continue
            query_token_budget = int(state_groups[owner]['query_token_budget'])
            response_tokens = response.tolist()[:query_token_budget]
            query, action_ids, rejection_reason = self._parse_eitr_probe_response(
                response_tokens
            )
            if rejection_reason is not None:
                stats[f'probe_{rejection_reason}'] += 1
                continue
            query_key = query.lower()
            if query_key in seen_queries_by_owner[owner]:
                stats['probe_duplicate_query'] += 1
                # Duplicate samples are still valid draws from pi_old. Dropping
                # them would condition the Monte Carlo estimator on uniqueness.
            seen_queries_by_owner[owner].add(query_key)
            valid_candidates.append((owner, query, action_ids))
            flat_queries.append(query)
            accepted_candidates_by_owner[owner] += 1
            stats['probe_query_accepted'] += 1

        retrieval_results = self.batch_search(flat_queries) if flat_queries else []
        for (owner, query, action_ids), retrieval_result in zip(valid_candidates, retrieval_results):
            group = state_groups[owner]
            group['extra_retrieval_calls'] = int(
                group.get('extra_retrieval_calls', 0)
            ) + 1
            if len(group['probes']) >= probe_count:
                continue
            effect = self._compact_retrieval_effect(retrieval_result)
            if effect:
                group['probes'].append({
                    'query': query,
                    'action_token_ids': action_ids,
                    'retrieval_effect': effect,
                })
                stats['probe_effect_accepted'] += 1
            else:
                stats['probe_empty_retrieval_effect'] += 1

        for group in state_groups:
            effective_count = len(group['probes'])
            group['effective_probe_count'] = effective_count
            group['eitr_eligible'] = effective_count >= 2
            stats['probe_effective_k_sum'] += effective_count
            if group['eitr_eligible']:
                stats['probe_state_effective'] += 1
            else:
                stats['probe_state_insufficient'] += 1
            self._register_compat_eitr_group(group)

    @staticmethod
    def _compact_retrieval_effect(retrieval_result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compact = []
        for rank, item in enumerate(retrieval_result or []):
            document = item.get('document') or {}
            doc_id = document.get('id') or document.get('title') or document.get('contents')
            if not doc_id:
                continue
            score = item.get('score')
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = -float(rank)
            compact.append({'doc_id': str(doc_id), 'score': score, 'rank': rank})
        return compact

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    # A closed tag with an empty payload is still a malformed
                    # tool/final action. In particular, <search></search> must
                    # not be counted as a valid search state for Conditional EITR.
                    action = match.group(1) if content else None
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> List[List[Dict[str, Any]]]:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            raw top-k retrieval records for each query
        """
        return self._batch_search(queries)['result']

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
