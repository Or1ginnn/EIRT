import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from .tensor_helper import TensorHelper, TensorConfig
from .ca_ecad import (
    ACTION_ANSWER,
    ACTION_INVALID,
    ACTION_SEARCH,
    build_policy_segment_ids,
    ordered_document_ids,
)
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
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    record_ca_ecad_trace: bool = False

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
                prompt_turn_ids: torch.Tensor = None,
                response_turn_ids: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        trace_enabled = prompt_turn_ids is not None or response_turn_ids is not None
        if trace_enabled and (prompt_turn_ids is None or response_turn_ids is None):
            raise ValueError("both prompt_turn_ids and response_turn_ids are required")
        turn_id_tensors = [prompt_turn_ids, response_turn_ids] if trace_enabled else None
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
            if trace_enabled:
                info_turn_ids = torch.where(
                    info != pad_id,
                    torch.zeros_like(info, dtype=torch.long),
                    torch.full_like(info, -1, dtype=torch.long),
                )
                turn_id_tensors.append(info_turn_ids)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)
        padded_turn_ids = None
        if trace_enabled:
            concatenated_turn_ids = torch.cat(turn_id_tensors, dim=1)
            padded_turn_ids = concatenated_turn_ids.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info, padded_turn_ids

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None,
                          cur_turn_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask, generation_turn_ids = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    prompt_turn_ids=right_side.get('generation_turn_ids'),
                    response_turn_ids=cur_turn_ids,
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask, generation_turn_ids = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    prompt_turn_ids=right_side.get('generation_turn_ids'),
                    response_turn_ids=cur_turn_ids,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        output = {
            'responses': responses[:, :max_len],
            'responses_with_info_mask': responses_with_info_mask[:, :max_len],
        }
        if generation_turn_ids is not None:
            output['generation_turn_ids'] = generation_turn_ids[:, :max_len]
        return output

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
        batch_size = gen_batch.batch['input_ids'].shape[0]
        turn_action_codes = None
        search_document_id_history = None
        if self.config.record_ca_ecad_trace:
            original_right_side['generation_turn_ids'] = torch.empty(
                (batch_size, 0), dtype=torch.long, device=initial_input_ids.device
            )
            turn_action_codes = torch.full(
                (batch_size, self.config.max_turns + 1),
                -1,
                dtype=torch.long,
                device=initial_input_ids.device,
            )
            search_document_id_history = [[] for _ in range(batch_size)]
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
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
            cur_turn_ids = None
            if self.config.record_ca_ecad_trace:
                cur_turn_ids = torch.where(
                    responses_ids != self.tokenizer.pad_token_id,
                    torch.full_like(responses_ids, step + 1, dtype=torch.long),
                    torch.full_like(responses_ids, -1, dtype=torch.long),
                )

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search, action_codes, document_ids = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )
            if self.config.record_ca_ecad_trace:
                turn_action_codes[:, step] = torch.tensor(
                    action_codes,
                    dtype=torch.long,
                    device=turn_action_codes.device,
                )
                for row_index, row_document_ids in enumerate(document_ids):
                    if row_document_ids is not None:
                        search_document_id_history[row_index].append(list(row_document_ids))
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
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
                next_obs_ids,
                cur_turn_ids=cur_turn_ids,
            )
            
        # final LLM rollout
        if active_mask.sum():
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
            cur_turn_ids = None
            if self.config.record_ca_ecad_trace:
                final_generation_turn = self.config.max_turns + 1
                cur_turn_ids = torch.where(
                    responses_ids != self.tokenizer.pad_token_id,
                    torch.full_like(responses_ids, final_generation_turn, dtype=torch.long),
                    torch.full_like(responses_ids, -1, dtype=torch.long),
                )

            # # Execute in environment and process observations
            _, dones, valid_action, is_search, action_codes, document_ids = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )
            if self.config.record_ca_ecad_trace:
                turn_action_codes[:, self.config.max_turns] = torch.tensor(
                    action_codes,
                    dtype=torch.long,
                    device=turn_action_codes.device,
                )
                if any(row_document_ids is not None for row_document_ids in document_ids):
                    raise RuntimeError("the final no-search generation unexpectedly executed retrieval")

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                cur_turn_ids=cur_turn_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(
            original_left_side,
            original_right_side,
            meta_info,
            turn_action_codes=turn_action_codes,
            search_document_id_history=search_document_id_history,
        )

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            turn_action_codes: torch.Tensor = None,
                            search_document_id_history: List[List[List[str]]] = None) -> Tuple[Dict, Dict]:
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

        non_tensors = None
        if self.config.record_ca_ecad_trace:
            if turn_action_codes is None or search_document_id_history is None:
                raise RuntimeError("CA-ECAD tracing is enabled but trace state is missing")
            generation_turn_ids = final_output.pop('generation_turn_ids')
            response_policy_mask = self.tensor_fn.create_attention_mask(
                final_output['responses_with_info_mask']
            )
            segment_ids = torch.full_like(generation_turn_ids, -1, dtype=torch.long)
            search_counts = torch.zeros(
                generation_turn_ids.shape[0],
                dtype=torch.long,
                device=generation_turn_ids.device,
            )
            for row_index in range(generation_turn_ids.shape[0]):
                row_segment_ids = build_policy_segment_ids(
                    generation_turn_ids[row_index].tolist(),
                    turn_action_codes[row_index].tolist(),
                    response_policy_mask[row_index].tolist(),
                )
                segment_ids[row_index] = torch.tensor(
                    row_segment_ids,
                    dtype=torch.long,
                    device=segment_ids.device,
                )
                search_counts[row_index] = len(search_document_id_history[row_index])
                executed_action_count = int((turn_action_codes[row_index] == ACTION_SEARCH).sum().item())
                if executed_action_count != int(search_counts[row_index].item()):
                    raise RuntimeError(
                        f"search trace mismatch for row {row_index}: "
                        f"actions={executed_action_count}, ids={int(search_counts[row_index].item())}"
                    )

            final_output['ca_ecad_generation_turn_ids'] = generation_turn_ids
            final_output['ca_ecad_turn_action_codes'] = turn_action_codes
            final_output['ca_ecad_policy_segment_ids'] = segment_ids
            final_output['ca_ecad_search_count'] = search_counts
            search_history_array = np.empty(len(search_document_id_history), dtype=object)
            search_history_array[:] = search_document_id_history
            non_tensors = {
                'ca_ecad_search_document_ids': search_history_array,
            }
        
        final_output = DataProto.from_dict(final_output, non_tensors=non_tensors)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(
        self,
        predictions: List[str],
        pad_token: str,
        active_mask=None,
        do_search=True,
    ) -> Tuple[List[str], List[int], List[int], List[int], List[int], List[Optional[Tuple[Optional[str], ...]]]]:
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
        if self.config.record_ca_ecad_trace:
            # Phase 2 defines a valid search by an executed non-empty query.
            # Keep the unmodified Search-R1 parser semantics outside trace mode.
            cur_actions = [
                None if action == 'search' and not content.strip() else action
                for action, content in zip(cur_actions, contents)
            ]
        next_obs, dones, valid_action, is_search = [], [], [], []
        action_codes, document_ids = [], []
        
        search_queries = [
            content
            for action, content, active in zip(cur_actions, contents, active_mask)
            if bool(active) and action == 'search'
        ]
        if do_search and search_queries:
            search_results = self.batch_search_with_metadata(search_queries)
            assert len(search_results) == len(search_queries)
        elif not do_search and not self.config.record_ca_ecad_trace:
            search_results = [('', tuple())] * sum(1 for action in cur_actions if action == 'search')
        else:
            search_results = []

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                action_codes.append(-1)
                document_ids.append(None)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                    action_codes.append(ACTION_ANSWER)
                    document_ids.append(None)
                elif action == 'search' and (do_search or not self.config.record_ca_ecad_trace):
                    formatted_result, ordered_ids = search_results.pop(0)
                    next_obs.append(f'\n\n<information>{formatted_result.strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                    action_codes.append(ACTION_SEARCH)
                    document_ids.append(ordered_ids)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
                    action_codes.append(ACTION_INVALID)
                    document_ids.append(None)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search, action_codes, document_ids

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
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        return [formatted for formatted, _ in self.batch_search_with_metadata(queries)]

    def batch_search_with_metadata(self, queries: List[str]) -> List[Tuple[str, Tuple[Optional[str], ...]]]:
        """Return model observations together with ordered corpus document IDs."""
        if not queries:
            return []
        results = self._batch_search(queries)['result']
        output = []
        for result in results:
            output.append((self._passages2string(result), ordered_document_ids(result)))
        return output

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        response = requests.post(self.config.search_url, json=payload)
        response.raise_for_status()
        result = response.json()
        if 'result' not in result or len(result['result']) != len(queries):
            raise RuntimeError("retriever response is not aligned with the query batch")
        return result

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
