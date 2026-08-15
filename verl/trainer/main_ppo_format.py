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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em, qa_em_format
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'web_questions', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle', 'strategyqa']:
        return qa_em_format.compute_score_em
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, structure_format_score=0., final_format_score=0., retrieval_score=0., format_score=0.,
                 reward_profile='pure_em', think_format_score=0.2,
                 answer_format_score=0.1, evidence_score=0.0,
                 answer_em_score=1.2, joint_success_bonus=0.0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.structure_format_score = structure_format_score
        self.final_format_score = final_format_score
        self.retrieval_score = retrieval_score
        self.reward_profile = reward_profile
        self.think_format_score = think_format_score
        self.answer_format_score = answer_format_score
        self.evidence_score = evidence_score
        self.answer_em_score = answer_em_score
        self.joint_success_bonus = joint_success_bonus
        self.configured_max_score = (
            float(think_format_score)
            + float(answer_format_score)
            + float(evidence_score)
            + float(answer_em_score)
            + float(joint_success_bonus)
            if reward_profile == 'mandatory_search'
            else 1.0
        )
        self.last_metrics = {}

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # Never leak diagnostics from a previous rule-reward batch when an
        # externally supplied reward-model score takes the early-return path.
        self.last_metrics = {}

        # A learned RM score cannot silently bypass the mandatory environment
        # gate. Combining both objectives would need an explicit definition.
        if self.reward_profile == 'mandatory_search' and 'rm_scores' in data.batch.keys():
            raise RuntimeError(
                "mandatory_search is incompatible with precomputed rm_scores"
            )

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        if (
            self.reward_profile == 'mandatory_search'
            and (
                'executed_search_count' not in data.batch.keys()
                or 'info_mask' not in data.batch.keys()
            )
        ):
            raise RuntimeError(
                "mandatory_search reward requires batch-aligned "
                "executed_search_count and info_mask from the environment"
            )

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # all_scores = []

        already_print_data_sources = {}
        reward_details = []
        trajectory_scores = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch['responses']
            valid_response_length = int(
                data_item.batch['attention_mask'][prompt_length:].sum().item()
            )
            valid_response_ids = response_ids[:valid_response_length]

            # Score only the generated trajectory.  The user prompt contains
            # literal <answer> examples and must not be treated as a model
            # answer.  Keep the assistant marker because the v0.3 structure
            # validator uses it to delimit the generated turn.
            response_str = self.tokenizer.decode(valid_response_ids)
            sequences_str = f"<|im_start|>assistant\n{response_str}"
            environment_observation_str = None
            model_generated_str = None
            if 'info_mask' in data_item.batch.keys():
                response_info_mask = data_item.batch['info_mask'][
                    prompt_length:prompt_length + valid_response_length
                ]
                environment_observation_str = (
                    qa_em_format.decode_environment_observation(
                        self.tokenizer,
                        valid_response_ids,
                        response_info_mask,
                    )
                )
                model_generated_str = (
                    qa_em_format.decode_model_generated_response(
                        self.tokenizer,
                        valid_response_ids,
                        response_info_mask,
                    )
                )

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            executed_search_count = (
                int(data_item.batch['executed_search_count'].item())
                if 'executed_search_count' in data_item.batch.keys()
                else None
            )

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score, details = compute_score_fn(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                structure_format_score=self.structure_format_score,
                final_format_score=self.final_format_score,
                retrieval_score=self.retrieval_score,
                format_score=self.format_score,
                reward_profile=self.reward_profile,
                executed_search_count=executed_search_count,
                environment_observation_str=environment_observation_str,
                model_generated_str=model_generated_str,
                think_format_score=self.think_format_score,
                answer_format_score=self.answer_format_score,
                evidence_score=self.evidence_score,
                answer_em_score=self.answer_em_score,
                joint_success_bonus=self.joint_success_bonus,
                return_details=True,
            )
            reward_details.append(details)
            trajectory_scores.append(float(score))

            reward_tensor[i, valid_response_length - 1] = score
            # all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        count = max(len(reward_details), 1)
        searched_count = sum(
            int(item['has_executed_search']) for item in reward_details
        )
        answer_bearing_count = sum(
            int(item['answer_bearing_evidence']) for item in reward_details
        )
        reward_group_metrics = qa_em_format.reward_group_diagnostics(
            trajectory_scores,
            (
                data.non_tensor_batch['uid']
                if 'uid' in data.non_tensor_batch
                else range(len(trajectory_scores))
            ),
        )
        self.last_metrics = {
            'reward/configured_max_score': self.configured_max_score,
            'reward/mandatory_search_profile': float(
                self.reward_profile == 'mandatory_search'
            ),
            'reward/soft_format_components': float(
                self.reward_profile == 'mandatory_search'
            ),
            'reward/answer_hard_gate': float(
                self.reward_profile == 'mandatory_search'
            ),
            'reward/think_format_score': float(self.think_format_score),
            'reward/answer_format_score': float(self.answer_format_score),
            'reward/evidence_score': float(self.evidence_score),
            'reward/answer_em_score': float(self.answer_em_score),
            'reward/joint_success_bonus': float(self.joint_success_bonus),
            'reward/answer_em_rate': sum(
                int(item['answer_em']) for item in reward_details
            ) / count,
            'reward/format_valid_rate': sum(
                int(item['format_valid']) for item in reward_details
            ) / count,
            'reward/think_format_valid_rate': sum(
                int(item['think_format_valid']) for item in reward_details
            ) / count,
            'reward/answer_format_valid_rate': sum(
                int(item['answer_format_valid']) for item in reward_details
            ) / count,
            'reward/tool_trace_consistent_rate': sum(
                int(item['tool_trace_consistent']) for item in reward_details
            ) / count,
            'reward/generated_information_rate': sum(
                int(item['generated_information_detected'])
                for item in reward_details
            ) / count,
            'reward/hard_reward_gate_pass_rate': sum(
                int(item['hard_reward_gate_pass']) for item in reward_details
            ) / count,
            'reward/answer_bearing_evidence_rate': answer_bearing_count / count,
            'reward/answer_bearing_given_search': (
                answer_bearing_count / searched_count
                if searched_count
                else 0.0
            ),
            'reward/evidence_bonus_applied_rate': sum(
                int(item['evidence_bonus_applied']) for item in reward_details
            ) / count,
            'reward/joint_success_bonus_rate': sum(
                int(item['joint_success_bonus_applied'])
                for item in reward_details
            ) / count,
            **{
                f'reward/{name}': value
                for name, value in reward_group_metrics.items()
            },
        }

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, 
                              structure_format_score=config.reward_model.structure_format_score, 
                              final_format_score=config.reward_model.final_format_score,
                              retrieval_score=config.reward_model.retrieval_score,
                              reward_profile=config.reward_model.reward_profile,
                              think_format_score=config.reward_model.think_format_score,
                              answer_format_score=config.reward_model.answer_format_score,
                              evidence_score=config.reward_model.evidence_score,
                              answer_em_score=config.reward_model.answer_em_score,
                              joint_success_bonus=config.reward_model.joint_success_bonus)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    try:
        trainer.init_workers()
        trainer.fit()
    finally:
        # The format-reward driver is also a short-lived Ray task. Explicitly
        # close W&B so the last (or only) history row is uploaded before exit.
        trainer.logger.finish()


if __name__ == '__main__':
    main()
