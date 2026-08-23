#!/usr/bin/env bash
set -euo pipefail

# CA-ECAD V11.1 Phase 3: Search-R1 v0.1 outcome reward with CA-ECAD token
# credits in place of GRPO's group-whitened outcome advantage.  PPO clipping,
# reference-KL and AdamW remain the upstream Search-R1 path.

: "${STORAGE_ROOT:?Set STORAGE_ROOT to a writable data-disk directory.}"
: "${BASE_MODEL:?Set BASE_MODEL to the Step800 checkpoint selected for Phase 3.}"
INITIAL_SUCCESS_PRIOR_PATH="${INITIAL_SUCCESS_PRIOR_PATH:-}"
RESUME_STATE_PATH="${RESUME_STATE_PATH:-}"

NUM_GPUS="${NUM_GPUS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
N_AGENT="${N_AGENT:-5}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-16}"
VAL_DATA_NUM="${VAL_DATA_NUM:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-4}"
LOGPROB_MICRO_BATCH_SIZE="${LOGPROB_MICRO_BATCH_SIZE:-8}"
REF_LOGPROB_MICRO_BATCH_SIZE="${REF_LOGPROB_MICRO_BATCH_SIZE:-8}"
# CA-ECAD keeps its batch-level peer statistics on the CPU already.  Do not
# also force the actor's parameters, gradients, and AdamW state there: on a
# shared server that can exhaust host RAM long before either training GPU is
# full.  These switches remain overridable for genuinely GPU-constrained runs.
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-false}"
ACTOR_GRAD_OFFLOAD="${ACTOR_GRAD_OFFLOAD:-false}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-false}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-true}"
ACTOR_LR="${ACTOR_LR:-5e-7}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.003}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:--1}"
DATA_DIR="${DATA_DIR:-${STORAGE_ROOT}/data/nq_hotpotqa_train}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
VAL_FILE="${VAL_FILE:-${STORAGE_ROOT}/data/nq_search/test.parquet}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
WANDB_PROJECT="${WANDB_PROJECT:-CA-ECAD-Search-Agent}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ca-ecad-v11-1-phase3}"
CHECK_ONLY="${CHECK_ONLY:-false}"

if (( NUM_GPUS < 1 )); then
  echo "NUM_GPUS must be positive" >&2
  exit 2
fi
if (( TRAIN_BATCH_SIZE < 1 || TRAIN_BATCH_SIZE % NUM_GPUS != 0 )); then
  echo "TRAIN_BATCH_SIZE must be positive and divisible by NUM_GPUS" >&2
  exit 2
fi
if (( N_AGENT < 2 )); then
  echo "N_AGENT must be at least 2 for CA-ECAD peers" >&2
  exit 2
fi
if (( PPO_MINI_BATCH_SIZE < 1 || PPO_MICRO_BATCH_SIZE < 1 || PPO_MINI_BATCH_SIZE % PPO_MICRO_BATCH_SIZE != 0 )); then
  echo "PPO mini/micro batches must be positive and mini divisible by micro" >&2
  exit 2
fi
for setting in ACTOR_PARAM_OFFLOAD ACTOR_GRAD_OFFLOAD ACTOR_OPTIMIZER_OFFLOAD REF_PARAM_OFFLOAD; do
  value="${!setting}"
  if [[ "${value}" != "true" && "${value}" != "false" ]]; then
    echo "${setting} must be true or false, got: ${value}" >&2
    exit 2
  fi
done
for path in "${BASE_MODEL}" "${INITIAL_SUCCESS_PRIOR_PATH}" "${TRAIN_FILE}" "${VAL_FILE}"; do
  if [[ -n "${path}" && ! -e "${path}" ]]; then
    echo "Required path does not exist: ${path}" >&2
    exit 2
  fi
done

if [[ -n "${INITIAL_SUCCESS_PRIOR_PATH}" && -n "${RESUME_STATE_PATH}" ]] || \
   [[ -z "${INITIAL_SUCCESS_PRIOR_PATH}" && -z "${RESUME_STATE_PATH}" ]]; then
  echo "Set exactly one of INITIAL_SUCCESS_PRIOR_PATH or RESUME_STATE_PATH" >&2
  exit 2
fi
CA_ECAD_STATE_PATH="${RESUME_STATE_PATH:-${INITIAL_SUCCESS_PRIOR_PATH}}"
if [[ -n "${RESUME_STATE_PATH}" && ! -e "${RESUME_STATE_PATH}" ]]; then
  echo "Required path does not exist: ${RESUME_STATE_PATH}" >&2
  exit 2
fi

python3 - "${CA_ECAD_STATE_PATH}" "${RESUME_STATE_PATH:+resume}" <<'PY'
import json
import math
import sys

path = sys.argv[1]
resume = len(sys.argv) > 2 and sys.argv[2] == 'resume'
with open(path, 'r', encoding='utf-8') as handle:
    payload = json.load(handle)
if resume:
    if payload.get('version') != 1:
        raise SystemExit('CA-ECAD resume state has unsupported version')
else:
    if payload.get('ready_for_phase3_implementation') is not True:
        raise SystemExit('Phase-2 analysis is not marked ready_for_phase3_implementation=true')
prior = float(payload['success_prior'])
if not math.isfinite(prior) or not 0.0 <= prior <= 1.0:
    raise SystemExit('CA-ECAD success_prior must be finite and in [0, 1]')
for name, expected in {'alpha': 2.0, 'eta': 0.25, 'kappa': 1.0}.items():
    actual = float(payload.get('hyperparameters', {}).get(name, float('nan')))
    if actual != expected:
        raise SystemExit(f'CA-ECAD state hyperparameter mismatch for {name}: {actual} != {expected}')
print(f'CA-ECAD Phase-3 preflight passed: success_prior={prior:.8f}, resume={resume}')
PY

if [[ "${CHECK_ONLY}" == "true" ]]; then
  echo "CHECK_ONLY=true: validated inputs only; no Ray process or training was started."
  exit 0
fi

mkdir -p "${STORAGE_ROOT}/ray" "${STORAGE_ROOT}/hydra" "${STORAGE_ROOT}/wandb" \
  "${STORAGE_ROOT}/tmp" "${STORAGE_ROOT}/checkpoints/${EXPERIMENT_NAME}"
export RAY_TMPDIR="${STORAGE_ROOT}/ray"
export TMPDIR="${STORAGE_ROOT}/tmp"
export WANDB_DIR="${STORAGE_ROOT}/wandb"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

if [[ -n "${RESUME_STATE_PATH}" ]]; then
  CA_ECAD_STATE_OVERRIDES=(
    "algorithm.ca_ecad.initial_success_prior_path=null"
    "algorithm.ca_ecad.resume_state_path=${RESUME_STATE_PATH}"
  )
else
  CA_ECAD_STATE_OVERRIDES=(
    "algorithm.ca_ecad.initial_success_prior_path=${INITIAL_SUCCESS_PRIOR_PATH}"
    "algorithm.ca_ecad.resume_state_path=null"
  )
fi

python3 -m verl.trainer.main_ppo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_data_num="${TRAIN_DATA_NUM}" \
  data.val_data_num="${VAL_DATA_NUM}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length=8192 \
  data.max_response_length=500 \
  data.max_start_length=2048 \
  data.max_obs_length=1024 \
  data.shuffle_train_dataloader=true \
  algorithm.adv_estimator=grpo \
  algorithm.ca_ecad.diagnostics_only=false \
  algorithm.ca_ecad.enabled=true \
  "${CA_ECAD_STATE_OVERRIDES[@]}" \
  algorithm.ca_ecad.alpha=2.0 \
  algorithm.ca_ecad.eta=0.25 \
  algorithm.ca_ecad.kappa=1.0 \
  algorithm.ca_ecad.success_prior_rho=0.01 \
  algorithm.ca_ecad.mode_balance_gamma=0.0 \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.grad_offload="${ACTOR_GRAD_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.n_agent="${N_AGENT}" \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="${LOGPROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size="${REF_LOGPROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}" \
  actor_rollout_ref.actor.state_masking=true \
  reward_model.enable=false \
  trainer.logger="['wandb']" \
  +trainer.val_before_train=false \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${STORAGE_ROOT}/checkpoints/${EXPERIMENT_NAME}" \
  hydra.run.dir="${STORAGE_ROOT}/hydra/${EXPERIMENT_NAME}" \
  max_turns=4 \
  retriever.url="${RETRIEVER_URL}" \
  retriever.topk=3
