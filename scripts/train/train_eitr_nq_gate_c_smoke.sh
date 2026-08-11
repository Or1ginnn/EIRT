#!/usr/bin/env bash
set -euo pipefail

# Phase-2 Conditional EITR smoke. Defaults target the A800 data disk and leave
# GPU 0 untouched.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"

NUM_GPUS="${NUM_GPUS:-2}"
STORAGE_ROOT="${STORAGE_ROOT:-/mnt/data1/zar/eitr_storage}"
STORAGE_ROOT="$(realpath -m "$STORAGE_ROOT")"
ALLOW_NON_DATA_DISK="${ALLOW_NON_DATA_DISK:-false}"
case "$STORAGE_ROOT" in
    /mnt/data1/zar|/mnt/data1/zar/*) ;;
    *)
        if [[ "$ALLOW_NON_DATA_DISK" != "true" ]]; then
            echo "STORAGE_ROOT must stay on /mnt/data1/zar; got: $STORAGE_ROOT" >&2
            exit 2
        fi
        ;;
esac
LEGACY_DATA_DIR="${DATA_DIR:-}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-${LEGACY_DATA_DIR:-$STORAGE_ROOT/data/nq_search}}"
VAL_DATA_DIR="${VAL_DATA_DIR:-${LEGACY_DATA_DIR:-$STORAGE_ROOT/data/nq_search}}"
BASE_MODEL="${BASE_MODEL:-$STORAGE_ROOT/models/Qwen2.5-3B}"
SAVE_FULL_CHECKPOINT="${SAVE_FULL_CHECKPOINT:-false}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-null}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
case "$SAVE_FULL_CHECKPOINT" in
    true|false) ;;
    *)
        echo "SAVE_FULL_CHECKPOINT must be true or false; got: $SAVE_FULL_CHECKPOINT" >&2
        exit 2
        ;;
esac
if [[ "$RESUME_FROM_CHECKPOINT" != "null" ]]; then
    if [[ ! -f "$RESUME_FROM_CHECKPOINT/trainer_state/driver_state.pt" ]]; then
        echo "Resume requires a full checkpoint: $RESUME_FROM_CHECKPOINT" >&2
        exit 2
    fi
    BASE_MODEL="$RESUME_FROM_CHECKPOINT"
fi
if [[ -z "${EITR_MODE+x}" && -n "${EITR_ENABLED+x}" ]]; then
    if [[ "$EITR_ENABLED" == "true" ]]; then
        EITR_MODE="eitr"
    else
        EITR_MODE="off"
    fi
fi
EITR_MODE="${EITR_MODE:-eitr}"
EITR_PROBE_PROBABILITY="${EITR_PROBE_PROBABILITY:-1.0}"
EITR_PROBE_COUNT="${EITR_PROBE_COUNT:-4}"
EITR_MIN_VALID_PROBE_COUNT="${EITR_MIN_VALID_PROBE_COUNT:-2}"
EITR_LAMBDA_ENV="${EITR_LAMBDA_ENV:-0.1}"
EITR_CORRECTION_PASSES="${EITR_CORRECTION_PASSES:-1}"
EITR_PROBE_MICRO_BATCH_SIZE="${EITR_PROBE_MICRO_BATCH_SIZE:-4}"
EITR_PROBE_LOGPROB_MICRO_BATCH_SIZE="${EITR_PROBE_LOGPROB_MICRO_BATCH_SIZE:-4}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-1024}"
MAX_TRAJECTORY_LENGTH="${MAX_TRAJECTORY_LENGTH:-8192}"
EITR_MAX_QUERY_TOKENS="${EITR_MAX_QUERY_TOKENS:-$MAX_RESPONSE_LENGTH}"
# Search-R1 v0.3 GRPO uses 5e-7 (v0.2 used 1e-6).
ACTOR_LR="${ACTOR_LR:-5e-7}"
EITR_LR="${EITR_LR:-$ACTOR_LR}"
EITR_POST_DIAGNOSTIC_FREQ="${EITR_POST_DIAGNOSTIC_FREQ:-1}"
EITR_SAME_BATCH_SCALE_DIAGNOSTIC="${EITR_SAME_BATCH_SCALE_DIAGNOSTIC:-false}"
EITR_UPDATE_DIRECTION_DIAGNOSTIC="${EITR_UPDATE_DIRECTION_DIAGNOSTIC:-false}"
EITR_SCORE_PATH_NOOP_DIRECTION_AUDIT="${EITR_SCORE_PATH_NOOP_DIRECTION_AUDIT:-false}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.003}"
STRUCTURE_FORMAT_SCORE="${STRUCTURE_FORMAT_SCORE:-0.2}"
FINAL_FORMAT_SCORE="${FINAL_FORMAT_SCORE:-0.1}"
RETRIEVAL_SCORE="${RETRIEVAL_SCORE:-0}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.285}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eitr-nq-phase2-smoke}"
WANDB_PROJECT="${WANDB_PROJECT:-EITR-Search-Agent}"
METRICS_LEVEL="${METRICS_LEVEL:-debug}"
# With an explicit TOTAL_TRAINING_STEPS budget, the trainer cycles the
# dataloader until that exact number of outer updates is complete.  Keep the
# smoke epoch default aligned as a readable fallback for its one-batch loader.
TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"
TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-32}"
VAL_DATA_NUM="${VAL_DATA_NUM:-64}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
SHUFFLE_TRAIN_DATALOADER="${SHUFFLE_TRAIN_DATALOADER:-false}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-16}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-32}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-32}"
N_AGENT="${N_AGENT:-5}"
ACTOR_FSDP_PARAM_OFFLOAD="${ACTOR_FSDP_PARAM_OFFLOAD:-false}"
ACTOR_FSDP_OPTIMIZER_OFFLOAD="${ACTOR_FSDP_OPTIMIZER_OFFLOAD:-false}"
ACTOR_BATCH_OFFLOAD="${ACTOR_BATCH_OFFLOAD:-false}"
REF_FSDP_PARAM_OFFLOAD="${REF_FSDP_PARAM_OFFLOAD:-false}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_PROBE_PROMPT_TOKENS="${MAX_PROBE_PROMPT_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$((MAX_PROBE_PROMPT_TOKENS + EITR_MAX_QUERY_TOKENS))}"
INFORMATIVE_JS_THRESHOLD="${INFORMATIVE_JS_THRESHOLD:-0.01}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:-5}"
CHECK_ONLY="${CHECK_ONLY:-false}"

case "$EITR_MODE" in
    off|probe_only|eitr) ;;
    *)
        echo "EITR_MODE must be one of: off, probe_only, eitr; got: $EITR_MODE" >&2
        exit 2
        ;;
esac

case "$SHUFFLE_TRAIN_DATALOADER" in
    true|false) ;;
    *)
        echo "SHUFFLE_TRAIN_DATALOADER must be true or false; got: $SHUFFLE_TRAIN_DATALOADER" >&2
        exit 2
        ;;
esac

for offload_value in \
    "$ACTOR_FSDP_PARAM_OFFLOAD" \
    "$ACTOR_FSDP_OPTIMIZER_OFFLOAD" \
    "$ACTOR_BATCH_OFFLOAD" \
    "$REF_FSDP_PARAM_OFFLOAD"; do
    case "$offload_value" in
        true|false) ;;
        *)
            echo "FSDP param offload flags must be true or false; got: $offload_value" >&2
            exit 2
            ;;
    esac
done

case "$METRICS_LEVEL" in
    core|debug) ;;
    *)
        echo "METRICS_LEVEL must be core or debug; got: $METRICS_LEVEL" >&2
        exit 2
        ;;
esac

case "$EITR_SAME_BATCH_SCALE_DIAGNOSTIC" in
    true|false) ;;
    *)
        echo "EITR_SAME_BATCH_SCALE_DIAGNOSTIC must be true or false; got: $EITR_SAME_BATCH_SCALE_DIAGNOSTIC" >&2
        exit 2
        ;;
esac

case "$EITR_UPDATE_DIRECTION_DIAGNOSTIC" in
    true|false) ;;
    *)
        echo "EITR_UPDATE_DIRECTION_DIAGNOSTIC must be true or false; got: $EITR_UPDATE_DIRECTION_DIAGNOSTIC" >&2
        exit 2
        ;;
esac
case "$EITR_SCORE_PATH_NOOP_DIRECTION_AUDIT" in
    true|false) ;;
    *)
        echo "EITR_SCORE_PATH_NOOP_DIRECTION_AUDIT must be true or false; got: $EITR_SCORE_PATH_NOOP_DIRECTION_AUDIT" >&2
        exit 2
        ;;
esac
if [[ $(printf '%s\n' "$EITR_SAME_BATCH_SCALE_DIAGNOSTIC" "$EITR_UPDATE_DIRECTION_DIAGNOSTIC" "$EITR_SCORE_PATH_NOOP_DIRECTION_AUDIT" | grep -c '^true$') -gt 1 ]]; then
    echo "Only one EITR diagnostic may be enabled at once" >&2
    exit 2
fi
if [[ "$EITR_SCORE_PATH_NOOP_DIRECTION_AUDIT" == true && "$METRICS_LEVEL" != debug ]]; then
    echo "EITR score-path audit requires METRICS_LEVEL=debug so all scientific diagnostics are persisted" >&2
    exit 2
fi

if (( EITR_PROBE_LOGPROB_MICRO_BATCH_SIZE != EITR_PROBE_MICRO_BATCH_SIZE )); then
    echo "EITR_PROBE_LOGPROB_MICRO_BATCH_SIZE must equal EITR_PROBE_MICRO_BATCH_SIZE for invariant probe scoring" >&2
    exit 2
fi
if (( EITR_PROBE_MICRO_BATCH_SIZE % EITR_PROBE_COUNT != 0 )); then
    echo "EITR_PROBE_MICRO_BATCH_SIZE must be divisible by EITR_PROBE_COUNT" >&2
    exit 2
fi

for positive_integer in \
    "$NUM_GPUS" \
    "$TRAIN_BATCH_SIZE" \
    "$PPO_MINI_BATCH_SIZE" \
    "$PPO_MICRO_BATCH_SIZE" \
    "$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE" \
    "$REF_LOG_PROB_MICRO_BATCH_SIZE" \
    "$N_AGENT"; do
    if [[ ! "$positive_integer" =~ ^[1-9][0-9]*$ ]]; then
        echo "GPU and batch-size settings must be positive integers; got: $positive_integer" >&2
        exit 2
    fi
done

IFS=',' read -r -a visible_gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#visible_gpu_ids[@]} != NUM_GPUS )); then
    echo "NUM_GPUS=$NUM_GPUS does not match CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    exit 2
fi
declare -A seen_gpu_ids=()
for gpu_id in "${visible_gpu_ids[@]}"; do
    gpu_id="${gpu_id//[[:space:]]/}"
    if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
        echo "CUDA_VISIBLE_DEVICES must contain physical numeric GPU ids; got: $gpu_id" >&2
        exit 2
    fi
    if [[ "$gpu_id" == "0" ]]; then
        echo "Physical GPU0 is reserved and must not be used" >&2
        exit 2
    fi
    if [[ -n "${seen_gpu_ids[$gpu_id]+x}" ]]; then
        echo "CUDA_VISIBLE_DEVICES contains duplicate GPU id: $gpu_id" >&2
        exit 2
    fi
    seen_gpu_ids[$gpu_id]=1
done

for divisible_setting in \
    "$TRAIN_BATCH_SIZE" \
    "$PPO_MINI_BATCH_SIZE" \
    "$PPO_MICRO_BATCH_SIZE" \
    "$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE" \
    "$REF_LOG_PROB_MICRO_BATCH_SIZE"; do
    if (( divisible_setting % NUM_GPUS != 0 )); then
        echo "Three-card/data-parallel batch settings must be divisible by NUM_GPUS=$NUM_GPUS; got: $divisible_setting" >&2
        exit 2
    fi
done
if (( PPO_MINI_BATCH_SIZE % PPO_MICRO_BATCH_SIZE != 0 )); then
    echo "PPO_MINI_BATCH_SIZE must be divisible by PPO_MICRO_BATCH_SIZE" >&2
    exit 2
fi
rollout_batch_size=$((TRAIN_BATCH_SIZE * N_AGENT))
if (( rollout_batch_size % PPO_MINI_BATCH_SIZE != 0 )); then
    echo "TRAIN_BATCH_SIZE*N_AGENT=$rollout_batch_size must be divisible by PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE" >&2
    exit 2
fi

case "$BASE_MODEL" in
    *parallel_search*|*parallel-search*|*finance*)
        echo "Phase 2 requires a clean single-query Search-R1 initialization, got: $BASE_MODEL" >&2
        exit 2
        ;;
esac

if [[ ! -e "$BASE_MODEL" ]]; then
    echo "Phase 2 model path does not exist: $BASE_MODEL" >&2
    exit 2
fi

if [[ ! -f "$TRAIN_DATA_DIR/train.parquet" ]]; then
    echo "Phase 2 training data is missing: $TRAIN_DATA_DIR/train.parquet" >&2
    exit 2
fi
if [[ ! -f "$VAL_DATA_DIR/test.parquet" ]]; then
    echo "Phase 2 validation data is missing: $VAL_DATA_DIR/test.parquet" >&2
    exit 2
fi

if (( MAX_PROBE_PROMPT_TOKENS < MAX_PROMPT_LENGTH )); then
    echo "MAX_PROBE_PROMPT_TOKENS must be >= MAX_PROMPT_LENGTH for exact same-state probes" >&2
    exit 2
fi

if (( EITR_MAX_QUERY_TOKENS != MAX_RESPONSE_LENGTH )); then
    echo "EITR_MAX_QUERY_TOKENS must equal MAX_RESPONSE_LENGTH for same-policy query probes" >&2
    exit 2
fi

if (( MAX_TRAJECTORY_LENGTH < MAX_RESPONSE_LENGTH )); then
    echo "MAX_TRAJECTORY_LENGTH must be >= MAX_RESPONSE_LENGTH" >&2
    exit 2
fi

if (( VLLM_MAX_MODEL_LEN < MAX_PROBE_PROMPT_TOKENS + EITR_MAX_QUERY_TOKENS )); then
    echo "VLLM_MAX_MODEL_LEN must cover exact probe state + query continuation" >&2
    exit 2
fi
if (( VLLM_MAX_MODEL_LEN < MAX_PROMPT_LENGTH + 500 )); then
    echo "VLLM_MAX_MODEL_LEN must cover the ordinary rollout prompt + response" >&2
    exit 2
fi

# Ray appends its own session/socket suffix. Keep this base path short because
# Linux AF_UNIX socket paths are limited to 107 bytes; experiment names belong
# in Hydra/log/checkpoint paths, not in RAY_TMPDIR.
export RAY_TMPDIR="${RAY_TMPDIR:-$STORAGE_ROOT/r}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$STORAGE_ROOT/checkpoints/$EXPERIMENT_NAME}"
LOG_DIR="${LOG_DIR:-$STORAGE_ROOT/logs}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$STORAGE_ROOT/hydra/$EXPERIMENT_NAME}"
export HF_HOME="${HF_HOME:-$STORAGE_ROOT/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$STORAGE_ROOT/cache/xdg}"
export TORCH_HOME="${TORCH_HOME:-$STORAGE_ROOT/cache/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$STORAGE_ROOT/cache/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$STORAGE_ROOT/cache/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$STORAGE_ROOT/cache/cuda}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$STORAGE_ROOT/cache/numba}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$STORAGE_ROOT/cache/pycache}"
export WANDB_DIR="${WANDB_DIR:-$STORAGE_ROOT/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$STORAGE_ROOT/cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$STORAGE_ROOT/wandb/config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$STORAGE_ROOT/wandb/data}"
export TMPDIR="${TMPDIR:-$STORAGE_ROOT/tmp}"

for writable_path in \
    "$RAY_TMPDIR" \
    "$CHECKPOINT_DIR" \
    "$LOG_DIR" \
    "$HYDRA_RUN_DIR" \
    "$HF_HOME" \
    "$TRANSFORMERS_CACHE" \
    "$XDG_CACHE_HOME" \
    "$TORCH_HOME" \
    "$TORCH_EXTENSIONS_DIR" \
    "$TRITON_CACHE_DIR" \
    "$CUDA_CACHE_PATH" \
    "$NUMBA_CACHE_DIR" \
    "$PYTHONPYCACHEPREFIX" \
    "$WANDB_DIR" \
    "$WANDB_CACHE_DIR" \
    "$WANDB_CONFIG_DIR" \
    "$WANDB_DATA_DIR" \
    "$TMPDIR"; do
    resolved_writable_path="$(realpath -m "$writable_path")"
    case "$resolved_writable_path" in
        "$STORAGE_ROOT"|"$STORAGE_ROOT"/*) ;;
        *)
            echo "Refusing writable path outside STORAGE_ROOT=$STORAGE_ROOT: $resolved_writable_path" >&2
            exit 2
            ;;
    esac
done

mkdir -p \
    "$RAY_TMPDIR" \
    "$CHECKPOINT_DIR" \
    "$LOG_DIR" \
    "$HYDRA_RUN_DIR" \
    "$HF_HOME" \
    "$TRANSFORMERS_CACHE" \
    "$XDG_CACHE_HOME" \
    "$TORCH_HOME" \
    "$TORCH_EXTENSIONS_DIR" \
    "$TRITON_CACHE_DIR" \
    "$CUDA_CACHE_PATH" \
    "$NUMBA_CACHE_DIR" \
    "$PYTHONPYCACHEPREFIX" \
    "$WANDB_DIR" \
    "$WANDB_CACHE_DIR" \
    "$WANDB_CONFIG_DIR" \
    "$WANDB_DATA_DIR" \
    "$TMPDIR"

if [[ "$CHECK_ONLY" == "true" ]]; then
    curl --fail --silent --show-error \
        --header 'Content-Type: application/json' \
        --data '{"queries":["who wrote Hamlet"],"topk":3,"return_scores":true}' \
        "$RETRIEVER_URL" >/dev/null
    echo "Phase 2 preflight passed: train=$TRAIN_DATA_DIR/train.parquet, val=$VAL_DATA_DIR/test.parquet, shuffle=$SHUFFLE_TRAIN_DATALOADER, model, data-disk write paths, exact-state limit, and retriever are ready."
    exit 0
fi

# Search-R1 v0.3 reward: answer EM plus trajectory/final-answer format shaping.
# Validation remains pure answer EM because main_ppo_format intentionally
# constructs its validation RewardManager without the training shaping scores.
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_format \
    data.train_files="$TRAIN_DATA_DIR/train.parquet" \
    data.val_files="$VAL_DATA_DIR/test.parquet" \
    data.train_data_num="$TRAIN_DATA_NUM" \
    data.val_data_num="$VAL_DATA_NUM" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_start_length=2048 \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.max_obs_length="$MAX_OBS_LENGTH" \
    data.max_trajectory_length="$MAX_TRAJECTORY_LENGTH" \
    data.shuffle_train_dataloader="$SHUFFLE_TRAIN_DATALOADER" \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$LR_WARMUP_STEPS_RATIO" \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size="$PPO_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_FSDP_PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_FSDP_OPTIMIZER_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.batch_offload="$ACTOR_BATCH_OFFLOAD" \
    actor_rollout_ref.actor.eitr.mode="$EITR_MODE" \
    actor_rollout_ref.actor.eitr.probe_source=online_same_state \
    actor_rollout_ref.actor.eitr.probe_probability="$EITR_PROBE_PROBABILITY" \
    actor_rollout_ref.actor.eitr.probe_count="$EITR_PROBE_COUNT" \
    actor_rollout_ref.actor.eitr.min_valid_probe_count="$EITR_MIN_VALID_PROBE_COUNT" \
    actor_rollout_ref.actor.eitr.probe_oversample=0 \
    actor_rollout_ref.actor.eitr.probe_micro_batch_size="$EITR_PROBE_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.eitr.probe_logprob_micro_batch_size="$EITR_PROBE_LOGPROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.eitr.max_query_tokens="$EITR_MAX_QUERY_TOKENS" \
    actor_rollout_ref.actor.eitr.probe_seed=20260805 \
    actor_rollout_ref.actor.eitr.max_action_tokens="$EITR_MAX_QUERY_TOKENS" \
    actor_rollout_ref.actor.eitr.max_probe_prompt_tokens="$MAX_PROBE_PROMPT_TOKENS" \
    actor_rollout_ref.actor.eitr.retrieval_score_temperature=0.1 \
    actor_rollout_ref.actor.eitr.min_state_coverage=0.0 \
    actor_rollout_ref.actor.eitr.informative_js_threshold="$INFORMATIVE_JS_THRESHOLD" \
    actor_rollout_ref.actor.eitr.min_informative_state_rate=0.0 \
    actor_rollout_ref.actor.eitr.correction_passes="$EITR_CORRECTION_PASSES" \
    actor_rollout_ref.actor.eitr.correction_optimizer=sgd \
    actor_rollout_ref.actor.eitr.correction_lr="$EITR_LR" \
    actor_rollout_ref.actor.eitr.post_diagnostic_freq="$EITR_POST_DIAGNOSTIC_FREQ" \
    actor_rollout_ref.actor.eitr.same_batch_scale_diagnostic="$EITR_SAME_BATCH_SCALE_DIAGNOSTIC" \
    actor_rollout_ref.actor.eitr.update_direction_diagnostic="$EITR_UPDATE_DIRECTION_DIAGNOSTIC" \
    actor_rollout_ref.actor.eitr.score_path_noop_direction_audit="$EITR_SCORE_PATH_NOOP_DIRECTION_AUDIT" \
    actor_rollout_ref.actor.eitr.lambda_env="$EITR_LAMBDA_ENV" \
    actor_rollout_ref.actor.eitr.log_ratio_clip=10.0 \
    actor_rollout_ref.actor.ppo_epochs="$PPO_EPOCHS" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len="$VLLM_MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size="$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent="$N_AGENT" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size="$REF_LOG_PROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.ref.fsdp_config.param_offload="$REF_FSDP_PARAM_OFFLOAD" \
    trainer.logger="['console','wandb']" \
    trainer.metrics_level="$METRICS_LEVEL" \
    +trainer.val_before_train=false \
    +trainer.val_only=false \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_full_checkpoint="$SAVE_FULL_CHECKPOINT" \
    trainer.resume_from_checkpoint="$RESUME_FROM_CHECKPOINT" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    reward_model.structure_format_score="$STRUCTURE_FORMAT_SCORE" \
    reward_model.final_format_score="$FINAL_FORMAT_SCORE" \
    reward_model.retrieval_score="$RETRIEVAL_SCORE" \
    hydra.run.dir="$HYDRA_RUN_DIR" \
    hydra.job.chdir=false \
    max_turns=4 \
    retriever.url="$RETRIEVER_URL" \
    retriever.topk=3 \
    2>&1 | tee "$LOG_DIR/$EXPERIMENT_NAME.log"
