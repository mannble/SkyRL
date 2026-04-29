set -ex

# Harbor CodeContests training with Qwen3.5-35B-A3B on Megatron.
# Compared with:
# - examples/train_integrations/harbor/run_codecontest.sh
# - examples/train/megatron/run_megatron_qwen3.5_35b_a3b.sh
# - examples/train/megatron/run_search_megatron.sh for batched=false multi-turn settings

# wandb api key.
# export WANDB_API_KEY=YOUR_KEY_HERE
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DAYTONA_API_KEY=dtn_42e241b88dd48c74e41b12639ddcfd0a90bf73d3de45ae8180eef09d5adacf95
export WANDB_API_KEY=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
# Pick the sandbox provider and provide the credentials.
# export DAYTONA_API_KEY=YOUR_KEY_HERE
# export MODAL_TOKEN_ID=YOUR_KEY_HERE
# export MODAL_TOKEN_SECRET=YOUR_KEY_HERE

#-----------------------
# Dataset setup
#-----------------------
# Prepare datasets first (downloads from HuggingFace and extracts tasks):
# uv run examples/train_integrations/harbor/prepare_harbor_dataset.py --dataset open-thoughts/CodeContests
# uv run examples/train_integrations/harbor/prepare_harbor_dataset.py --dataset open-thoughts/OpenThoughts-TB-dev
DATA_DIR="$HOME/data/harbor"
TRAIN_DATA="['$DATA_DIR/endless-terminal-docker']"
EVAL_DATA="['$DATA_DIR/OpenThoughts-TB-dev']"

#-----------------------
# Directory setup
#-----------------------
MODEL_NAME="/home/test/test1714/wxh/Qwen3.5-9B"
SERVED_MODEL_NAME="Qwen3.5-9B"
RUN_NAME="ET-meta4-terminus2-qwen3_5_9b_megatron"
TRIALS_DIR="/home/test/test1714/wxh/skyrl/$RUN_NAME/trials_run"
CKPTS_DIR="/home/test/test1714/wxh/skyrl/$RUN_NAME/ckpts"
EXPORTS_DIR="/home/test/test1714/wxh/skyrl/$RUN_NAME/exports"
LOG_DIR="/tmp/skyrl-logs/$RUN_NAME"

#-----------------------
# Training setup
#-----------------------
MINI_BATCH_SIZE=32
MAX_MODEL_LEN=32768
APPLY_OVERLONG_FILTERING=true

# Dr. GRPO parameters
LOSS_REDUCTION="seq_mean_token_sum_norm"
GRPO_NORM_BY_STD=false
USE_KL_LOSS=false

# Qwen3.5 Megatron note: sample packing is not yet supported for GDN layers.
USE_SAMPLE_PACKING=false

# Keep thinking tokens in the replayed chat history, matching the existing Harbor script.
CHAT_TEMPLATE_PATH="$(dirname "$0")/../../../skyrl/train/utils/templates/qwen3_acc_thinking.jinja2"

#----------------
# Infrastructure setup
#----------------
NUM_NODES=1
NUM_GPUS=4

MEGATRON_TP=4      # 张量并行度（受限于 num_query_groups=4）
MEGATRON_PP=1      # 流水线并行度（PP>1 与 Megatron RoPE 动态序列长度不兼容）
MEGATRON_CP=1      # 上下文并行度
MEGATRON_EP=1      # 专家并行度（稠密模型必须设为1！）
MEGATRON_ETP=1     # 专家张量并行度

NUM_INFERENCE_ENGINES=1
INFERENCE_ENGINE_TP=4

OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

ENABLE_RATE_LIMITING=true
TRAJECTORIES_PER_SECOND=5
MAX_CONCURRENCY=32

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(dirname "$0")/../../../skyrl-agent"

uv run --isolated --extra megatron --extra harbor -m examples.train_integrations.harbor.entrypoints.main_harbor \
  data.train_data=$TRAIN_DATA \
  data.val_data=$EVAL_DATA \
  trainer.policy.model.path=$MODEL_NAME \
  generator.inference_engine.served_model_name=$SERVED_MODEL_NAME \
  harbor_trial_config.agent.name=terminus-2 \
  harbor_trial_config.trials_dir=$TRIALS_DIR \
  trainer.export_path=$EXPORTS_DIR \
  trainer.ckpt_path=$CKPTS_DIR \
  trainer.log_path=$LOG_DIR \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.grpo_norm_by_std=$GRPO_NORM_BY_STD \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.placement.colocate_all=false \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.use_sample_packing=$USE_SAMPLE_PACKING \
  generator.inference_engine.engine_init_kwargs.chat_template=$CHAT_TEMPLATE_PATH \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  trainer.epochs=3 \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=false \
  trainer.eval_interval=20 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$MINI_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=5 \
  trainer.hf_save_interval=5 \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=4 \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  trainer.logger=wandb \
  trainer.project_name=meta \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=latest \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.enable_http_endpoint=true \
  generator.inference_engine.http_endpoint_host=127.0.0.1 \
  generator.inference_engine.http_endpoint_port=8000 \
  generator.rate_limit.enabled=$ENABLE_RATE_LIMITING \
  generator.rate_limit.trajectories_per_second=$TRAJECTORIES_PER_SECOND \
  generator.rate_limit.max_concurrency=$MAX_CONCURRENCY \
  generator.meta.enabled=true \
  generator.meta.interval_batches=2 \
  generator.meta.max_candidates=2 \
  generator.meta.canary_num_tasks=16 \
  generator.meta.canary_n_samples=4 \
  generator.meta.override_base=/home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2 \
  generator.meta.log_dir=/tmp/skyrl-logs \
  generator.meta.llm_model=${META_LLM_MODEL:-$SERVED_MODEL_NAME} \
  generator.meta.llm_base_url=${META_LLM_BASE_URL:-"http://127.0.0.1:8000/v1"} \
  generator.meta.llm_api_key=${META_LLM_API_KEY:-"EMPTY"} \
  generator.meta.llm_temperature=0.7 \
  generator.meta.llm_max_tokens=4096 \
  "$@"