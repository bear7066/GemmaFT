# Be sure to use bf16
# import transformers
# transformers.utils.import_utils.is_torch_bf16_gpu_available()

set -e  # 任何指令失敗即停止

# change this for RTX5090
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_NAME="${MODEL_NAME:-google/gemma-3-4b-it}"
DATA_PATH="${DATA_PATH:-/path/to/your/data.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/path/to/your/images}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/gemma3_stage1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-deepspeed_config/stage1.json}"

NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

uv run deepspeed \
    --num_gpus "$NUM_GPUS" \
    --master_port "$MASTER_PORT" \
    stage1/train.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    \
    --model_id "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    \
    --bf16 True \
    --tf32 False \
    \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --optim "paged_adamw_8bit" \
    \
    --learning_rate 1e-5 \
    --vision_lr 2e-5 \
    --projector_lr 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    --report_to "none"
