#!/usr/bin/env bash
set -euo pipefail

PROJ=/DATA/storage/ab_students/align_gemini_project
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export HF_HOME="/DATA/storage/ab_students/.cache/huggingface"
export MODELSCOPE_CACHE="/DATA/storage/ab_students/.cache/modelscope"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"

VLM_MODEL="${VLM_MODEL:-$PROJ/weights/Qwen2.5-VL-7B-Instruct}"
SEMANTIC_DATASET="${SEMANTIC_DATASET:-$PROJ/data/semantic_preference_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ/output/vlm_dpo_adapter}"

[[ -f "$VLM_MODEL/config.json" ]] || { echo "Missing model: $VLM_MODEL" >&2; exit 1; }
[[ -f "$SEMANTIC_DATASET" ]] || { echo "Missing dataset: $SEMANTIC_DATASET" >&2; exit 1; }

echo "=== Qwen2.5-VL-7B DPO (paper settings, 1 GPU) ==="

swift rlhf \
  --rlhf_type dpo \
  --model "$VLM_MODEL" \
  --dataset "$SEMANTIC_DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --train_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_llm false \
  --freeze_vit true \
  --freeze_aligner true \
  --torch_dtype float16 \
  --max_pixels "$MAX_PIXELS" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-6 \
  --max_length 8192 \
  --beta 0.05 \
  --loss_type sigmoid \
  --rpo_alpha 0 \
  --gradient_checkpointing true \
  --save_steps 100 \
  --save_total_limit 3 \
  --logging_steps 5

echo "=== Training finished ==="
