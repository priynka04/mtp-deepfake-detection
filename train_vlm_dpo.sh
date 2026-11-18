#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

export HF_HOME="/DATA/storage/ab_students/.cache/huggingface"
export MODELSCOPE_CACHE="/DATA/storage/ab_students/.cache/modelscope"

echo "=== Starting Branch 1: Qwen2.5-VL-7B DPO Training via ms-swift ==="

swift dpo \
    --model_type qwen2_5-vl-7b-instruct \
    --model_id_or_path weights/Qwen2.5-VL-7B-Instruct \
    --custom_train_dataset_path data/vlm_dpo_train.jsonl \
    --sft_type lora \
    --lora_target_modules QKV ALL \
    --dpo_beta 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.05 \
    --num_train_epochs 3 \
    --max_length 2048 \
    --output_dir output/vlm_dpo_adapter \
    --gradient_checkpointing true \
    --use_vllm false

echo "=== VLM Branch Training Finished ==="
