#!/usr/bin/env bash
P=/DATA/storage/ab_students/align_gemini_project
cd $P
export XDG_CACHE_HOME=/DATA/storage/ab_students/.cache
export HF_HOME=$XDG_CACHE_HOME/huggingface
export MODELSCOPE_CACHE=$XDG_CACHE_HOME/modelscope
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export TMPDIR=/DATA/storage/ab_students/tmp
export MAX_PIXELS=1003520
for g in Nano flux-dev flux-kera flux-kontext flux-pro gpt4o jimeng keling minimax; do
  R=eval/vlm_full/$g.result.jsonl
  if [ -f $R ] && [ "$(wc -l < $R)" -eq 4000 ]; then echo "skip $g"; continue; fi
  rm -f $R
  $P/venv_train/bin/python make_vlm_input.py lists/by_gen/$g.txt eval/vlm_full/$g.input.jsonl || { echo "INPUT FAILED $g"; exit 1; }
  echo "start $g $(date)"
  $P/venv_infer/bin/swift infer --adapters $P/weights/vlm_adapter --infer_backend vllm --vllm_tensor_parallel_size 1 --vllm_max_model_len 8192 --max_model_len 8192 --vllm_gpu_memory_utilization 0.85 --max_pixels 1003520 --write_batch_size 500 --temperature 0 --max_new_tokens 512 --val_dataset $P/eval/vlm_full/$g.input.jsonl --result_path $P/$R --seed 42 --model $P/weights/Qwen2.5-VL-7B-Instruct --load_args false > logs/vlm_full_$g.log 2>&1
  if [ -f $R ]; then N=$(wc -l < $R); else N=0; fi
  echo "end $g $(date) lines=$N"
  if [ "$N" -ne 4000 ]; then echo "CHUNK FAILED $g"; exit 1; fi
done
echo "ALL DONE $(date)"
