#!/usr/bin/env bash
P=/DATA/storage/ab_students/align_gemini_project
cd $P
STEP=${STEP:-10}
GENS=${GENS:-"Nano flux-dev flux-kera flux-kontext flux-pro gpt4o jimeng keling minimax"}
OUT=eval/vlm_lp_s$STEP
mkdir -p $OUT lists/lp logs
export XDG_CACHE_HOME=/DATA/storage/ab_students/.cache
export HF_HOME=$XDG_CACHE_HOME/huggingface
export MODELSCOPE_CACHE=$XDG_CACHE_HOME/modelscope
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export TMPDIR=/DATA/storage/ab_students/tmp
export MAX_PIXELS=1003520
for g in $GENS; do
  L=lists/lp/${g}_s$STEP.txt
  awk -v s=$STEP '(NR-1)%s==0' lists/by_gen/$g.txt > $L
  EXP=$(wc -l < $L)
  R=$OUT/$g.result.jsonl
  if [ -f $R ] && [ "$(wc -l < $R)" -eq "$EXP" ]; then echo "skip $g"; continue; fi
  rm -f $R
  $P/venv_train/bin/python make_vlm_input.py $L $OUT/$g.input.jsonl || { echo "INPUT FAILED $g"; exit 1; }
  echo "start $g $(date) expected=$EXP"
  $P/venv_infer/bin/swift infer --adapters $P/weights/vlm_adapter --infer_backend vllm --vllm_tensor_parallel_size 1 --vllm_max_model_len 8192 --max_model_len 8192 --vllm_gpu_memory_utilization 0.85 --max_pixels 1003520 --write_batch_size 500 --temperature 0 --max_new_tokens 512 --logprobs true --top_logprobs 5 --val_dataset $P/$OUT/$g.input.jsonl --result_path $P/$R --seed 42 --model $P/weights/Qwen2.5-VL-7B-Instruct --load_args false > logs/vlm_lp_s${STEP}_$g.log 2>&1
  if [ -f $R ]; then N=$(wc -l < $R); else N=0; fi
  echo "end $g $(date) lines=$N expected=$EXP"
  if [ "$N" -ne "$EXP" ]; then echo "CHUNK FAILED $g (logs/vlm_lp_s${STEP}_$g.log dekho)"; exit 1; fi
done
echo "ALL DONE $(date)"
