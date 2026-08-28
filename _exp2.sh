#!/bin/bash
# Experiment 2: rerun ablation / addition / steering at doubled n for tighter CIs.
# The volume cannot hold all five model caches at once, so each model's weights are
# dropped once its run finishes. Re-downloads are fast on this volume (~1 min/15GB).
cd /workspace/tcr || exit 1
export HF_HOME=/workspace/hf HF_TOKEN=$(cat .hftoken 2>/dev/null)

wait_gpu() {
  for _ in $(seq 1 180); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$u" -lt 2000 ] && return 0
    sleep 10
  done
}

run() {  # run <model> <hf-id> <layer> <bs> <cache-dir-to-drop-after>
  echo "[exp2] START $1"
  wait_gpu
  python3 -u rerun_steering_gen.py --model "$1" --hf-id "$2" --layer "$3" \
    --auto-grid --n-abl 240 --n-add 240 --n-steer-h 200 --n-steer-b 120 --bs "$4" \
    > "/workspace/tcr/exp2_$1.log" 2>&1
  echo "[exp2] DONE $1 rc=$?"
  grep -aE "^\[steer\]" "/workspace/tcr/exp2_$1.log" | tail -4
  df -h /workspace | tail -1
  [ -n "$5" ] && rm -rf "/workspace/hf/hub/$5"
}

# Llama is already running outside this script; wait it out and drop its weights.
while pgrep -f "rerun_steering_gen.py --model Meta-Llama" > /dev/null; do sleep 20; done
echo "[exp2] llama finished"
rm -rf /workspace/hf/hub/models--unsloth--Meta-Llama-3.1-70B-Instruct
df -h /workspace | tail -1

run gemma-3-27b-it           google/gemma-3-27b-it              51 16 models--google--gemma-3-27b-it
run Qwen3-14B                Qwen/Qwen3-14B                     33 24 models--Qwen--Qwen3-14B
run c4ai-command-r7b-12-2024 CohereLabs/c4ai-command-r7b-12-2024 26 24 models--CohereLabs--c4ai-command-r7b-12-2024
run Mistral-7B-Instruct-v0.3 mistralai/Mistral-7B-Instruct-v0.3 26 24 ""
echo "[exp2] ALL DONE"
