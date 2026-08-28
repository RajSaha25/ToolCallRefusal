#!/bin/bash
# Finish every remaining GPU run in one pass, sequentially (one model fits the GPU
# at a time). Each step waits for the GPU to actually drain before the next loads --
# two models loading at once already caused one OOM.
cd /workspace/tcr || exit 1
export HF_HOME=/workspace/hf HF_TOKEN=$(cat .hftoken)

wait_gpu() {
  for _ in $(seq 1 90); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$u" -lt 2000 ] && { echo "[chain] gpu free (${u} MiB)"; return 0; }
    sleep 10
  done
  echo "[chain] WARNING gpu still busy (${u} MiB), proceeding anyway"
}

step() {  # step <label> <logfile> <cmd...>
  local label=$1 logf=$2; shift 2
  echo "[chain] START $label"
  wait_gpu
  "$@" > "$logf" 2>&1
  echo "[chain] DONE $label rc=$?"
}

# 1. Command-R: its GPU stage may already be running from outside this script
while pgrep -f "rerun_against_relabels.py --stage gpu --model c4ai" > /dev/null; do sleep 15; done
echo "[chain] command-r gpu stage finished"

step "command-r steering" /workspace/tcr/steer_cmdr.log \
  python3 -u rerun_steering_gen.py --model c4ai-command-r7b-12-2024 \
    --hf-id CohereLabs/c4ai-command-r7b-12-2024 --layer 26 --auto-grid --bs 24

# 2. Mistral: needs a GPU stage first, only to record proj_gap
step "mistral gpu stage" /workspace/tcr/gpu_mistral2.log \
  python3 -u rerun_against_relabels.py --stage gpu --model Mistral-7B-Instruct-v0.3 \
    --hf-id mistralai/Mistral-7B-Instruct-v0.3 --layer 26 --batch 8

step "mistral steering" /workspace/tcr/steer_mistral2.log \
  python3 -u rerun_steering_gen.py --model Mistral-7B-Instruct-v0.3 \
    --hf-id mistralai/Mistral-7B-Instruct-v0.3 --layer 26 --auto-grid --bs 24

echo "[chain] ALL DONE"
