#!/bin/bash
# Patching + suppression rerun for all five families, unattended.
#
# Assumes: repo at /workspace/tcr, dataset in data/, directions_*.pt present in
# relabel_analysis/, HF token in .hftoken, and enough volume for all five model
# caches (~241GB of weights; 400GB volume recommended so no cleanup is needed).
#
#   bash run_patching_all.sh 2>&1 | tee patching_all.log
set -u
cd /workspace/tcr || exit 1
export HF_HOME=/workspace/hf
[ -f .hftoken ] && export HF_TOKEN=$(cat .hftoken)

wait_gpu() {
  for _ in $(seq 1 180); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$u" -lt 2000 ] && { echo "[chain] gpu free (${u} MiB)"; return 0; }
    sleep 10
  done
  echo "[chain] WARNING gpu still busy, continuing"
}

run() {  # run <model> <hf-id> <layer> <bs>
  echo "[chain] START $1"
  wait_gpu
  python3 -u patching_rerun.py --model "$1" --hf-id "$2" --layer "$3" \
    --n 300 --bs "$4" > "/workspace/tcr/patch_$1.log" 2>&1
  echo "[chain] DONE $1 rc=$?"
  grep -aE "baseline unsafe|flipped to safe|patched degenerate" "/workspace/tcr/patch_$1.log"
  df -h /workspace | tail -1
}

# smallest first, so a configuration problem surfaces in two minutes not twenty
run Mistral-7B-Instruct-v0.3     mistralai/Mistral-7B-Instruct-v0.3   26 24
run c4ai-command-r7b-12-2024     CohereLabs/c4ai-command-r7b-12-2024  26 24
run Qwen3-14B                    Qwen/Qwen3-14B                       33 24
run gemma-3-27b-it               google/gemma-3-27b-it                51 16
run Meta-Llama-3.1-70B-Instruct  unsloth/Meta-Llama-3.1-70B-Instruct  67 16

echo "[chain] ALL DONE"
