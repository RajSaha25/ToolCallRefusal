#!/bin/bash
# Qwen retry for experiment 2. Its gpu_*.json predates the proj_gap field (it was
# the first model run), so --auto-grid has nothing to read. Its gap is 311.33 per
# the committed interp_artifacts/Qwen3-14B/interp_summary.json, which gives exactly
# the published grid and addition coefficient, so pass them explicitly.
cd /workspace/tcr || exit 1
export HF_HOME=/workspace/hf HF_TOKEN=$(cat .hftoken 2>/dev/null)
while pgrep -f _exp2.sh > /dev/null; do sleep 20; done
echo "[exp2b] main chain finished"
for _ in $(seq 1 180); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$u" -lt 2000 ] && break
  sleep 10
done
echo "[exp2b] gpu free (${u} MiB); qwen"
python3 -u rerun_steering_gen.py --model Qwen3-14B --hf-id Qwen/Qwen3-14B --layer 33 \
  --grid 0,200,450,700 --add-coef 1245.0 \
  --n-abl 240 --n-add 240 --n-steer-h 200 --n-steer-b 120 --bs 24 \
  > /workspace/tcr/exp2_Qwen3-14B.log 2>&1
echo "[exp2b] qwen rc=$?"
grep -aE "^\[steer\]" /workspace/tcr/exp2_Qwen3-14B.log | tail -4
echo "[exp2b] ALL DONE"
