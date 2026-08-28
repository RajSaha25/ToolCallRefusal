#!/bin/bash
# Gemma tool-mode, reconstructed by injecting the schemas into the prompt text
# (its template cannot carry them). make_renderer detects this automatically.
# Validated first: injection gives 12.5% unsafe on harmful tool-normal prompts vs
# 14.6% stored under the same scorer.
cd /workspace/tcr || exit 1
export HF_HOME=/workspace/hf HF_TOKEN=$(cat .hftoken)
while pgrep -f _chain_mistral.sh > /dev/null; do sleep 20; done
echo "[gemma-tools] mistral chain finished"
for _ in $(seq 1 90); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$u" -lt 2000 ] && break
  sleep 10
done
echo "[gemma-tools] gpu free (${u} MiB); gemma gpu stage"
python3 -u rerun_against_relabels.py --stage gpu --model gemma-3-27b-it \
  --hf-id google/gemma-3-27b-it --layer 51 --batch 8 \
  > /workspace/tcr/gpu_gemma2.log 2>&1
echo "[gemma-tools] gpu stage rc=$?"
for _ in $(seq 1 90); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$u" -lt 2000 ] && break
  sleep 10
done
echo "[gemma-tools] gemma steering"
python3 -u rerun_steering_gen.py --model gemma-3-27b-it --hf-id google/gemma-3-27b-it \
  --layer 51 --auto-grid --bs 16 > /workspace/tcr/steer_gemma2.log 2>&1
echo "[gemma-tools] steering rc=$?"
echo "[gemma-tools] ALL DONE"
