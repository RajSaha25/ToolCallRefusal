#!/bin/bash
# Wait for the Llama steering run to finish, free its weights, then run Gemma's
# steering on a grid scaled to Gemma's own projection gap (18944), using the same
# ratios Qwen's published grid implies: 0, 0.64x, 1.45x, 2.25x.
cd /workspace/tcr || exit 1
while pgrep -f "rerun_steering_gen.py --model Meta-Llama" > /dev/null; do sleep 20; done
echo "[chain] llama steering finished"
if [ ! -f relabel_analysis/steer_raw_Meta-Llama-3.1-70B-Instruct.json ]; then
  echo "[chain] ERROR llama output missing, not proceeding"; exit 1
fi
rm -rf /workspace/hf/hub/models--NousResearch--Meta-Llama-3.1-70B-Instruct
echo "[chain] freed llama weights; disk:"; df -h /workspace | tail -1
export HF_HOME=/workspace/hf HF_TOKEN=$(cat .hftoken)
python3 -u rerun_steering_gen.py --model gemma-3-27b-it --hf-id google/gemma-3-27b-it \
  --layer 51 --add-coef 75777.6 --grid 0,12163,27374,42586 --bs 16 \
  > /workspace/tcr/steer_gemma.log 2>&1
echo "[chain] gemma steering finished rc=$?"
