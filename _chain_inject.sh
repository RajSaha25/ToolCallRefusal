#!/bin/bash
# Runs after _finish_all.sh: the Gemma tool-injection diagnostic.
cd /workspace/tcr || exit 1
export HF_HOME=/workspace/hf HF_TOKEN=$(cat .hftoken)
while pgrep -f _finish_all.sh > /dev/null; do sleep 20; done
echo "[inject-chain] main chain finished"
for _ in $(seq 1 90); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$u" -lt 2000 ] && break
  sleep 10
done
echo "[inject-chain] gpu free (${u} MiB); starting gemma injection"
python3 -u gemma_tool_injection.py --n 120 --bs 16 > /workspace/tcr/inject_gemma.log 2>&1
echo "[inject-chain] done rc=$?"
