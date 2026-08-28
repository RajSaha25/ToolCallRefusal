#!/bin/bash
# Mistral retry: its first attempt died because transformers 4.57 needs the slow
# tokenizer path for Mistral v0.3, which wants sentencepiece + protobuf. Those are
# installed now. Runs after the Gemma injection so only one model is resident.
cd /workspace/tcr || exit 1
export HF_HOME=/workspace/hf HF_TOKEN=$(cat .hftoken)
while pgrep -f _chain_inject.sh > /dev/null; do sleep 20; done
echo "[mistral-chain] injection finished"
for _ in $(seq 1 90); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$u" -lt 2000 ] && break
  sleep 10
done
echo "[mistral-chain] gpu free (${u} MiB)"
python3 -u rerun_against_relabels.py --stage gpu --model Mistral-7B-Instruct-v0.3 \
  --hf-id mistralai/Mistral-7B-Instruct-v0.3 --layer 26 --batch 8 \
  > /workspace/tcr/gpu_mistral2.log 2>&1
echo "[mistral-chain] gpu stage rc=$?"
for _ in $(seq 1 90); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$u" -lt 2000 ] && break
  sleep 10
done
python3 -u rerun_steering_gen.py --model Mistral-7B-Instruct-v0.3 \
  --hf-id mistralai/Mistral-7B-Instruct-v0.3 --layer 26 --auto-grid --bs 24 \
  > /workspace/tcr/steer_mistral2.log 2>&1
echo "[mistral-chain] steering rc=$?"
echo "[mistral-chain] ALL DONE"
