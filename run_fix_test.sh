#!/usr/bin/env bash
# One-shot launcher for fix_test_ablation.py on a CUDA box (Raj's B200 / RunPod A100-80GB).
# Usage: HF_TOKEN=hf_xxx bash run_fix_test.sh [model]   (default google/gemma-3-27b-it)
set -euo pipefail
MODEL="${1:-google/gemma-3-27b-it}"
python -c "import torch, transformers" 2>/dev/null || pip install -q "torch" "transformers>=4.50" accelerate
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
python fix_test_ablation.py --model "$MODEL" --n-dir 64 --n-test 4 --n-kl 24 --max-new 120 \
  --out "fix_test_$(basename "$MODEL").json" 2>&1 | tee "fix_test_$(basename "$MODEL").log"
