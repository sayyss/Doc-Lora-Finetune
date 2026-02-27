#!/bin/bash
# Setup script for H100 pod — D2L training with GPT-OSS 20B
# Tested: Python 3.11, CUDA 12.8, H100 80GB, torch 2.9.1
#
# Usage:
#   export HF_TOKEN=hf_xxxxx  # optional but recommended
#   bash scripts/setup_h100_pod.sh
#
# After setup, run sanity test:
#   WANDB_MODE=disabled .venv/bin/python -m accelerate.commands.launch train.py \
#       configs/main_exp/gpt_oss_20b.yaml \
#       --model_name_or_path=openai/gpt-oss-20b --max_steps=1
set -e

echo "=== D2L GPT-OSS 20B Training Setup ==="

# 1. Install uv
if ! command -v uv &> /dev/null; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Create venv
echo ">>> Creating venv..."
uv venv --python 3.11 .venv
source .venv/bin/activate

# 3. Install torch 2.9.1+cu128 (last version compatible with flash-attn 2.8.3)
echo ">>> Installing PyTorch 2.9.1+cu128..."
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# 4. Install kernels (MXFP4 triton kernels for native GPT-OSS on Hopper)
echo ">>> Installing kernels..."
uv pip install kernels

# 5. Install flash-attn from prebuilt wheel (torch 2.9, cu128, Python 3.11)
# Building from source takes 1+ hour; prebuilt wheel installs in seconds
echo ">>> Installing flash-attn (prebuilt wheel)..."
uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.9-cp311-cp311-linux_x86_64.whl"

# 6. Install wheel (needed for editable install)
echo ">>> Installing wheel..."
uv pip install wheel

# 7. Install project dependencies
echo ">>> Installing project dependencies..."
uv pip install -e .

# 8. Pin torch so 'uv run' doesn't replace it
# (uv run re-syncs from pyproject.toml which can upgrade torch and break flash-attn)
echo ">>> Pinning torch version..."
if ! grep -q 'torch==' pyproject.toml; then
    sed -i 's/"transformers>=4.55.0",/"transformers>=4.55.0",\n    "torch==2.9.1",/' pyproject.toml
fi

# 9. HuggingFace login
if [ -n "$HF_TOKEN" ]; then
    echo ">>> Logging into HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN"
else
    echo ">>> WARNING: HF_TOKEN not set. Set it for faster downloads."
    echo "   export HF_TOKEN=hf_xxxxx"
fi

# 10. Verify setup
echo ""
echo ">>> Verifying setup..."
.venv/bin/python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability()
    print(f'Compute capability: {cap[0]}.{cap[1]}')
    print(f'Hopper+: {cap[0] >= 9}')

import transformers
print(f'Transformers: {transformers.__version__}')

try:
    import kernels
    print('MXFP4 kernels: available')
except ImportError:
    print('MXFP4 kernels: NOT available (will dequantize to bf16)')

from flash_attn import flash_attn_func
print('Flash Attention: OK')

import triton
print(f'Triton: {triton.__version__}')
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Run sanity test (1 step):"
echo "  WANDB_MODE=disabled .venv/bin/python -m accelerate.commands.launch train.py \\"
echo "      configs/main_exp/gpt_oss_20b.yaml \\"
echo "      --model_name_or_path=openai/gpt-oss-20b --max_steps=1"
echo ""
echo "Run full training (~20k steps):"
echo "  .venv/bin/python -m accelerate.commands.launch train.py \\"
echo "      configs/main_exp/gpt_oss_20b.yaml \\"
echo "      --model_name_or_path=openai/gpt-oss-20b --max_steps=20000"
echo ""
echo "IMPORTANT: Always use '.venv/bin/python' directly, NOT 'uv run'"
echo "  (uv run re-syncs packages and can break flash-attn)"
