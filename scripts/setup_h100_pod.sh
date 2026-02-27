#!/bin/bash
# Setup script for H100 pod — D2L training with GPT-OSS 20B
# Requirements: Python 3.11, CUDA 12.6+ driver, H100 GPU
set -e

echo "=== D2L GPT-OSS 20B Training Setup ==="

# 1. Install uv (fast Python package manager)
if ! command -v uv &> /dev/null; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Create venv with Python 3.11
echo ">>> Creating venv..."
uv venv --python 3.11 .venv
source .venv/bin/activate

# 3. Install torch 2.8.0+cu126 (bundles triton 3.4 for native MXFP4)
echo ">>> Installing PyTorch 2.8.0+cu126..."
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# 4. Install kernels (MXFP4 triton kernels for GPT-OSS)
echo ">>> Installing kernels..."
uv pip install kernels

# 5. Install flash-attn (no compilation needed with --no-build-isolation)
echo ">>> Installing flash-attn..."
uv pip install flash-attn --no-build-isolation

# 6. Install project dependencies
echo ">>> Installing project dependencies..."
uv pip install -e .

# 7. Login to HuggingFace (needed for gated models)
echo ">>> HuggingFace login..."
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: Set HF_TOKEN env var or run: huggingface-cli login"
else
    huggingface-cli login --token "$HF_TOKEN"
fi

# 8. Verify setup
echo ">>> Verifying setup..."
python -c "
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
    print('MXFP4 kernels: NOT available')

try:
    import flash_attn
    print(f'Flash Attention: {flash_attn.__version__}')
except ImportError:
    print('Flash Attention: NOT available')

import triton
print(f'Triton: {triton.__version__}')
"

echo ""
echo "=== Setup complete! ==="
echo "Run the sanity test with:"
echo "  WANDB_MODE=disabled .venv/bin/python -m accelerate.commands.launch train.py --config configs/main_exp/gpt_oss_20b.yaml --model_name_or_path openai/gpt-oss-20b"
