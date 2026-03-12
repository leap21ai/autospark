#!/bin/bash
# Deploy KernelAgent to dgx01 and set up environment
set -euo pipefail

DGX01="trey@100.90.153.43"
SSH_OPTS="-o StrictHostKeyChecking=no -i $HOME/.ssh/id_ed25519"
REMOTE_DIR="/home/trey/autospark"

echo "=== Step 1: Sync KernelAgent to dgx01 ==="
rsync -avz --delete \
  -e "ssh $SSH_OPTS" \
  /Users/trey/dev/gitlab/autospark/KernelAgent/ \
  "$DGX01:$REMOTE_DIR/KernelAgent/"

echo "=== Step 2: Set up environment on dgx01 ==="
ssh $SSH_OPTS "$DGX01" bash -s <<'REMOTE_SETUP'
set -euo pipefail

REMOTE_DIR="/home/trey/autospark"
VENV="/home/trey/finetune-env"

# Verify GPU
echo "--- GPU check ---"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# Verify existing venv has what we need
echo "--- Checking finetune-env ---"
source "$VENV/bin/activate"
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
import triton
print(f'Triton: {triton.__version__}')
"

# Install KernelAgent deps into existing venv
echo "--- Installing KernelAgent dependencies ---"
cd "$REMOTE_DIR/KernelAgent"
pip install -q openai anthropic jinja2 python-dotenv gradio requests numpy omegaconf 2>&1 | tail -5

# Create .env file
echo "--- Creating .env ---"
cat > "$REMOTE_DIR/KernelAgent/.env" <<'ENV'
# LLM Provider — using Anthropic API
ANTHROPIC_API_KEY=placeholder

# Agent config
NUM_KERNEL_SEEDS=2
MAX_REFINEMENT_ROUNDS=5
LOG_LEVEL=INFO

# CUDA environment
CUDA_HOME=/usr/local/cuda-13.0
TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
TORCH_CUDA_ARCH_LIST=12.1a
ENV

echo "--- Setup complete ---"
echo "Ready to run KernelAgent on DGX Spark (GB10)"
REMOTE_SETUP

echo "=== Done ==="
