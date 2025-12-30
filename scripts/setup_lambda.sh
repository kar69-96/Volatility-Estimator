#!/bin/bash
# Setup script for Lambda Labs instance
# Creates virtual environment and installs all dependencies with correct CUDA versions

set -e

echo "=========================================="
echo "Lambda Labs Setup Script"
echo "=========================================="

# Check if venv already exists
if [ -d "venv" ]; then
    echo ""
    echo "Virtual environment 'venv' already exists."
    read -p "Do you want to remove it and create a fresh one? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
    else
        echo "Using existing virtual environment..."
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Step 1: Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "Step 2: Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Step 3: Upgrading pip..."
pip install --upgrade pip

# Detect CUDA version
echo ""
echo "Step 4: Detecting CUDA version..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "  Found CUDA: $CUDA_VERSION"
else
    echo "  CUDA not found via nvcc, defaulting to 11.8"
    CUDA_VERSION="11.8"
fi

# Determine PyTorch index URL
if [[ "$CUDA_VERSION" == "12.1" ]] || [[ "$CUDA_VERSION" == "12"* ]]; then
    PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
    echo "  Using PyTorch index for CUDA 12.1"
elif [[ "$CUDA_VERSION" == "11.8" ]] || [[ "$CUDA_VERSION" == "11"* ]]; then
    PYTORCH_INDEX="https://download.pytorch.org/whl/cu118"
    echo "  Using PyTorch index for CUDA 11.8"
else
    PYTORCH_INDEX="https://download.pytorch.org/whl/cu118"
    echo "  Defaulting to CUDA 11.8 PyTorch index"
fi

# Install PyTorch packages from index
echo ""
echo "Step 5: Installing PyTorch packages with matching CUDA versions..."
pip install --no-cache-dir torch torchvision torchaudio --index-url "$PYTORCH_INDEX"

# Verify PyTorch installation
echo ""
echo "Step 6: Verifying PyTorch installation..."
python3 -c "
import torch
print(f'  ✓ PyTorch {torch.__version__}')
print(f'  ✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  ✓ CUDA version: {torch.version.cuda}')
    print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}')
import torchvision
print(f'  ✓ torchvision {torchvision.__version__}')
print(f'  ✓ torchvision path: {torchvision.__file__}')
# Verify it's NOT from system packages
assert '/usr/lib/python3/dist-packages' not in torchvision.__file__, 'ERROR: torchvision is from system packages!'
print('  ✓ torchvision is from virtual environment (not system packages)')
import torchaudio
print(f'  ✓ torchaudio {torchaudio.__version__}')
"

# Install remaining requirements
echo ""
echo "Step 7: Installing remaining requirements..."
pip install --no-cache-dir -r requirements.txt

# Install Lambda Labs CLI for auto-termination
echo ""
echo "Step 7.5: Installing Lambda Labs CLI (for auto-termination)..."
echo "  Installing lambdacloud (official Lambda Labs CLI)..."
pip install --no-cache-dir lambdacloud || echo "  ⚠ Failed to install lambdacloud"
echo "  Installing lambda-cli (alternative CLI)..."
pip install --no-cache-dir lambda-cli || echo "  ⚠ Failed to install lambda-cli"

if command -v lambdacloud &> /dev/null; then
    echo "  ✓ lambdacloud installed successfully"
    echo "  Note: Authenticate with: lambdacloud auth login"
    echo "        Or set LAMBDA_API_KEY environment variable"
elif command -v lambda &> /dev/null; then
    echo "  ✓ lambda CLI installed successfully"
    echo "  Note: Authenticate with: lambda auth"
else
    echo "  ⚠ Neither CLI found in PATH after installation"
    echo "  You may need to activate the venv or add to PATH"
fi

# Verify critical dependencies
echo ""
echo "Step 8: Verifying critical dependencies and Lambda CLI..."
python3 -c "
print('  Checking Pillow...')
import PIL
from PIL import Image
assert hasattr(Image, 'Resampling'), 'Pillow too old - Resampling not available'
print(f'  ✓ Pillow {PIL.__version__} (Resampling available)')

print('  Checking transformers...')
import transformers
print(f'  ✓ transformers {transformers.__version__}')

print('  Checking T5 support...')
from transformers.models.t5 import T5ForConditionalGeneration
print('  ✓ T5ForConditionalGeneration available')

print('  Checking AutoModelForSeq2SeqLM...')
from transformers import AutoModelForSeq2SeqLM
print('  ✓ AutoModelForSeq2SeqLM available')

print('  Checking PEFT...')
from peft import LoraConfig, get_peft_model
print('  ✓ PEFT available')
"

# Verify Lambda CLI
if command -v lambdacloud &> /dev/null; then
    echo "  ✓ Lambda CLI (lambdacloud) available for auto-termination"
elif command -v lambda &> /dev/null; then
    echo "  ✓ Lambda CLI (lambda) available for auto-termination"
else
    echo "  ⚠ Lambda CLI not found in PATH (may need to activate venv)"
fi

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment in future sessions:"
echo "  source venv/bin/activate"
echo ""
echo "To run training on A100:"
echo "  source venv/bin/activate"
echo "  export CUDA_VISIBLE_DEVICES=0  # Use GPU 0 (A100)"
echo "  python3 scripts/train_sample.py"
echo ""
echo "Note: The training script will automatically:"
echo "  - Detect A100 and optimize batch size (128) and enable mixed precision"
echo "  - Use CUDA_VISIBLE_DEVICES=0 to ensure single GPU usage"
echo ""
