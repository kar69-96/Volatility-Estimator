#!/bin/bash
# Setup script for Lambda Labs instance
# Installs all dependencies with correct CUDA versions

set -e

echo "=========================================="
echo "Lambda Labs Setup Script"
echo "=========================================="

# Detect CUDA version
echo ""
echo "Step 1: Detecting CUDA version..."
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
echo "Step 2: Installing PyTorch packages with matching CUDA versions..."
pip install --no-cache-dir torch torchvision torchaudio --index-url "$PYTORCH_INDEX"

# Verify PyTorch installation
echo ""
echo "Step 3: Verifying PyTorch installation..."
python3 -c "
import torch
print(f'  ✓ PyTorch {torch.__version__}')
print(f'  ✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  ✓ CUDA version: {torch.version.cuda}')
    print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}')
import torchvision
print(f'  ✓ torchvision {torchvision.__version__}')
import torchaudio
print(f'  ✓ torchaudio {torchaudio.__version__}')
"

# Install remaining requirements
echo ""
echo "Step 4: Installing remaining requirements..."
pip install --no-cache-dir -r requirements.txt

# Verify critical dependencies
echo ""
echo "Step 5: Verifying critical dependencies..."
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

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "You can now run: python3 scripts/train.py"
echo ""

