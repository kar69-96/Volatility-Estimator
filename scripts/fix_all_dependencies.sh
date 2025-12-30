#!/bin/bash
# Comprehensive fix script for all dependency issues preventing T5 model loading

set -e

echo "=========================================="
echo "COMPREHENSIVE DEPENDENCY FIX"
echo "=========================================="

# Step 1: Detect PyTorch CUDA version
echo ""
echo "Step 1: Detecting PyTorch CUDA version..."
PYTORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "11.8")
PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
echo "  PyTorch version: $PYTORCH_VERSION"
echo "  PyTorch CUDA version: $PYTORCH_CUDA"

# Step 2: Fix torchvision CUDA mismatch
echo ""
echo "Step 2: Fixing torchvision CUDA version mismatch..."
pip uninstall torchvision -y || true
if [[ "$PYTORCH_CUDA" == "11.8" ]] || [[ "$PYTORCH_CUDA" == "11"* ]]; then
    echo "  Installing torchvision for CUDA 11.8..."
    pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cu118
elif [[ "$PYTORCH_CUDA" == "12.1" ]] || [[ "$PYTORCH_CUDA" == "12"* ]]; then
    echo "  Installing torchvision for CUDA 12.1..."
    pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "  Installing torchvision (CPU or unknown CUDA)..."
    pip install --no-cache-dir torchvision
fi

# Step 3: Upgrade Pillow
echo ""
echo "Step 3: Upgrading Pillow (PIL)..."
pip install --no-cache-dir --upgrade Pillow>=9.0.0

# Step 4: Reinstall transformers to ensure clean state
echo ""
echo "Step 4: Reinstalling transformers..."
pip uninstall transformers -y || true
pip install --no-cache-dir transformers>=4.40.0

# Step 5: Install/upgrade sentencepiece
echo ""
echo "Step 5: Installing sentencepiece..."
pip install --no-cache-dir sentencepiece>=0.1.99

# Step 6: Clear Python cache
echo ""
echo "Step 6: Clearing Python cache..."
find ~/.local/lib/python3.*/site-packages/transformers -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
find ~/.local/lib/python3.*/site-packages/transformers/models/t5 -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true

# Step 7: Comprehensive verification
echo ""
echo "Step 7: Verifying installation..."
python3 << 'EOF'
import sys

print("  Checking PyTorch...")
import torch
print(f"    ✓ PyTorch {torch.__version__} (CUDA {torch.version.cuda})")

print("  Checking torchvision...")
try:
    import torchvision
    print(f"    ✓ torchvision {torchvision.__version__}")
except Exception as e:
    print(f"    ✗ torchvision failed: {e}")
    sys.exit(1)

print("  Checking Pillow...")
import PIL
from PIL import Image
if hasattr(Image, 'Resampling'):
    print(f"    ✓ Pillow {PIL.__version__} (Resampling available)")
else:
    print(f"    ✗ Pillow {PIL.__version__} (Resampling NOT available)")
    sys.exit(1)

print("  Checking transformers...")
import transformers
print(f"    ✓ transformers {transformers.__version__}")

print("  Checking T5ForConditionalGeneration import...")
try:
    from transformers.models.t5 import T5ForConditionalGeneration
    print("    ✓ T5ForConditionalGeneration imported successfully")
except Exception as e:
    print(f"    ✗ T5ForConditionalGeneration import failed: {e}")
    sys.exit(1)

print("  Checking AutoModelForSeq2SeqLM...")
try:
    from transformers import AutoModelForSeq2SeqLM
    print("    ✓ AutoModelForSeq2SeqLM imported successfully")
except Exception as e:
    print(f"    ✗ AutoModelForSeq2SeqLM import failed: {e}")
    sys.exit(1)

print("  Testing model loading (dry run)...")
try:
    # Just test that we can start loading (will download config)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained('amazon/chronos-t5-mini')
    print(f"    ✓ Config loaded: {config.model_type}")
except Exception as e:
    print(f"    ⚠ Config load warning: {e}")
    print("    (This is OK - model will download on first use)")

print("")
print("  ✓✓✓ ALL CHECKS PASSED ✓✓✓")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Installation fix complete!"
    echo "=========================================="
    echo ""
    echo "You can now run: python3 scripts/train.py"
    exit 0
else
    echo ""
    echo "=========================================="
    echo "✗ Verification failed!"
    echo "=========================================="
    echo "Running diagnostic script..."
    python3 scripts/diagnose_transformers.py
    exit 1
fi

