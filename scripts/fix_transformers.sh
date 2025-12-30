#!/bin/bash
# Fix script to properly reinstall transformers with T5 support

set -e

echo "=========================================="
echo "Fixing transformers installation..."
echo "=========================================="

# Uninstall transformers completely
echo "Step 1: Uninstalling transformers..."
pip uninstall transformers -y || true

# Also remove any cached wheels
echo "Step 2: Clearing pip cache..."
pip cache purge || true

# Reinstall transformers with T5 support
echo "Step 3: Installing transformers>=4.40.0..."
pip install --no-cache-dir transformers>=4.40.0

# Install sentencepiece (required for T5 tokenization)
echo "Step 4: Installing sentencepiece..."
pip install --no-cache-dir sentencepiece>=0.1.99

# Upgrade Pillow (required for newer transformers versions)
echo "Step 5: Upgrading Pillow (PIL)..."
pip install --no-cache-dir --upgrade Pillow>=9.0.0

# Verify installation
echo "Step 6: Verifying T5 availability..."
python3 -c "
import PIL
print(f'✓ Pillow version: {PIL.__version__}')
from PIL import Image
if hasattr(Image, 'Resampling'):
    print('✓ PIL.Image.Resampling available')
else:
    print('✗ PIL.Image.Resampling NOT available - Pillow too old!')
    exit(1)
from transformers.models.t5 import T5ForConditionalGeneration
print('✓ T5ForConditionalGeneration imported successfully')
from transformers import AutoModelForSeq2SeqLM
print('✓ AutoModelForSeq2SeqLM imported successfully')
import transformers
print(f'✓ transformers version: {transformers.__version__}')
" || {
    echo "✗ Verification failed!"
    echo "Running diagnostic script..."
    python3 scripts/diagnose_transformers.py
    exit 1
}

echo ""
echo "=========================================="
echo "✓ Installation complete!"
echo "=========================================="

