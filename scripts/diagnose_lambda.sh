#!/bin/bash
# Diagnostic script to check Lambda instance setup

echo "=========================================="
echo "Lambda Instance Diagnostic"
echo "=========================================="
echo ""

echo "1. Current directory:"
pwd
echo ""

echo "2. Files in current directory:"
ls -la
echo ""

echo "3. Checking for scripts directory:"
if [ -d "scripts" ]; then
    echo "  ✓ scripts directory exists"
    echo "  Contents:"
    ls -la scripts/
else
    echo "  ✗ scripts directory NOT found"
fi
echo ""

echo "4. Checking for train_sample.py:"
if [ -f "scripts/train_sample.py" ]; then
    echo "  ✓ train_sample.py found"
else
    echo "  ✗ train_sample.py NOT found"
    echo "  Searching for it..."
    find ~ -name "train_sample.py" 2>/dev/null | head -5
fi
echo ""

echo "5. Checking for project root:"
if [ -f "requirements.txt" ] && [ -d "src" ]; then
    echo "  ✓ Project files found (requirements.txt, src/)"
else
    echo "  ✗ Project files NOT found"
    echo "  You need to upload/clone the project"
fi
echo ""

echo "6. Checking for virtual environment:"
if [ -d "venv" ]; then
    echo "  ✓ venv directory exists"
else
    echo "  ✗ venv directory NOT found"
    echo "  Run: bash scripts/setup_lambda.sh"
fi
echo ""

echo "=========================================="
echo "Diagnostic complete"
echo "=========================================="

