#!/usr/bin/env python3
"""Test script to verify the setup is correct before training."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch not found: {e}")
        return False
    
    try:
        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ Transformers not found: {e}")
        return False
    
    try:
        import peft
        print(f"  ✓ PEFT available")
    except ImportError:
        print(f"  ⚠ PEFT not found (LoRA won't work)")
    
    try:
        import pandas as pd
        import numpy as np
        print(f"  ✓ Pandas, NumPy available")
    except ImportError as e:
        print(f"  ✗ Pandas/NumPy not found: {e}")
        return False
    
    return True

def test_model_loading():
    """Test that the Chronos model can be loaded."""
    print("\nTesting model loading...")
    try:
        from src.models.chronos import ChronosVolatility
        print("  Attempting to load Chronos model (this may download the model)...")
        # Try to load without LoRA first to test basic loading
        model = ChronosVolatility(use_lora=False)
        print("  ✓ Model loaded successfully (without LoRA)")
        
        # Test forward pass with dummy data
        import torch
        dummy_input = torch.randn(2, 60)  # batch_size=2, seq_len=60
        output = model(dummy_input)
        print(f"  ✓ Forward pass works: input shape {dummy_input.shape} -> output shape {output.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test that data can be loaded."""
    print("\nTesting data loading...")
    try:
        import pandas as pd
        from pathlib import Path
        
        data_dir = Path('data/cache')
        test_ticker = 'AAPL'
        data_path = data_dir / f'{test_ticker}.parquet'
        
        if not data_path.exists():
            print(f"  ⚠ Data file not found: {data_path}")
            return False
        
        df = pd.read_parquet(data_path)
        print(f"  ✓ Loaded {test_ticker} data: {len(df)} rows")
        print(f"    Columns: {list(df.columns)}")
        
        # Test data preparation
        from src.training.data import prepare_raw_signal, compute_target
        
        raw_signal = prepare_raw_signal(df)
        print(f"  ✓ Prepared raw signal: {len(raw_signal)} values")
        
        # Compute returns for target
        returns = pd.Series(np.log(df['close'] / df['close'].shift(1)), index=df.index).dropna()
        target = compute_target(returns, horizon=20)
        print(f"  ✓ Computed targets: {len(target)} values")
        
        # Test dataset creation
        from src.training.data import VolatilityDataset
        dataset = VolatilityDataset(raw_signal, target, seq_length=60, horizon=20)
        print(f"  ✓ Created dataset: {len(dataset)} samples")
        
        if len(dataset) > 0:
            # Test getting a sample
            sample_input, sample_target = dataset[0]
            print(f"  ✓ Sample shape: input {sample_input.shape}, target {sample_target.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Chronos Fine-tuning Setup Test")
    print("=" * 60)
    
    all_passed = True
    
    if not test_imports():
        print("\n❌ Import test failed. Please install missing dependencies.")
        all_passed = False
        return
    
    if not test_model_loading():
        print("\n❌ Model loading test failed. Check model implementation.")
        all_passed = False
        return
    
    if not test_data_loading():
        print("\n❌ Data loading test failed. Check data files.")
        all_passed = False
        return
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! You're ready to start training.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run training: python scripts/train.py")
    print("  2. Or test on Lambda Labs GPU instance")

if __name__ == '__main__':
    import numpy as np  # Import here for use in test_data_loading
    main()

