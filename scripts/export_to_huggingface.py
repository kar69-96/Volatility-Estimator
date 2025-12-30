#!/usr/bin/env python3
"""Export trained ChronosVolatility model to Hugging Face format."""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from src.models.chronos import ChronosVolatility
from src.models.base_model import get_device

try:
    from peft import PeftModel
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

try:
    from transformers import AutoModelForSeq2SeqLM
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


def create_model_card(model_name, base_model, description, training_info=None):
    """Create a model card (README.md) for Hugging Face."""
    
    card_content = f"""---
library_name: transformers
tags:
- finance
- volatility
- time-series
- forecasting
- chronos
- peft
- lora
base_model: {base_model}
license: apache-2.0
---

# {model_name}

## Model Description

{description}

This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model}) using LoRA (Low-Rank Adaptation) for volatility prediction on stock market data.

## Model Architecture

- **Base Model**: {base_model}
- **Task**: Volatility Prediction (Quantile Regression)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Config**: r=8, alpha=16, target_modules=["q", "k", "v", "o"]
- **Custom Heads**: 
  - Value Embedding: Linear(1 -> hidden_dim) for embedding raw time series values
  - Quantile Head: Linear(hidden_dim -> 3) for predicting q10, q50, q90 quantiles

## Input Format

- **Input**: Single-channel time series of squared returns (shape: batch_size, seq_length=60)
- **Sequence Length**: 60 trading days
- **Preprocessing**: Input should be squared log returns: `(log(price_t / price_{t-1}))^2`

## Output Format

- **Output**: Quantiles in log-variance space (shape: batch_size, 3)
  - Column 0: q10 (10th percentile)
  - Column 1: q50 (50th percentile / median)
  - Column 2: q90 (90th percentile)

To convert to annualized volatility:
```python
variance = np.exp(log_variance)
volatility = np.sqrt(variance) * np.sqrt(252)  # Annualized
```

## Usage

### Loading the Model

```python
from src.models.chronos import ChronosVolatility
import torch

# Load model
model = ChronosVolatility(use_lora=True)
model.load_custom_heads("path/to/heads.pt")
model.base.load_adapter("path/to/adapter")  # For PEFT adapter

# Or use from_pretrained (if properly saved):
# from peft import PeftModel
# base_model = AutoModelForSeq2SeqLM.from_pretrained("{base_model}")
# model.base = PeftModel.from_pretrained(base_model, "{model_name}")
# model.load_custom_heads("path/to/heads.pt")
```

### Making Predictions

```python
import numpy as np

# Prepare input: squared returns sequence (60 days)
input_seq = torch.FloatTensor(squared_returns).unsqueeze(0)  # (1, 60)

# Get predictions
model.eval()
with torch.no_grad():
    quantiles_log_var = model(input_seq)  # (1, 3)

# Convert to volatility
quantiles_var = np.exp(quantiles_log_var.numpy())
quantiles_vol = np.sqrt(quantiles_var) * np.sqrt(252)  # Annualized %

print(f"10th percentile: {{quantiles_vol[0][0]:.2f}}%")
print(f"Median: {{quantiles_vol[0][1]:.2f}}%")
print(f"90th percentile: {{quantiles_vol[0][2]:.2f}}%")
```

## Training Details

"""
    
    if training_info:
        card_content += f"""
### Training Data
- **Dataset**: {training_info.get('dataset', 'Stock market data (AAPL, MSFT, GOOG, SPY, TSLA)')}
- **Training Period**: {training_info.get('period', 'Last 10 years')}
- **Train/Val Split**: {training_info.get('split', '80/20')}

### Training Configuration
- **Epochs**: {training_info.get('epochs', '50')}
- **Learning Rate**: {training_info.get('lr', '1e-5')}
- **Batch Size**: {training_info.get('batch_size', '32')}
- **Sequence Length**: 60
- **Prediction Horizon**: 20 days
- **Loss Function**: Quantile loss (pinball loss) for q10, q50, q90

### Evaluation Metrics
- **Validation Loss**: {training_info.get('val_loss', 'N/A')}
- **QLIKE Loss**: {training_info.get('qlike_loss', 'N/A')}
"""
    
    card_content += """
## Limitations

- The model is trained on specific stock market data and may not generalize well to other markets or time periods
- Predictions are probabilistic (quantiles) and should be interpreted with uncertainty
- The model requires at least 60 days of historical data for predictions

## Citation

If you use this model, please cite:

```bibtex
@misc{{chronos-volatility,
  title={{Chronos Fine-tuned for Volatility Prediction}},
  author={{Your Name}},
  year={{2024}},
  publisher={{Hugging Face}}
}}
```

## License

This model is licensed under Apache 2.0. The base Chronos model is also licensed under Apache 2.0.
"""
    
    return card_content


def export_model(
    checkpoint_path,
    output_dir,
    model_name=None,
    description=None,
    push_to_hub=False,
    hub_repo_id=None,
    training_info=None
):
    """
    Export trained ChronosVolatility model to Hugging Face format.
    
    Args:
        checkpoint_path: Path to trained model checkpoint (.pt file)
        output_dir: Directory to save exported model
        model_name: Name for the model (default: infer from output_dir)
        description: Model description for model card
        push_to_hub: Whether to push to Hugging Face Hub
        hub_repo_id: Hugging Face Hub repository ID (e.g., "username/model-name")
        training_info: Dictionary with training metadata
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from {checkpoint_path}...")
    device = get_device('auto')
    
    # Initialize model
    model = ChronosVolatility(use_lora=True).to(device)
    
    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle both full checkpoint and state_dict-only formats
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
        checkpoint_meta = state_dict
    else:
        model.load_state_dict(state_dict)
        checkpoint_meta = {}
    
    model.eval()
    print("✓ Model loaded")
    
    # Get base model ID
    base_model_id = model.model_id
    
    # Extract base model and LoRA adapters
    # Check if model.base is a PEFT model
    if _PEFT_AVAILABLE and hasattr(model.base, 'save_pretrained'):
        # PEFT model - save adapters
        adapter_dir = output_dir / "adapter"
        adapter_dir.mkdir(exist_ok=True)
        
        print(f"Saving LoRA adapters to {adapter_dir}...")
        try:
            model.base.save_pretrained(str(adapter_dir))
            print("✓ LoRA adapters saved")
        except Exception as e:
            print(f"Warning: Could not save adapters using save_pretrained: {e}")
            print("Falling back to saving state dict...")
            # Fallback: save adapter state dict
            adapter_state = model.base.state_dict()
            torch.save(adapter_state, adapter_dir / "adapter_model.bin")
            print("✓ LoRA adapters saved (fallback method)")
    else:
        print("Warning: Model doesn't appear to be a PEFT model. Saving full base model state...")
        # Save base model state dict as fallback
        adapter_dir = output_dir / "adapter"
        adapter_dir.mkdir(exist_ok=True)
        torch.save(model.base.state_dict(), adapter_dir / "base_model.bin")
        print("✓ Base model state saved")
    
    # Save custom heads (quantile_head and value_embedding)
    custom_heads = {
        'quantile_head.state_dict': model.quantile_head.state_dict(),
        'value_embedding.state_dict': model.value_embedding.state_dict(),
        'hidden_dim': model.hidden_dim,
        'base_model_id': base_model_id,
    }
    
    heads_path = output_dir / "heads.pt"
    print(f"Saving custom heads to {heads_path}...")
    torch.save(custom_heads, heads_path)
    print("✓ Custom heads saved")
    
    # Save model config
    config = {
        'model_type': 'ChronosVolatility',
        'base_model': base_model_id,
        'use_lora': True,
        'hidden_dim': model.hidden_dim,
        'quantiles': ['q10', 'q50', 'q90'],
        'seq_length': 60,
        'horizon': 20,
        'created_at': datetime.now().isoformat(),
    }
    
    config_path = output_dir / "config.json"
    print(f"Saving config to {config_path}...")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print("✓ Config saved")
    
    # Create model card
    if model_name is None:
        model_name = output_dir.name
    
    if description is None:
        description = f"""
A fine-tuned Chronos model for predicting stock market volatility. 
This model predicts quantiles (q10, q50, q90) of log-realized variance 
for a 20-day forward horizon using 60 days of historical squared returns.
"""
    
    card_content = create_model_card(
        model_name=model_name,
        base_model=base_model_id,
        description=description,
        training_info=training_info or checkpoint_meta
    )
    
    readme_path = output_dir / "README.md"
    print(f"Saving model card to {readme_path}...")
    with open(readme_path, 'w') as f:
        f.write(card_content)
    print("✓ Model card saved")
    
    # Create loading script example
    example_code = '''"""
Example script to load and use the exported ChronosVolatility model.
"""

import sys
from pathlib import Path

# Add parent directory to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import json
from src.models.chronos import ChronosVolatility

try:
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False
    print("Warning: peft and transformers required for loading")

def load_exported_model(model_dir):
    """
    Load exported model from directory.
    
    Args:
        model_dir: Path to exported model directory
    """
    if not _DEPS_AVAILABLE:
        raise ImportError("peft and transformers required. Install with: pip install peft transformers")
    
    model_dir = Path(model_dir)
    
    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    
    # Initialize base model
    print(f"Loading base model: {config['base_model']}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config['base_model'])
    
    # Load LoRA adapters
    adapter_path = model_dir / "adapter"
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        print("Loading LoRA adapters...")
        model_wrapper = PeftModel.from_pretrained(base_model, str(adapter_path))
    else:
        print("No adapter found, using base model")
        model_wrapper = base_model
    
    # Create ChronosVolatility wrapper (don't initialize LoRA, we'll set base manually)
    chronos_model = ChronosVolatility(use_lora=False)
    chronos_model.base = model_wrapper
    chronos_model.hidden_dim = config['hidden_dim']
    chronos_model.model_id = config['base_model']
    
    # Load custom heads
    print("Loading custom heads...")
    heads = torch.load(model_dir / "heads.pt", map_location='cpu')
    chronos_model.quantile_head.load_state_dict(heads['quantile_head.state_dict'])
    chronos_model.value_embedding.load_state_dict(heads['value_embedding.state_dict'])
    
    chronos_model.eval()
    print("✓ Model loaded successfully")
    return chronos_model

# Usage example:
# model = load_exported_model("path/to/exported/model")
# input_seq = torch.FloatTensor(squared_returns).unsqueeze(0)  # (1, 60)
# with torch.no_grad():
#     quantiles = model(input_seq)
'''
    
    example_path = output_dir / "example_usage.py"
    with open(example_path, 'w') as f:
        f.write(example_code)
    print("✓ Example usage script saved")
    
    print(f"\n{'='*60}")
    print("Export Complete!")
    print(f"{'='*60}")
    print(f"Model exported to: {output_dir}")
    print(f"\nContents:")
    print(f"  - adapter/       : LoRA adapter weights")
    print(f"  - heads.pt       : Custom head weights (quantile_head, value_embedding)")
    print(f"  - config.json    : Model configuration")
    print(f"  - README.md      : Model card")
    print(f"  - example_usage.py: Example loading script")
    
    # Push to Hub if requested
    if push_to_hub:
        if hub_repo_id is None:
            raise ValueError("hub_repo_id is required when push_to_hub=True")
        
        try:
            from huggingface_hub import HfApi, login
            import os
            
            print(f"\nPushing to Hugging Face Hub: {hub_repo_id}...")
            
            # Check for token
            token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
            if token:
                login(token=token)
            else:
                print("Warning: No HF_TOKEN found. Run 'huggingface-cli login' or set HF_TOKEN env var")
                login()
            
            api = HfApi()
            
            # Create repo if needed
            try:
                api.create_repo(repo_id=hub_repo_id, exist_ok=True, private=False)
            except Exception as e:
                print(f"Note: {e}")
            
            # Upload files
            api.upload_folder(
                folder_path=str(output_dir),
                repo_id=hub_repo_id,
                commit_message="Upload ChronosVolatility model"
            )
            
            print(f"✓ Successfully pushed to https://huggingface.co/{hub_repo_id}")
            
        except ImportError:
            print("\nWarning: huggingface_hub not installed. Install with: pip install huggingface_hub")
            print("Skipping Hub upload.")
        except Exception as e:
            print(f"\nError pushing to Hub: {e}")
            print("Model files are still available locally.")


def main():
    """Main export function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export ChronosVolatility model to Hugging Face format')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/checkpoints/chronos_5ticker.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/huggingface/chronos-volatility',
        help='Output directory for exported model'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Model name (default: infer from output directory)'
    )
    parser.add_argument(
        '--description',
        type=str,
        default=None,
        help='Model description'
    )
    parser.add_argument(
        '--push-to-hub',
        action='store_true',
        help='Push to Hugging Face Hub'
    )
    parser.add_argument(
        '--hub-repo-id',
        type=str,
        default=None,
        help='Hugging Face Hub repository ID (e.g., username/model-name)'
    )
    
    args = parser.parse_args()
    
    export_model(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        model_name=args.name,
        description=args.description,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
    )


if __name__ == '__main__':
    main()

