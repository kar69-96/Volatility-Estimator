"""
Simple Chronos wrapper for volatility prediction.

Input: Single-channel time series (squared returns)
Output: Quantiles (q10, q50, q90) in log-variance space
"""

import torch
import torch.nn as nn
from pathlib import Path

try:
    from transformers import AutoModelForSeq2SeqLM
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


class ChronosVolatility(nn.Module):
    """
    Simple Chronos wrapper for volatility prediction.
    
    Input: Single-channel time series (squared returns) as float tensors
    Output: Quantiles (q10, q50, q90) in log-variance space
    
    Uses Chronos encoder to extract features from raw time series data.
    """
    
    def __init__(self, model_id='amazon/chronos-t5-mini', use_lora=True):
        super().__init__()
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required. Install with: pip install transformers>=4.40.0")
        
        # Load pretrained Chronos
        try:
            self.base = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        except ValueError as e:
            if "T5ForConditionalGeneration" in str(e) or "T5" in str(e):
                raise ImportError(
                    f"T5 models are not available in your transformers installation. "
                    f"This usually means transformers is too old or incomplete. "
                    f"Try: pip install --upgrade transformers>=4.40.0 sentencepiece\n"
                    f"Original error: {e}"
                )
            raise RuntimeError(
                f"Failed to load Chronos model '{model_id}'. "
                f"Original error: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Chronos model '{model_id}'. "
                f"Make sure transformers>=4.40.0 and sentencepiece are installed. "
                f"Original error: {e}"
            )
        self.model_id = model_id
        
        # Get hidden dimension from config
        # Try different possible config attributes
        hidden_dim = getattr(self.base.config, 'd_model', None)
        if hidden_dim is None:
            hidden_dim = getattr(self.base.config, 'hidden_size', None)
        if hidden_dim is None:
            # Fallback: try to get from encoder config
            if hasattr(self.base.config, 'encoder') and hasattr(self.base.config.encoder, 'hidden_size'):
                hidden_dim = self.base.config.encoder.hidden_size
            else:
                hidden_dim = 64  # Default fallback
        
        self.hidden_dim = hidden_dim
        
        # LoRA adapters
        if use_lora and _PEFT_AVAILABLE:
            # T5 models use different module names than GPT-style models
            # T5 uses: "q", "k", "v", "o" in T5Attention layers
            # We can use pattern matching or specify T5-specific modules
            # For T5, target the attention layers' q, k, v, o projections
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q", "k", "v", "o"],  # T5 attention projection names
                lora_dropout=0.1,
                task_type="SEQ_2_SEQ_LM"  # Specify task type for T5
            )
            self.base = get_peft_model(self.base, lora_config)
        elif use_lora and not _PEFT_AVAILABLE:
            raise ImportError("PEFT library is required for LoRA. Install with: pip install peft")
        
        # Quantile regression head (q10, q50, q90)
        self.quantile_head = nn.Linear(hidden_dim, 3)
        
        # Value embedding layer: maps scalar time series values to hidden_dim
        # This allows us to feed continuous values to the encoder
        self.value_embedding = nn.Linear(1, hidden_dim)
        
        # Freeze base model parameters (only train LoRA + heads)
        if use_lora:
            # With LoRA, only LoRA params are trainable
            pass
        else:
            # Without LoRA, freeze base model
            for param in self.base.parameters():
                param.requires_grad = False
        
    def forward(self, input_seq, attention_mask=None):
        """
        Forward pass.
        
        Args:
            input_seq: Raw time series (squared returns), shape (batch, seq_len)
            attention_mask: Optional attention mask
        
        Returns:
            Quantiles in log-variance space: (batch, 3) where columns are [q10, q50, q90]
        """
        # Ensure input is 2D: (batch, seq_len)
        if input_seq.dim() == 3:
            input_seq = input_seq.squeeze(-1)
        elif input_seq.dim() != 2:
            raise ValueError(f"Expected 2D input (batch, seq_len), got {input_seq.dim()}D")
        
        batch_size, seq_len = input_seq.shape
        
        # Embed time series values: (batch, seq_len) -> (batch, seq_len, hidden_dim)
        # Add feature dimension for embedding
        input_seq_3d = input_seq.unsqueeze(-1)  # (batch, seq_len, 1)
        input_emb = self.value_embedding(input_seq_3d)  # (batch, seq_len, hidden_dim)
        
        # Get encoder from base model
        encoder = self.base.get_encoder() if hasattr(self.base, 'get_encoder') else self.base.encoder
        
        # Pass through encoder
        # Use inputs_embeds to bypass tokenization
        encoder_outputs = encoder(
            inputs_embeds=input_emb,
            attention_mask=attention_mask
        )
        
        # Extract last hidden state from encoder
        # Use mean pooling over sequence dimension for better representation
        hidden_states = encoder_outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
        
        # Use mean pooling across sequence length for final representation
        if attention_mask is not None:
            # Mask out padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states = hidden_states * mask_expanded
            hidden = hidden_states.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            hidden = hidden_states.mean(dim=1)  # (batch, hidden_dim)
        
        # Predict quantiles
        quantiles = self.quantile_head(hidden)  # (batch, 3)
        
        return quantiles

