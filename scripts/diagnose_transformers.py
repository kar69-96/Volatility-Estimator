#!/usr/bin/env python3
"""Diagnostic script to check transformers installation and T5 availability."""

import sys

print("=" * 60)
print("TRANSFORMERS DIAGNOSTIC REPORT")
print("=" * 60)

# Check transformers version
try:
    import transformers
    print(f"\n✓ transformers installed: version {transformers.__version__}")
except ImportError as e:
    print(f"\n✗ transformers NOT installed: {e}")
    sys.exit(1)

# Check T5 module
print("\n" + "-" * 60)
print("Checking T5 module availability...")
try:
    from transformers.models import t5
    print(f"✓ transformers.models.t5 module found: {t5.__file__}")
    
    # List all attributes
    t5_attrs = [x for x in dir(t5) if not x.startswith('_')]
    print(f"  Available attributes ({len(t5_attrs)}): {', '.join(t5_attrs[:20])}{'...' if len(t5_attrs) > 20 else ''}")
    
    # Check for T5ForConditionalGeneration specifically
    if hasattr(t5, 'T5ForConditionalGeneration'):
        print("  ✓ T5ForConditionalGeneration found in t5 module")
    else:
        print("  ✗ T5ForConditionalGeneration NOT found in t5 module")
        
except ImportError as e:
    print(f"✗ Failed to import transformers.models.t5: {e}")

# Check T5 modeling
print("\n" + "-" * 60)
print("Checking T5 modeling module...")
try:
    from transformers.models.t5 import modeling_t5
    print(f"✓ transformers.models.t5.modeling_t5 found: {modeling_t5.__file__}")
    
    modeling_attrs = [x for x in dir(modeling_t5) if 'T5' in x and not x.startswith('_')]
    print(f"  T5-related classes: {', '.join(modeling_attrs) if modeling_attrs else 'None found'}")
    
    if hasattr(modeling_t5, 'T5ForConditionalGeneration'):
        print("  ✓ T5ForConditionalGeneration found in modeling_t5")
    else:
        print("  ✗ T5ForConditionalGeneration NOT found in modeling_t5")
        
except ImportError as e:
    print(f"✗ Failed to import modeling_t5: {e}")

# Check AutoModelForSeq2SeqLM
print("\n" + "-" * 60)
print("Checking AutoModelForSeq2SeqLM...")
try:
    from transformers import AutoModelForSeq2SeqLM
    print(f"✓ AutoModelForSeq2SeqLM imported successfully")
    
    # Check model mappings
    if hasattr(AutoModelForSeq2SeqLM, '_model_mapping'):
        mapping = AutoModelForSeq2SeqLM._model_mapping
        print(f"  Model mapping found with {len(mapping)} entries")
        t5_entries = {k: v for k, v in mapping.items() if 'T5' in str(k) or 'T5' in str(v)}
        if t5_entries:
            print(f"  T5-related mappings: {t5_entries}")
        else:
            print("  ✗ No T5 entries in model mapping")
    else:
        print("  ✗ No _model_mapping attribute")
        
except ImportError as e:
    print(f"✗ Failed to import AutoModelForSeq2SeqLM: {e}")

# Try to manually import T5ForConditionalGeneration
print("\n" + "-" * 60)
print("Attempting direct T5ForConditionalGeneration import...")
for import_path in [
    'transformers.models.t5.T5ForConditionalGeneration',
    'transformers.models.t5.modeling_t5.T5ForConditionalGeneration',
    'transformers.T5ForConditionalGeneration',
]:
    try:
        parts = import_path.split('.')
        module_name = '.'.join(parts[:-1])
        class_name = parts[-1]
        module = __import__(module_name, fromlist=[class_name])
        if hasattr(module, class_name):
            print(f"  ✓ Found via: {import_path}")
            break
    except Exception as e:
        pass
else:
    print("  ✗ T5ForConditionalGeneration not found via any import path")

# Check what model classes are available
print("\n" + "-" * 60)
print("Checking available Seq2Seq model classes...")
try:
    from transformers.models.auto import auto_factory
    if hasattr(auto_factory, 'MODEL_MAPPING'):
        print("  Checking MODEL_MAPPING...")
        # This might give us clues
    print("  (Model mapping check completed)")
except Exception as e:
    print(f"  Could not check auto_factory: {e}")

# Try to load config for chronos model
print("\n" + "-" * 60)
print("Attempting to load Chronos config...")
try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained('amazon/chronos-t5-mini')
    print(f"  ✓ Config loaded successfully")
    print(f"    Model type: {config.model_type}")
    print(f"    Config class: {type(config).__name__}")
except Exception as e:
    print(f"  ✗ Failed to load config: {e}")

# Final recommendation
print("\n" + "=" * 60)
print("RECOMMENDATION:")
print("=" * 60)
print("If T5ForConditionalGeneration is missing, try:")
print("  1. pip uninstall transformers -y")
print("  2. pip install transformers>=4.40.0 --no-cache-dir")
print("  3. Verify with: python -c 'from transformers.models.t5 import T5ForConditionalGeneration; print(\"OK\")'")
print("=" * 60)

