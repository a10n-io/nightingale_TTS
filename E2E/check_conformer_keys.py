"""
Check actual Conformer block keys in Python model
"""
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"

# Load Python weights
flow_weights = load_file(str(MODELS_DIR / "s3gen.safetensors"))

# Find all encoder.encoders keys
encoder_keys = [k for k in flow_weights.keys() if k.startswith("flow.encoder.encoders")]
encoder_keys.sort()

print("=" * 80)
print("PYTHON ENCODER.ENCODERS KEYS (first 50)")
print("=" * 80)
for key in encoder_keys[:50]:
    print(f"  {key}")

# Check specific keys for encoder.encoders.0
print("\n" + "=" * 80)
print("ENCODER.ENCODERS.0 KEYS (first block)")
print("=" * 80)
encoder_0_keys = [k for k in encoder_keys if ".encoders.0." in k or "encoders_0." in k]
for key in encoder_0_keys[:30]:
    print(f"  {key}")
