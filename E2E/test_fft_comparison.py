"""
Test if MLX FFT matches PyTorch FFT
"""
import torch
import numpy as np
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
STFT_DIR = PROJECT_ROOT / "test_audio" / "stft_dump"

# Load the source signals
python_data = load_file(str(STFT_DIR / "python_stft_input.safetensors"))
swift_data = load_file(str(STFT_DIR / "swift_stft_input.safetensors"))

python_source = python_data["source_signal"]  # [B, 1, T]
swift_source = swift_data["source_signal"]    # [B, T, 1]

print("=" * 80)
print("SOURCE SIGNAL COMPARISON")
print("=" * 80)

# Compare source signals first
print(f"\nPython source shape: {python_source.shape}")
print(f"Swift source shape: {swift_source.shape}")

# Transpose Swift to match Python
swift_source_t = swift_source.permute(0, 2, 1)  # [B, T, 1] -> [B, 1, T]

print(f"\nPython source range: [{python_source.min().item():.6f}, {python_source.max().item():.6f}]")
print(f"Swift source range: [{swift_source_t.min().item():.6f}, {swift_source_t.max().item():.6f}]")

# Correlation
source_corr = np.corrcoef(
    python_source.flatten().numpy(),
    swift_source_t.flatten().numpy()
)[0, 1]

print(f"\n{'✅' if source_corr > 0.99 else '❌'} Source signal correlation: {source_corr:.8f}")

if source_corr < 0.99:
    print("\n❌ Source signals don't match! This is a problem BEFORE STFT.")
    print("   The bug is in the source generation (m_source), not STFT.")
else:
    print("\n✅ Source signals match! The bug is in STFT processing.")

print("=" * 80)
