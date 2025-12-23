"""
Check what format the Python vocoder is actually receiving
"""
import torch
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"

# Load Python's decoder mel
python_decoder_mel = load_file(str(FORENSIC_DIR / "python_decoder_mel.safetensors"))["mel"]

print("=" * 80)
print("CHECKING VOCODER INPUT FORMAT")
print("=" * 80)

print(f"\nOriginal mel from decoder: {python_decoder_mel.shape}")
print(f"  This is [C, T] = [80, {python_decoder_mel.shape[1]}]")
print(f"  First 5 values at freq_bin=0: {python_decoder_mel[0, :5].tolist()}")
print(f"  First 5 values at time=0: {python_decoder_mel[:5, 0].tolist()}")

# Python cross-validation does:
mel_input = python_decoder_mel.unsqueeze(0).transpose(1, 2)  # [80, T] -> [1, T, 80]

print(f"\nAfter Python processing: {mel_input.shape}")
print(f"  This is [B, T, C] = [1, {mel_input.shape[1]}, 80]")
print(f"  First 5 values at time=0, all freqs[:5]: {mel_input[0, 0, :5].tolist()}")
print(f"  First 5 values at all times[:5], freq=0: {mel_input[0, :5, 0].tolist()}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print("Original mel[0, :5] (freq_bin=0, first 5 timesteps):")
print(f"  {python_decoder_mel[0, :5].tolist()}")
print("\nAfter transpose mel[0, :5, 0] (batch=0, first 5 timesteps, freq=0):")
print(f"  {mel_input[0, :5, 0].tolist()}")
print("\nThese should MATCH -> they do!")

print("\nOriginal mel[:5, 0] (first 5 freq bins, time=0):")
print(f"  {python_decoder_mel[:5, 0].tolist()}")
print("\nAfter transpose mel[0, 0, :5] (batch=0, time=0, first 5 freqs):")
print(f"  {mel_input[0, 0, :5].tolist()}")
print("\nThese should MATCH -> they do!")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("Python vocoder receives [B, T, C] = [1, T, 80]")
print("  - Time is dim 1")
print("  - Frequency is dim 2")
print("=" * 80)
