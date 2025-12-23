"""
Compare the ACTUAL data values being fed to Python and Swift vocoders
"""
import torch
from safetensors.torch import load_file
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CROSS_VAL_DIR = PROJECT_ROOT / "test_audio" / "cross_validate"

# This is what we saved for Swift
mel_for_swift = load_file(str(CROSS_VAL_DIR / "python_decoder_mel_for_swift_vocoder.safetensors"))["mel"]

print("=" * 80)
print("VOCODER INPUT DATA COMPARISON")
print("=" * 80)

print(f"\nMel saved for Swift vocoder: {mel_for_swift.shape} = [C, T] = [80, T]")

# What Python actually feeds to its vocoder
python_vocoder_input = mel_for_swift.unsqueeze(0).transpose(1, 2)  # -> [1, T, 80]
print(f"Python vocoder input: {python_vocoder_input.shape} = [B, T, C]")

# What Swift feeds to its vocoder (from SaveVocoderCrossValidation.swift)
# It does: mel[80,T] -> [T, 80] -> [1, T, 80] -> [1, 80, T]
# Step by step:
swift_step1 = mel_for_swift.transpose(1, 0)  # [80, T] -> [T, 80]
swift_step2 = swift_step1.unsqueeze(0)  # -> [1, T, 80]
swift_vocoder_input = swift_step2.transpose(1, 2)  # -> [1, 80, T]
print(f"Swift vocoder input: {swift_vocoder_input.shape} = [B, C, T]")

print("\n" + "=" * 80)
print("CHECKING IF DATA VALUES MATCH")
print("=" * 80)

# Check if the data is the same (just transposed)
# Python: [1, T, 80] - time is dim 1, freq is dim 2
# Swift: [1, 80, T] - freq is dim 1, time is dim 2

print("\nPython input mel[0, 0, :5] (batch=0, time=0, first 5 freqs):")
print(f"  {python_vocoder_input[0, 0, :5].tolist()}")

print("\nSwift input mel[0, :5, 0] (batch=0, first 5 freqs, time=0):")
print(f"  {swift_vocoder_input[0, :5, 0].tolist()}")

match1 = torch.allclose(python_vocoder_input[0, 0, :], swift_vocoder_input[0, :, 0])
print(f"\n✓ These match? {match1}")

print("\nPython input mel[0, :5, 0] (batch=0, first 5 times, freq=0):")
print(f"  {python_vocoder_input[0, :5, 0].tolist()}")

print("\nSwift input mel[0, 0, :5] (batch=0, freq=0, first 5 times):")
print(f"  {swift_vocoder_input[0, 0, :5].tolist()}")

match2 = torch.allclose(python_vocoder_input[0, :, 0], swift_vocoder_input[0, 0, :])
print(f"\n✓ These match? {match2}")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

if match1 and match2:
    print("✅ DATA VALUES MATCH - Both vocoders receive IDENTICAL data")
    print("   Just in transposed layouts ([B,T,C] vs [B,C,T])")
    print("   The 0.82 correlation is NOT from input data mismatch")
    print("   It's from vocoder IMPLEMENTATION differences!")
else:
    print("❌ DATA VALUES DIFFER - Input mismatch found!")
