"""
The Hybrid Test: Load Swift's mel spectrogram and vocode it with Python.

This will tell us if the problem is:
1. Swift mel is wrong (decoder output formatting)
2. Swift vocoder is wrong (weights or implementation)
"""
import torch
import mlx.core as mx
from safetensors import safe_open

# Load Python S3Gen model
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("HYBRID TEST: Swift Mel → Python Vocoder")
print("=" * 80)

# Initialize Python model
print("\nLoading Python S3Gen model...")
model = ChatterboxMultilingualTTS(ckpt_dir="/Users/a10n/Projects/nightingale_TTS/models/chatterbox")
print("✅ Model loaded")

# Load Swift mel from cross-validation
print("\nLooking for Swift mel spectrogram from cross-validation...")

# The cross-validation should have saved the mel, but let me check
# Actually, let's load the Python tokens and run Swift S3Gen to get the mel
print("❌ We need to export Swift mel first")
print("\nTo complete this test:")
print("1. Modify Swift S3Gen to save the mel spectrogram before vocoding")
print("2. Re-run Swift cross-validation")
print("3. Load that mel here and vocode with Python")
print("\nFor now, let's verify the Python vocoder works on Python mel...")

# Generate Python mel
print("\nGenerating Python mel spectrogram...")
tokens_file = "/Users/a10n/Projects/nightingale_TTS/test_audio/cross_validate/python_speech_tokens.safetensors"
with safe_open(tokens_file, framework="pt") as f:
    python_tokens = f.get_tensor("speech_tokens")

print(f"Python tokens shape: {python_tokens.shape}")

# TODO: Generate mel with Python decoder
# For now, this test is incomplete - we need to export Swift mel first
print("\n⚠️  Test incomplete: Need to export Swift mel spectrogram")
print("   Modify synthesizeFromTokens() in ChatterboxEngine.swift to save mel before vocoding")
