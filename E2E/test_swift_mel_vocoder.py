#!/usr/bin/env python3
"""Test Swift-generated mel with Python vocoder to isolate the issue."""

import numpy as np
import torch
from safetensors import safe_open
import scipy.io.wavfile as wavfile

# Load Python vocoder - use the vocoder directly
print("Loading Python vocoder...")
from chatterbox.models.s3gen.hifigan import HiFTGenerator
import json

# Load vocoder config and weights
vocoder_config_path = '/Users/a10n/Projects/nightingale_TTS/models/exp/s3gen/vocoder_config.json'
vocoder_weights_path = '/Users/a10n/Projects/nightingale_TTS/models/exp/s3gen/vocoder.pth'

with open(vocoder_config_path) as f:
    vocoder_config = json.load(f)

vocoder = HiFTGenerator(**vocoder_config['model'])
checkpoint = torch.load(vocoder_weights_path, map_location='cpu')
vocoder.load_state_dict(checkpoint['generator'])
vocoder.eval()
print("  Vocoder loaded successfully!")

# Load Swift-generated mel
print("\nLoading Swift-generated mel...")
with safe_open('/Users/a10n/Projects/nightingale_TTS/E2E/swift_generated_mel_raw.safetensors', framework="numpy") as f:
    swift_mel = f.get_tensor("mel")

print(f"Swift mel shape: {swift_mel.shape}")
print(f"Swift mel range: [{swift_mel.min():.4f}, {swift_mel.max():.4f}]")
print(f"Swift mel mean: {swift_mel.mean():.4f}")
print(f"Swift mel std: {swift_mel.std():.4f}")

# Analyze frequency structure
print("\nSwift mel channel energies:")
for i in [0, 10, 20, 30, 40, 50, 60, 70, 79]:
    energy = swift_mel[0, i, :].mean()
    print(f"  Channel {i:2d}: {energy:.4f}")

# Convert to torch and try vocoding directly
print("\n1. Testing Swift mel WITHOUT transformation...")
swift_mel_torch = torch.from_numpy(swift_mel).float()
try:
    with torch.no_grad():
        audio_no_transform = vocoder(swift_mel_torch)
    audio_np = audio_no_transform.squeeze().cpu().numpy()
    wavfile.write('/Users/a10n/Projects/nightingale_TTS/E2E/swift_mel_no_transform.wav', 24000, audio_np)
    print("  ✅ Saved: E2E/swift_mel_no_transform.wav")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Try with linear transformation (swift uses mel * 0.75 - 5.0)
print("\n2. Testing Swift mel WITH current transform (mel * 0.75 - 5.0)...")
swift_mel_transformed = swift_mel * 0.75 - 5.0
swift_mel_transformed = np.clip(swift_mel_transformed, -10.0, -1.0)
swift_mel_transformed_torch = torch.from_numpy(swift_mel_transformed).float()
try:
    with torch.no_grad():
        audio_transformed = vocoder(swift_mel_transformed_torch)
    audio_np = audio_transformed.squeeze().cpu().numpy()
    wavfile.write('/Users/a10n/Projects/nightingale_TTS/E2E/swift_mel_with_transform.wav', 24000, audio_np)
    print("  ✅ Saved: E2E/swift_mel_with_transform.wav")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Load Python reference mel for comparison
print("\n3. Loading Python reference mel...")
try:
    python_mel = np.load('/Users/a10n/Projects/nightingale_TTS/E2E/reference_outputs/samantha/expressive_surprise_en/step8_python_audio.npy', allow_pickle=True).item()
    # Actually the mel should be in a different file
    # Let me check what we have
    import os
    ref_dir = '/Users/a10n/Projects/nightingale_TTS/E2E/reference_outputs/samantha/expressive_surprise_en'
    files = [f for f in os.listdir(ref_dir) if 'mel' in f.lower() and f.endswith('.npy')]
    print(f"  Available mel files: {files}")

    if 'test_mel_BTC.npy' in files:
        python_mel = np.load(os.path.join(ref_dir, 'test_mel_BTC.npy'))
        print(f"  Python mel shape: {python_mel.shape}")
        print(f"  Python mel range: [{python_mel.min():.4f}, {python_mel.max():.4f}]")
        print(f"  Python mel mean: {python_mel.mean():.4f}")

        print("\n  Python mel channel energies:")
        for i in [0, 10, 20, 30, 40, 50, 60, 70, 79]:
            energy = python_mel[0, i, :].mean()
            print(f"    Channel {i:2d}: {energy:.4f}")
except Exception as e:
    print(f"  Could not load Python reference: {e}")

print("\n✅ Done! Check the generated audio files:")
print("  - E2E/swift_mel_no_transform.wav (raw Swift mel)")
print("  - E2E/swift_mel_with_transform.wav (with current transformation)")
