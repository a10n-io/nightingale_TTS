#!/usr/bin/env python3
"""Test Python vocoder with saved mel spectrogram."""
import torch
from pathlib import Path
import safetensors.torch as st
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import wave
import numpy as np

# Load model
MODELS_DIR = Path("models/chatterbox")
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading Chatterbox model...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load mel spectrogram
mel_path = Path("test_audio/test_mel.safetensors")
mel_data = st.load_file(mel_path)
mel = mel_data["mel"].to(device)  # [1, 80, 248]

print(f"Mel shape: {mel.shape}")
print(f"Mel range: [{mel.min():.6f}, {mel.max():.6f}]")
print(f"Mel mean: {mel.mean():.6f}")

# Run vocoder
print("\nRunning Python vocoder...")
vocoder = model.s3gen.flow.decoder.vocoder
with torch.no_grad():
    audio = vocoder(mel)  # [1, T_audio]

print(f"Audio shape: {audio.shape}")
print(f"Audio range: [{audio.min():.6f}, {audio.max():.6f}]")
print(f"Audio mean: {audio.mean():.6f}")

# Calculate RMS
rms = torch.sqrt(torch.mean(audio ** 2)).item()
print(f"Audio RMS: {rms:.6f}")

# Save audio
audio_np = audio.squeeze(0).cpu().numpy()
output_path = Path("test_audio/python_vocoder_test.wav")

with wave.open(str(output_path), 'wb') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)  # 16-bit
    wav.setframerate(24000)
    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
    wav.writeframes(audio_int16.tobytes())

print(f"\n✅ Saved audio to: {output_path}")
print(f"Duration: {len(audio_np) / 24000:.2f}s")
