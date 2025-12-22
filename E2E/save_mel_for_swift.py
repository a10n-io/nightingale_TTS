"""Save mel spectrogram before vocoder for Swift testing."""
import sys
sys.path.insert(0, "/Users/a10n/Projects/nightingale_TTS/python/chatterbox/src")

import numpy as np
import torch
from pathlib import Path

OUTPUT_DIR = Path("/Users/a10n/Projects/nightingale_TTS/E2E/reference_outputs/samantha/expressive_surprise_en")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create a simple test mel spectrogram
# Shape: [1, 80, 50] - 1 batch, 80 mel channels, 50 time frames
B, C, T = 1, 80, 50

# Use a simple pattern that will make frequency issues obvious
# Low frequencies should be high energy, high frequencies low energy
mel = np.zeros((B, C, T), dtype=np.float32)

# Set up a frequency gradient - mel bin 0 (low freq) = high energy, mel bin 79 (high freq) = low energy
for c in range(C):
    # Value decreases from -2 (low freq) to -8 (high freq)
    # This simulates log-mel where low freqs have more energy
    mel[:, c, :] = -2 - (c / C) * 6

print(f"Test mel shape: {mel.shape}")
print(f"Test mel range: [{mel.min():.4f}, {mel.max():.4f}]")
print(f"Mel bin 0 (low freq) mean: {mel[0, 0, :].mean():.4f}")
print(f"Mel bin 79 (high freq) mean: {mel[0, 79, :].mean():.4f}")

# Save in different formats for Swift testing
np.save(OUTPUT_DIR / "test_mel_BCT.npy", mel)  # [B, C, T] format (Python/PyTorch)
np.save(OUTPUT_DIR / "test_mel_BTC.npy", mel.transpose(0, 2, 1))  # [B, T, C] format

print(f"\nSaved test mel spectrograms to {OUTPUT_DIR}")
print(f"  - test_mel_BCT.npy: [B, C, T] = {mel.shape}")
print(f"  - test_mel_BTC.npy: [B, T, C] = {mel.transpose(0, 2, 1).shape}")

# Now run through Python vocoder
print("\nRunning Python vocoder...")
from chatterbox.tts import ChatterboxTTS
model = ChatterboxTTS.from_pretrained(device="mps")

mel_tensor = torch.from_numpy(mel).to("mps")
s3gen = model.s3gen

# Run through vocoder
with torch.no_grad():
    wav, _ = s3gen.hift_inference(mel_tensor)

wav_np = wav.squeeze().cpu().numpy()
print(f"Python vocoder output: {wav_np.shape}, range=[{wav_np.min():.4f}, {wav_np.max():.4f}]")

# Analyze frequency content
from scipy.fft import rfft, rfftfreq
yf = np.abs(rfft(wav_np))
xf = rfftfreq(len(wav_np), 1/24000)
dom_freq = xf[np.argmax(yf)]
print(f"Dominant frequency: {dom_freq:.1f} Hz")

# Save audio
np.save(OUTPUT_DIR / "test_vocoder_python.npy", wav_np)
print(f"Saved: {OUTPUT_DIR / 'test_vocoder_python.npy'}")

print("\nDone! Now run Swift with the same test mel to compare.")
