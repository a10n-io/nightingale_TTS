#!/usr/bin/env python3
"""Analyze single audio file."""
import wave
import numpy as np
from pathlib import Path

filepath = Path("test_audio/chatterbox_engine_test.wav")

with wave.open(str(filepath), 'rb') as wav:
    n_channels = wav.getnchannels()
    sampwidth = wav.getsampwidth()
    framerate = wav.getframerate()
    n_frames = wav.getnframes()
    frames = wav.readframes(n_frames)

# Convert to numpy array
if sampwidth == 2:
    dtype = np.int16
    data = np.frombuffer(frames, dtype=dtype)
    data = data.astype(np.float32) / 32768.0
else:
    raise ValueError(f"Unsupported sample width: {sampwidth}")

# Basic stats
duration = n_frames / framerate
rms = np.sqrt(np.mean(data ** 2))
peak = np.max(np.abs(data))

# FFT analysis
fft = np.fft.rfft(data)
freqs = np.fft.rfftfreq(len(data), 1/framerate)
magnitudes = np.abs(fft)

# Find dominant frequencies
top_indices = np.argsort(magnitudes)[-10:][::-1]
dominant_freqs = freqs[top_indices]

# Speech has more energy in lower frequencies (100-3000 Hz)
speech_band_mask = (freqs >= 100) & (freqs <= 3000)
speech_energy = np.sum(magnitudes[speech_band_mask])

# High frequencies (3000-8000 Hz)
high_band_mask = (freqs >= 3000) & (freqs <= 8000)
high_energy = np.sum(magnitudes[high_band_mask])

total_energy = np.sum(magnitudes)

print(f"\nFile: {filepath.name}")
print(f"Duration: {duration:.2f}s")
print(f"RMS: {rms:.6f}")
print(f"Peak: {peak:.6f}")
print(f"\nEnergy distribution:")
print(f"  Speech band (100-3000 Hz): {100*speech_energy/total_energy:.1f}%")
print(f"  High band (3000-8000 Hz): {100*high_energy/total_energy:.1f}%")
print(f"\nTop 5 dominant frequencies:")
for i, freq in enumerate(dominant_freqs[:5]):
    print(f"  {i+1}. {freq:.1f} Hz (magnitude: {magnitudes[top_indices[i]]:.2e})")

if magnitudes[top_indices[0]] > 10 * magnitudes[top_indices[1]]:
    print(f"\n⚠️  WARNING: Looks like a SINGLE TONE")
else:
    print(f"\n✅ Looks like ACTUAL AUDIO (broad spectrum)")
