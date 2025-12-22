#!/usr/bin/env python3
"""Analyze audio files to check if they're actual speech or just tones."""
import wave
import numpy as np
from pathlib import Path
import struct

def analyze_wav(filepath):
    """Analyze a WAV file for basic characteristics."""
    with wave.open(str(filepath), 'rb') as wav:
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        framerate = wav.getframerate()
        n_frames = wav.getnframes()

        # Read all frames
        frames = wav.readframes(n_frames)

    # Convert to numpy array
    if sampwidth == 1:
        dtype = np.uint8
        data = np.frombuffer(frames, dtype=dtype)
        data = (data.astype(np.float32) - 128) / 128.0
    elif sampwidth == 2:
        dtype = np.int16
        data = np.frombuffer(frames, dtype=dtype)
        data = data.astype(np.float32) / 32768.0
    elif sampwidth == 4:
        dtype = np.int32
        data = np.frombuffer(frames, dtype=dtype)
        data = data.astype(np.float32) / 2147483648.0
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

    # Check if it's a single tone (one dominant frequency with harmonics)
    # vs actual speech (broad frequency spectrum)
    freq_variance = np.var(magnitudes)

    # Speech has more energy in lower frequencies (100-3000 Hz)
    speech_band_mask = (freqs >= 100) & (freqs <= 3000)
    speech_energy = np.sum(magnitudes[speech_band_mask])

    # High frequencies (3000-8000 Hz)
    high_band_mask = (freqs >= 3000) & (freqs <= 8000)
    high_energy = np.sum(magnitudes[high_band_mask])

    total_energy = np.sum(magnitudes)

    print(f"\n{'='*60}")
    print(f"File: {filepath.name}")
    print(f"{'='*60}")
    print(f"Duration: {duration:.2f}s")
    print(f"Sample rate: {framerate} Hz")
    print(f"Channels: {n_channels}")
    print(f"RMS: {rms:.6f}")
    print(f"Peak: {peak:.6f}")
    print(f"Frequency variance: {freq_variance:.2e}")
    print(f"\nEnergy distribution:")
    print(f"  Speech band (100-3000 Hz): {100*speech_energy/total_energy:.1f}%")
    print(f"  High band (3000-8000 Hz): {100*high_energy/total_energy:.1f}%")
    print(f"\nTop 5 dominant frequencies:")
    for i, freq in enumerate(dominant_freqs[:5]):
        print(f"  {i+1}. {freq:.1f} Hz (magnitude: {magnitudes[top_indices[i]]:.2e})")

    # Check for single tone
    if magnitudes[top_indices[0]] > 10 * magnitudes[top_indices[1]]:
        print(f"\n⚠️  WARNING: Looks like a SINGLE TONE (one freq dominates)")
    else:
        print(f"\n✅ Looks like ACTUAL AUDIO (broad spectrum)")

# Analyze the files
cross_val_dir = Path("test_audio/cross_validate")

print("\n" + "="*60)
print("AUDIO ANALYSIS: Single Tone vs Actual Speech")
print("="*60)

files = [
    cross_val_dir / "python_tokens_python_audio.wav",  # Baseline
    cross_val_dir / "swift_tokens_swift_audio.wav",     # Full Swift pipeline
    cross_val_dir / "python_tokens_swift_audio.wav",    # Swift S3Gen only
    Path("test_audio/chatterbox_engine_test.wav"),      # GenerateAudio output
]

for filepath in files:
    if filepath.exists():
        analyze_wav(filepath)
    else:
        print(f"\n⚠️  File not found: {filepath}")
