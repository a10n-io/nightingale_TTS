#!/usr/bin/env python3
"""Save a mel spectrogram for vocoder testing."""
import torch
from pathlib import Path
import safetensors.torch as st

# Load Python decoder trace
trace_path = Path("test_audio/python_decoder_trace.safetensors")
trace = st.load_file(trace_path)

# The decoder output is already a mel spectrogram [1, 80, T]
decoder_output = trace["decoder_output"]  # [1, 80, 696]

# Take a smaller chunk for faster testing
mel = decoder_output[:, :, :248]  # [1, 80, 248] - matches typical audio length

print(f"Mel spectrogram shape: {mel.shape}")
print(f"Mel range: [{mel.min():.6f}, {mel.max():.6f}]")
print(f"Mel mean: {mel.mean():.6f}")

# Save for Swift to load
output_path = Path("test_audio/test_mel.safetensors")
st.save_file({"mel": mel.contiguous()}, output_path)
print(f"\n✅ Saved mel spectrogram to: {output_path}")
