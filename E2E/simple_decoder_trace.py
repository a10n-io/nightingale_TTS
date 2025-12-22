#!/usr/bin/env python3
"""Simple decoder trace using saved inputs."""
import torch
from pathlib import Path
import safetensors.torch as st
import os

# Change to project root
os.chdir(Path(__file__).parent.parent)

# Load trace
trace_path = Path("test_audio/python_decoder_trace.safetensors")
trace = st.load_file(trace_path)

print("Loaded trace:")
print(f"  decoder_output: range=[{trace['decoder_output'].min():.6f}, {trace['decoder_output'].max():.6f}], mean={trace['decoder_output'].mean():.6f}")
print("\n✅ This is what Python decoder output should be!")
