"""Trace Python vocoder to understand the correct flow."""
import sys
sys.path.insert(0, "/Users/a10n/Projects/nightingale_TTS/python/chatterbox/src")

import numpy as np
import torch
from pathlib import Path
import inspect

print("Loading model...")
from chatterbox.tts import ChatterboxTTS
model = ChatterboxTTS.from_pretrained(device="mps")

s3gen = model.s3gen

# Check mel2wav
print(f"\n--- Checking mel2wav ---")
if hasattr(s3gen, 'mel2wav'):
    mel2wav = s3gen.mel2wav
    print(f"mel2wav type: {type(mel2wav).__name__}")
    
    # Check the inference method
    if hasattr(mel2wav, 'inference'):
        print("\n--- mel2wav.inference source ---")
        source = inspect.getsource(mel2wav.inference)
        lines = source.split('\n')[:40]
        for i, line in enumerate(lines):
            print(f"{i+1:3}: {line}")
    
    # Check decode method
    if hasattr(mel2wav, 'decode'):
        print("\n--- mel2wav.decode source ---")
        source = inspect.getsource(mel2wav.decode)
        lines = source.split('\n')[:60]
        for i, line in enumerate(lines):
            print(f"{i+1:3}: {line}")
