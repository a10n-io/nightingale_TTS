#!/usr/bin/env python3
"""
Test the baked voice by generating a sample audio file.
"""

import torch
import torchaudio as ta
from pathlib import Path
from datetime import datetime
import random
import numpy as np
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")
print(f"Random seed: {SEED}")

# Load model from local directory
MODEL_DIR = "/Users/a10n/Projects/nightingale_TTS/models/chatterbox"
print(f"Loading Chatterbox Multilingual TTS from: {MODEL_DIR}")
model = ChatterboxMultilingualTTS.from_local(MODEL_DIR, device=device)

# Load baked voice
BAKED_VOICE = "/Users/a10n/Projects/nightingale_TTS/baked_voices/samantha/baked_voice.pt"
print(f"Loading baked voice from: {BAKED_VOICE}")
model.conds = Conditionals.load(BAKED_VOICE, map_location=device)

# Test text in English and Dutch
test_cases = [
    ("Hello, this is a test of the baked voice system.", "en", "english"),
    ("Hallo, dit is een test van het ingebakken stemsysteem.", "nl", "dutch"),
]

output_dir = Path("/Users/a10n/Projects/nightingale_TTS/test_audio")
output_dir.mkdir(exist_ok=True)

# Generate timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("\nGenerating test audio files...")
for text, lang, lang_name in test_cases:
    print(f"\n{lang.upper()}: {text}")
    wav = model.generate(text, language_id=lang)
    filename = f"python_test_{lang_name}_{timestamp}.wav"
    output_path = output_dir / filename
    ta.save(str(output_path), wav, model.sr)
    print(f"✓ Saved to: {output_path}")

print("\n" + "="*60)
print("Test complete! Check the test_audio folder for output files.")
print("="*60)
