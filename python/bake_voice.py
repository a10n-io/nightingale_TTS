#!/usr/bin/env python3
"""
Bake voice embeddings from reference audio for Chatterbox Multilingual TTS.
This pre-computes all voice embeddings at full precision for faster inference.

Usage:
    python bake_voice.py --voice samantha
    python bake_voice.py --voice sujano --exag 0.5
"""

import torch
import librosa
from pathlib import Path
import argparse
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Parse arguments
parser = argparse.ArgumentParser(description="Bake voice embeddings from reference audio")
parser.add_argument("--voice", "-v", required=True,
                    help="Voice name (e.g., 'samantha', 'sujano'). Will use baked_voices/{name}/ref_audio.wav")
parser.add_argument("--exag", "-e", type=float, default=0.5,
                    help="Exaggeration parameter (default: 0.5)")
args = parser.parse_args()

# Paths
VOICE_DIR = Path(f"/Users/a10n/Projects/nightingale_TTS/baked_voices/{args.voice}")
REF_AUDIO = VOICE_DIR / "ref_audio.wav"
OUTPUT_FILE = VOICE_DIR / "baked_voice.pt"

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Check reference audio
if not REF_AUDIO.exists():
    raise FileNotFoundError(f"Reference audio not found: {REF_AUDIO}")

# Get audio duration
wav, sr = librosa.load(str(REF_AUDIO), sr=None)
duration = len(wav) / sr
print(f"Reference audio: {REF_AUDIO}")
print(f"Duration: {duration:.2f}s")
print(f"Sample rate: {sr}Hz")

# Load model from local directory
MODEL_DIR = "/Users/a10n/Projects/nightingale_TTS/models/chatterbox"
print(f"\nLoading Chatterbox Multilingual TTS from: {MODEL_DIR}")
model = ChatterboxMultilingualTTS.from_local(MODEL_DIR, device=device)
print("Model loaded successfully")

# Bake voice embeddings at full precision
print(f"\nBaking voice embeddings from reference audio...")
print(f"Exaggeration: {args.exag}")
model.prepare_conditionals(str(REF_AUDIO), exaggeration=args.exag)

# Save the baked voice conditionals
print(f"\nSaving baked voice to: {OUTPUT_FILE}")
model.conds.save(OUTPUT_FILE)

# Verify the saved file
file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
print(f"✓ Baked voice saved successfully ({file_size_mb:.2f} MB)")

# Test loading
print("\nVerifying saved voice...")
from chatterbox.mtl_tts import Conditionals
loaded_conds = Conditionals.load(OUTPUT_FILE, map_location=device)
print("✓ Baked voice can be loaded successfully")

print("\n" + "="*60)
print("Voice baking complete!")
print("="*60)
print(f"\nTo use this baked voice in your code:")
print(f"```python")
print(f"from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals")
print(f"")
print(f"model = ChatterboxMultilingualTTS.from_pretrained(device='mps')")
print(f"model.conds = Conditionals.load('{OUTPUT_FILE}', map_location='mps')")
print(f"")
print(f"# Now generate without audio_prompt_path")
print(f"wav = model.generate('Hello world', language_id='en')")
print(f"```")
