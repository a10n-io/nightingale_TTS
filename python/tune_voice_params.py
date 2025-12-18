#!/usr/bin/env python3
"""
Test different parameter combinations to find the best match for the reference voice.
Chatterbox has several parameters that affect voice similarity:
- exaggeration: Controls expressiveness (0.0-1.0, default 0.5)
- cfg_weight: Classifier-free guidance weight (0.0-1.0, default 0.5)
- temperature: Sampling temperature (default 0.8)

Usage:
    python tune_voice_params.py --voice samantha
    python tune_voice_params.py --voice sujano
"""

import torch
import torchaudio as ta
from pathlib import Path
import argparse
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Parse arguments
parser = argparse.ArgumentParser(description="Test parameter combinations for voice matching")
parser.add_argument("--voice", "-v", required=True,
                    help="Voice name (e.g., 'samantha', 'sujano'). Will use baked_voices/{name}/ref_audio.wav")
args = parser.parse_args()

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Voice-specific paths
VOICE_DIR = Path(f"/Users/a10n/Projects/nightingale_TTS/baked_voices/{args.voice}")
REF_AUDIO = VOICE_DIR / "ref_audio.wav"
OUTPUT_DIR = VOICE_DIR / "tuning"

if not REF_AUDIO.exists():
    print(f"Error: Reference audio not found at: {REF_AUDIO}")
    exit(1)

OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Voice: {args.voice}")
print(f"Reference audio: {REF_AUDIO}")

# Load model from local directory
MODEL_DIR = "/Users/a10n/Projects/nightingale_TTS/models/chatterbox"
print(f"Loading Chatterbox Multilingual TTS from: {MODEL_DIR}")
model = ChatterboxMultilingualTTS.from_local(MODEL_DIR, device=device)

# Test text
test_text = "Hello, this is a test of different parameter settings to match the reference voice."

# Test different parameter combinations
# According to the docs:
# - Lower cfg_weight (e.g., 0.3) can improve pacing for fast speakers
# - Higher exaggeration with lower cfg_weight for more expressive speech
# - Default settings (exaggeration=0.5, cfg_weight=0.5) work well for most

test_configs = [
    # (exaggeration, cfg_weight, temperature, description)
    (0.5, 0.5, 0.8, "default"),
    (0.3, 0.3, 0.8, "lower_both"),
    (0.7, 0.3, 0.8, "expressive"),
    (0.5, 0.3, 0.8, "fast_speaker"),
    (0.5, 0.7, 0.8, "higher_cfg"),
    (0.3, 0.5, 0.8, "lower_exag"),
    (0.7, 0.5, 0.8, "higher_exag"),
    (0.5, 0.5, 1.0, "higher_temp"),
    (0.5, 0.5, 0.6, "lower_temp"),
    (0.0, 0.0, 0.8, "minimal_guidance"),
]

print(f"\nGenerating {len(test_configs)} variations...")
print("This will help identify the best parameter combination for your voice.\n")

for exag, cfg, temp, desc in test_configs:
    print(f"Testing {desc}: exaggeration={exag}, cfg_weight={cfg}, temperature={temp}")

    # Prepare conditionals with current exaggeration
    model.prepare_conditionals(REF_AUDIO, exaggeration=exag)

    # Generate with current parameters
    wav = model.generate(
        test_text,
        language_id="en",
        exaggeration=exag,
        cfg_weight=cfg,
        temperature=temp,
    )

    # Save output
    output_path = OUTPUT_DIR / f"test_exag{exag}_cfg{cfg}_temp{temp}_{desc}.wav"
    ta.save(str(output_path), wav, model.sr)
    print(f"  ✓ Saved: {output_path.name}\n")

print("="*60)
print("Tuning complete!")
print("="*60)
print(f"\nAll variations saved to: {OUTPUT_DIR}")
print("\nListen to each file and compare with your reference audio.")
print("The filename indicates the parameters used.")
print("\nParameter guide:")
print("  exaggeration: 0.0-1.0 (controls expressiveness)")
print("  cfg_weight: 0.0-1.0 (classifier-free guidance)")
print("  temperature: 0.6-1.0 (sampling randomness)")
print("\nTips:")
print("  - Lower cfg_weight (0.3) if reference has fast speaking style")
print("  - Higher exaggeration (0.7) for more dramatic/expressive speech")
print("  - Lower exaggeration (0.3) for more neutral delivery")
print("  - cfg_weight=0.0 helps with cross-language accent matching")
