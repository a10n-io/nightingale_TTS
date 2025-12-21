#!/usr/bin/env python3
"""Debug script to analyze problematic short phrase samples."""

import os
import sys
import json
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

# Add paths
sys.path.insert(0, '/Users/a10n/Projects/nightingale_TTS/python/chatterbox/src')

import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS

# Problematic samples to debug
CUT_OFF_SAMPLES = [
    {"id": 7, "text": "One, two, three.", "voice": "samantha"},
    {"id": 16, "text": "12:30 PM.", "voice": "samantha"},
    {"id": 16, "text": "12:30 PM.", "voice": "sujano"},
]

TRAILING_SOUND_SAMPLES = [
    {"id": 6, "text": "Hmm...", "voice": "sujano"},
    {"id": 13, "text": "Good morning.", "voice": "sujano"},
    {"id": 25, "text": "The end.", "voice": "sujano"},
]

VOICE_REFS = {
    "samantha": "/Users/a10n/Projects/nightingale_TTS/test_audio/reference_voices/samantha.wav",
    "sujano": "/Users/a10n/Projects/nightingale_TTS/test_audio/reference_voices/sujano.wav",
}

OUTPUT_DIR = "/Users/a10n/Projects/nightingale_TTS/test_audio/debug_samples"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print("Loading ChatterboxTTS model...")
    device = "mps"
    model = ChatterboxTTS.from_pretrained(device=device)

    print("\n" + "="*60)
    print("ANALYZING CUT-OFF SAMPLES")
    print("="*60)

    for sample in CUT_OFF_SAMPLES:
        text = sample["text"]
        voice = sample["voice"]
        sample_id = sample["id"]

        print(f"\n--- {voice} #{sample_id}: '{text}' ---")

        # Get reference audio
        ref_path = VOICE_REFS[voice]
        ref_audio = torchaudio.load(ref_path)

        # Generate
        wav = model.generate(
            text=text,
            audio_prompt=ref_audio,
            exaggeration=0.5,
            cfg_weight=0.5,
            temperature=0.0,  # Deterministic
        )

        # Save
        filename = f"cutoff_{voice}_{sample_id}_{text.replace(' ', '_').replace(',', '').replace(':', '').replace('.', '')}.wav"
        output_path = os.path.join(OUTPUT_DIR, filename)
        torchaudio.save(output_path, wav.unsqueeze(0).cpu(), model.sr)

        duration_ms = (wav.shape[0] / model.sr) * 1000
        print(f"  Generated {wav.shape[0]} samples ({duration_ms:.0f}ms)")
        print(f"  Saved to: {output_path}")

    print("\n" + "="*60)
    print("ANALYZING TRAILING SOUND SAMPLES")
    print("="*60)

    for sample in TRAILING_SOUND_SAMPLES:
        text = sample["text"]
        voice = sample["voice"]
        sample_id = sample["id"]

        print(f"\n--- {voice} #{sample_id}: '{text}' ---")

        # Get reference audio
        ref_path = VOICE_REFS[voice]
        ref_audio = torchaudio.load(ref_path)

        # Generate
        wav = model.generate(
            text=text,
            audio_prompt=ref_audio,
            exaggeration=0.5,
            cfg_weight=0.5,
            temperature=0.0,  # Deterministic
        )

        # Save
        filename = f"trailing_{voice}_{sample_id}_{text.replace(' ', '_').replace(',', '').replace(':', '').replace('.', '').replace('...', '')}.wav"
        output_path = os.path.join(OUTPUT_DIR, filename)
        torchaudio.save(output_path, wav.unsqueeze(0).cpu(), model.sr)

        duration_ms = (wav.shape[0] / model.sr) * 1000
        print(f"  Generated {wav.shape[0]} samples ({duration_ms:.0f}ms)")
        print(f"  Saved to: {output_path}")

if __name__ == "__main__":
    main()
