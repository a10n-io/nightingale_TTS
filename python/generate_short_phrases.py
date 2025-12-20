#!/usr/bin/env python3
"""
Generate short phrase test audio files.
Uses the local chatterbox with hallucination fixes.
"""
import json
import os
import sys
import re

# Add paths
sys.path.insert(0, '/Users/a10n/Projects/nightingale_TTS/python/chatterbox/src')

import torch
import torchaudio
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
SHORT_PHRASES_PATH = PROJECT_ROOT / "E2E" / "short_phrases.json"
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "short_phrases"
BAKED_VOICES_DIR = PROJECT_ROOT / "baked_voices"


def sanitize_filename(text: str) -> str:
    """Convert text to a safe filename."""
    # Replace spaces with underscores
    s = text.replace(" ", "_")
    # Remove problematic characters
    s = re.sub(r'[^\w\-_.]', '', s)
    return s


def main():
    from chatterbox.tts import ChatterboxTTS

    # Load test phrases
    with open(SHORT_PHRASES_PATH) as f:
        phrases = json.load(f)

    print(f"Loaded {len(phrases)} test phrases")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading ChatterboxTTS model...")
    device = "mps"
    model = ChatterboxTTS.from_pretrained(device=device)
    print("Model loaded!")
    print()

    voices = ["samantha", "sujano"]
    total = len(phrases) * len(voices)
    count = 0

    for voice in voices:
        print("=" * 60)
        print(f"VOICE: {voice}")
        print("=" * 60)

        # Get reference audio path
        ref_audio_path = BAKED_VOICES_DIR / voice / "ref_audio.wav"

        for phrase in phrases:
            count += 1
            text = phrase["text"]
            phrase_id = phrase["id"]

            print(f"[{count}/{total}] {voice} #{phrase_id}: \"{text}\"")

            # Generate
            wav = model.generate(
                text=text,
                audio_prompt_path=str(ref_audio_path),
                exaggeration=0.5,
                cfg_weight=0.5,
                temperature=0.001,  # Near-deterministic (0.0 causes div by zero)
            )

            # Save
            safe_text = sanitize_filename(text)
            filename = f"{voice}_{phrase_id:02d}_{safe_text}.wav"
            output_path = OUTPUT_DIR / filename

            # Ensure wav is 2D (channels, samples)
            wav_out = wav.cpu()
            if wav_out.dim() == 1:
                wav_out = wav_out.unsqueeze(0)
            elif wav_out.dim() == 3:
                wav_out = wav_out.squeeze(0)
            torchaudio.save(str(output_path), wav_out, model.sr)

            duration_ms = (wav.shape[0] / model.sr) * 1000
            print(f"  -> {wav.shape[0]} samples ({duration_ms:.0f}ms)")
            print()

    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Generated {count} audio files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
