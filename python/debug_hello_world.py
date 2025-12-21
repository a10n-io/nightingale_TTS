#!/usr/bin/env python3
"""
Debug script for "Hello world." generation with samantha voice.
Tests the Attention Slicing approach.
"""
import logging
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Suppress noisy loggers
logging.getLogger('numba').setLevel(logging.ERROR)
logging.getLogger('torio').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

sys.path.insert(0, '/Users/a10n/Projects/nightingale_TTS/python/chatterbox/src')

import torch
import torchaudio
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "debug"
BAKED_VOICES_DIR = PROJECT_ROOT / "baked_voices"


def main():
    from chatterbox.tts import ChatterboxTTS

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading ChatterboxTTS model...")
    device = "mps"
    model = ChatterboxTTS.from_pretrained(device=device)
    print("Model loaded!")
    print()

    # Test case
    voice = "samantha"
    text = "Hello world."
    ref_audio_path = BAKED_VOICES_DIR / voice / "ref_audio.wav"

    print("=" * 70)
    print(f"TESTING ATTENTION SLICING: '{text}' with voice '{voice}'")
    print("=" * 70)
    print()

    # Generate
    print("Generating...")
    wav = model.generate(
        text=text,
        audio_prompt_path=str(ref_audio_path),
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.001,
    )

    # Save
    output_path = OUTPUT_DIR / "hello_world_attention_slice.wav"
    wav_out = wav.cpu()
    if wav_out.dim() == 1:
        wav_out = wav_out.unsqueeze(0)
    elif wav_out.dim() == 3:
        wav_out = wav_out.squeeze(0)
    torchaudio.save(str(output_path), wav_out, model.sr)

    duration_ms = (wav.shape[-1] / model.sr) * 1000
    print()
    print(f"Saved: {output_path}")
    print(f"Duration: {duration_ms:.0f}ms ({wav.shape[-1]} samples)")
    print()

    # Test a few more phrases
    test_phrases = [
        "Hi.",
        "One two three.",
        "Thank you very much for your help today.",
    ]

    for phrase in test_phrases:
        print("-" * 70)
        print(f"Testing: '{phrase}'")

        wav = model.generate(
            text=phrase,
            audio_prompt_path=str(ref_audio_path),
            exaggeration=0.5,
            cfg_weight=0.5,
            temperature=0.001,
        )

        safe_name = phrase.replace(" ", "_").replace(".", "").replace(",", "")[:30]
        output_path = OUTPUT_DIR / f"test_{safe_name}.wav"

        wav_out = wav.cpu()
        if wav_out.dim() == 1:
            wav_out = wav_out.unsqueeze(0)
        elif wav_out.dim() == 3:
            wav_out = wav_out.squeeze(0)
        torchaudio.save(str(output_path), wav_out, model.sr)

        duration_ms = (wav.shape[-1] / model.sr) * 1000
        print(f"  -> {duration_ms:.0f}ms saved to {output_path.name}")

    print()
    print("=" * 70)
    print("DONE - Check test_audio/debug/ for results")
    print("=" * 70)


if __name__ == "__main__":
    main()
