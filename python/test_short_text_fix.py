#!/usr/bin/env python3
"""
Test the short text hallucination fix.

This tests that short texts like "Hi." no longer produce "now" or other
hallucination sounds at the start.
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
logging.getLogger('urllib3').setLevel(logging.ERROR)

sys.path.insert(0, '/Users/a10n/Projects/nightingale_TTS/python/chatterbox/src')

import torch
import torchaudio
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "short_text_fix"
BAKED_VOICES_DIR = PROJECT_ROOT / "baked_voices"


def main():
    from chatterbox.tts import ChatterboxTTS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ChatterboxTTS model...")
    device = "mps"
    model = ChatterboxTTS.from_pretrained(device=device)
    print("Model loaded!")
    print()

    voice = "samantha"
    ref_audio_path = BAKED_VOICES_DIR / voice / "ref_audio.wav"

    # Test cases: short texts that previously caused hallucination
    test_cases = [
        "Hi.",
        "No.",
        "Yes.",
        "Bye.",
        "OK.",
        "What?",
        "Hey!",
        "Stop.",
        "Go.",
        "Help!",
    ]

    print("=" * 70)
    print("TESTING SHORT TEXT FIX")
    print("=" * 70)
    print()

    for text in test_cases:
        print(f"Generating: '{text}'")

        wav = model.generate(
            text=text,
            audio_prompt_path=str(ref_audio_path),
            exaggeration=0.5,
            cfg_weight=0.5,
            temperature=0.001,
        )

        # Save
        safe_name = text.replace(" ", "_").replace(".", "").replace("?", "Q").replace("!", "E")[:20]
        output_path = OUTPUT_DIR / f"short_{safe_name}.wav"

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
    print("DONE - Check outputs in:", OUTPUT_DIR)
    print()
    print("Listen to the files and verify:")
    print("1. No 'now', 'ah', or other sounds before the word")
    print("2. The word is clear and properly pronounced")
    print("3. Duration is reasonable (200-600ms for short words)")
    print("=" * 70)


if __name__ == "__main__":
    main()
