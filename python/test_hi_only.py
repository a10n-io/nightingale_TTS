#!/usr/bin/env python3
"""
Focus test: Just "Hi." to understand the hallucination issue.
Tests with both baked voices to compare behavior.
"""
import logging
import sys
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logging.getLogger('numba').setLevel(logging.ERROR)
logging.getLogger('torio').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

sys.path.insert(0, '/Users/a10n/Projects/nightingale_TTS/python/chatterbox/src')

import torch
import torchaudio
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "hi_focus"
BAKED_VOICES_DIR = PROJECT_ROOT / "baked_voices"


def main():
    from chatterbox.tts import ChatterboxTTS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FOCUS TEST: 'Hi.' with BOTH baked voices")
    print("=" * 70)
    print()

    print("Loading model...")
    device = "mps"
    model = ChatterboxTTS.from_pretrained(device=device)
    print("Model loaded!")
    print()

    voices = ["samantha", "sujano"]
    text = "Hi."

    for voice in voices:
        print("-" * 50)
        print(f"Voice: {voice}")
        print("-" * 50)

        ref_audio_path = BAKED_VOICES_DIR / voice / "ref_audio.wav"

        print(f"Generating: '{text}'")

        wav = model.generate(
            text=text,
            audio_prompt_path=str(ref_audio_path),
            exaggeration=0.5,
            cfg_weight=0.5,
            temperature=0.001,
        )

        output_path = OUTPUT_DIR / f"hi_{voice}.wav"
        wav_out = wav.cpu()
        if wav_out.dim() == 1:
            wav_out = wav_out.unsqueeze(0)
        elif wav_out.dim() == 3:
            wav_out = wav_out.squeeze(0)
        torchaudio.save(str(output_path), wav_out, model.sr)

        duration_ms = (wav.shape[-1] / model.sr) * 1000
        print(f"Duration: {duration_ms:.0f}ms")
        print(f"Saved: {output_path}")
        print()

    print("=" * 70)
    print("COMPARE THE TWO FILES:")
    print(f"  - {OUTPUT_DIR}/hi_samantha.wav")
    print(f"  - {OUTPUT_DIR}/hi_sujano.wav")
    print("=" * 70)


if __name__ == "__main__":
    main()
