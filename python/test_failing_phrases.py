#!/usr/bin/env python3
"""
Quick test of the specific failing phrases.
"""
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/a10n/Projects/nightingale_TTS/python/chatterbox/src')

import torch
import torchaudio
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "fix_test"
BAKED_VOICES_DIR = PROJECT_ROOT / "baked_voices"

# The specific failing phrases
FAILING_PHRASES = [
    ("Testing.", "samantha"),   # was "community as a break it"
    ("What?", "sujano"),        # was "yeah"
    ("I see.", "samantha"),     # was "does I See"
    ("Hello world.", "samantha"), # was "the world"
    ("Hmm...", "samantha"),     # was "heu"
    ("Wait!", "sujano"),        # was "heygothere"
]


def main():
    from chatterbox.tts import ChatterboxTTS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = ChatterboxTTS.from_pretrained(device="mps")
    print("Model loaded!")
    print()

    for text, voice in FAILING_PHRASES:
        print(f"Testing: '{text}' with {voice}")

        ref_audio_path = BAKED_VOICES_DIR / voice / "ref_audio.wav"

        wav = model.generate(
            text=text,
            audio_prompt_path=str(ref_audio_path),
            exaggeration=0.5,
            cfg_weight=0.5,
            temperature=0.001,
        )

        duration_ms = (wav.shape[-1] / model.sr) * 1000
        print(f"  Duration: {duration_ms:.0f}ms")

        # Save
        safe_text = text.replace(" ", "_").replace(".", "").replace("?", "").replace("!", "")
        filename = f"{voice}_{safe_text}.wav"
        output_path = OUTPUT_DIR / filename

        wav_out = wav.cpu()
        if wav_out.dim() == 1:
            wav_out = wav_out.unsqueeze(0)
        elif wav_out.dim() == 3:
            wav_out = wav_out.squeeze(0)
        torchaudio.save(str(output_path), wav_out, model.sr)
        print(f"  Saved: {output_path}")
        print()

    print("=" * 60)
    print(f"Test files saved to: {OUTPUT_DIR}")
    print("Please listen to verify the fixes work.")


if __name__ == "__main__":
    main()
