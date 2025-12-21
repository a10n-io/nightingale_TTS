#!/usr/bin/env python3
"""
Debug script to understand severe hallucination in failing phrases.
Tests with different parameters to find what helps.
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
logging.getLogger('chatterbox').setLevel(logging.DEBUG)

sys.path.insert(0, '/Users/a10n/Projects/nightingale_TTS/python/chatterbox/src')

import torch
import torchaudio
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "debug"
BAKED_VOICES_DIR = PROJECT_ROOT / "baked_voices"


def test_phrase(model, text, voice, cfg_weight, temperature, output_prefix):
    """Test a phrase with specific parameters."""
    ref_audio_path = BAKED_VOICES_DIR / voice / "ref_audio.wav"

    print(f"\n  Testing: text='{text}', voice={voice}")
    print(f"  Params: cfg={cfg_weight}, temp={temperature}")

    wav = model.generate(
        text=text,
        audio_prompt_path=str(ref_audio_path),
        exaggeration=0.5,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )

    duration_ms = (wav.shape[-1] / model.sr) * 1000
    print(f"  Duration: {duration_ms:.0f}ms")

    # Save
    output_path = OUTPUT_DIR / f"{output_prefix}_{voice}_{text.replace(' ', '_').replace('.', '').replace('?', '').replace('!', '')}.wav"
    wav_out = wav.cpu()
    if wav_out.dim() == 1:
        wav_out = wav_out.unsqueeze(0)
    elif wav_out.dim() == 3:
        wav_out = wav_out.squeeze(0)
    torchaudio.save(str(output_path), wav_out, model.sr)
    print(f"  Saved: {output_path}")

    return output_path


def main():
    from chatterbox.tts import ChatterboxTTS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DEBUGGING SEVERE HALLUCINATIONS")
    print("=" * 70)

    # Failing phrases from user report
    failing_phrases = [
        ("Testing.", "samantha"),  # "community as a break it"
        ("What?", "sujano"),       # "yeah"
        ("I see.", "samantha"),    # "does I See"
        ("Hello world.", "samantha"),  # "the world"
    ]

    print("\nLoading model...")
    device = "mps"
    model = ChatterboxTTS.from_pretrained(device=device)
    print("Model loaded!")

    # Test configurations to try
    configs = [
        ("baseline", {"cfg_weight": 0.5, "temperature": 0.001}),
        ("high_cfg", {"cfg_weight": 2.0, "temperature": 0.001}),
        ("very_high_cfg", {"cfg_weight": 5.0, "temperature": 0.001}),
        ("normal_temp", {"cfg_weight": 0.5, "temperature": 0.8}),
        ("high_cfg_normal_temp", {"cfg_weight": 2.0, "temperature": 0.8}),
    ]

    for text, voice in failing_phrases:
        print("\n" + "=" * 60)
        print(f"PHRASE: '{text}' with {voice}")
        print("=" * 60)

        for config_name, params in configs:
            test_phrase(model, text, voice,
                       params["cfg_weight"],
                       params["temperature"],
                       config_name)

    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)
    print(f"\nCheck audio files in: {OUTPUT_DIR}")
    print("\nCompare the different parameter configurations to see which helps.")


if __name__ == "__main__":
    main()
