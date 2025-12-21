#!/usr/bin/env python3
"""
Test text padding/prefixing as a fix for short text hallucination.

ROOT CAUSE SUMMARY:
- For short texts like "Hi.", conditioning tokens dominate (85% of input)
- Model's attention jumps from SOT directly to punctuation, skipping content tokens
- Early frames generate random/filler tokens ("now" hallucination)
- Later frames copy conditioning patterns

PROPOSED FIX:
- Prepend filler text (e.g., "Say: ") to give model "ramp up" time
- The filler gives attention a path to sweep through before reaching content
- Cut the filler audio in post-processing

This tests whether the approach improves attention progression.
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
logging.getLogger('LlamaModel').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

sys.path.insert(0, '/Users/a10n/Projects/nightingale_TTS/python/chatterbox/src')

import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "padding_fix"
BAKED_VOICES_DIR = PROJECT_ROOT / "baked_voices"


def test_text_padding():
    from chatterbox.tts import ChatterboxTTS, punc_norm

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ChatterboxTTS model...")
    device = "mps"
    model = ChatterboxTTS.from_pretrained(device=device)
    print("Model loaded!")
    print()

    voice = "samantha"
    ref_audio_path = BAKED_VOICES_DIR / voice / "ref_audio.wav"

    # Prepare conditioning once
    model.prepare_conditionals(str(ref_audio_path), exaggeration=0.5)

    # Test cases: original vs padded
    test_cases = [
        # (name, text, description)
        ("original_hi", "Hi.", "Original 'Hi.' - expect hallucination"),
        ("padded_hi", "Say, hi.", "Padded 'Say, hi.' - should help attention"),
        ("longer_prefix", "Okay then, hi.", "Longer prefix - more ramp up"),
        ("repeat_hi", "Hi, hi.", "Repeated 'Hi, hi.' - doubling content"),
        ("original_hello", "Hello.", "Original 'Hello.' - short but longer"),
        ("padded_hello", "Well then, hello.", "Padded 'Hello.' for comparison"),
    ]

    for name, text, description in test_cases:
        print("=" * 70)
        print(f"TEST: {name}")
        print(f"TEXT: '{text}'")
        print(f"DESC: {description}")
        print("=" * 70)

        # Normalize and tokenize to see token count
        norm_text = punc_norm(text)
        text_tokens = model.tokenizer.text_to_tokens(norm_text).to(device)
        sot = model.t3.hp.start_text_token
        eot = model.t3.hp.stop_text_token
        text_tokens_full = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens_full = F.pad(text_tokens_full, (0, 1), value=eot)

        print(f"Normalized: '{norm_text}'")
        print(f"Token count: {text_tokens_full.shape[-1]}")
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
        output_path = OUTPUT_DIR / f"{name}.wav"
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
    print("LISTEN TO THE OUTPUTS:")
    print(f"  cd {OUTPUT_DIR} && ls -la")
    print()
    print("Compare 'original_hi.wav' vs 'padded_hi.wav'")
    print("If padding helps, the padded version should have cleaner onset.")
    print("=" * 70)


def test_text_repetition():
    """
    Another approach: repeat the text content to increase its presence
    in the attention window.
    """
    from chatterbox.tts import ChatterboxTTS, punc_norm

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TESTING TEXT REPETITION APPROACH")
    print("=" * 70)
    print()

    print("Loading ChatterboxTTS model...")
    device = "mps"
    model = ChatterboxTTS.from_pretrained(device=device)
    print("Model loaded!")
    print()

    voice = "samantha"
    ref_audio_path = BAKED_VOICES_DIR / voice / "ref_audio.wav"
    model.prepare_conditionals(str(ref_audio_path), exaggeration=0.5)

    # Original
    text = "Hi."
    print(f"Generating original: '{text}'")
    wav1 = model.generate(
        text=text,
        audio_prompt_path=str(ref_audio_path),
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.001,
    )

    # Repeated internally (as prefix)
    text2 = "Hi, hi, hi."
    print(f"Generating triple repeat: '{text2}'")
    wav2 = model.generate(
        text=text2,
        audio_prompt_path=str(ref_audio_path),
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.001,
    )

    # Save
    for name, wav, txt in [("repeat_single", wav1, text), ("repeat_triple", wav2, text2)]:
        output_path = OUTPUT_DIR / f"{name}.wav"
        wav_out = wav.cpu()
        if wav_out.dim() == 1:
            wav_out = wav_out.unsqueeze(0)
        elif wav_out.dim() == 3:
            wav_out = wav_out.squeeze(0)
        torchaudio.save(str(output_path), wav_out, model.sr)
        duration_ms = (wav.shape[-1] / model.sr) * 1000
        print(f"  '{txt}' -> {duration_ms:.0f}ms -> {output_path.name}")


if __name__ == "__main__":
    test_text_padding()
    # test_text_repetition()  # Uncomment to test
