#!/usr/bin/env python3
"""
Debug script to investigate the ROOT CAUSE of "now" hallucination for short texts.
This analyzes the conditioning tokens and their relationship to early generation.
"""
import logging
import sys
import warnings
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Set root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Suppress noisy loggers
logging.getLogger('numba').setLevel(logging.ERROR)
logging.getLogger('torio').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

sys.path.insert(0, '/Users/a10n/Projects/nightingale_TTS/python/chatterbox/src')

import torch
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
BAKED_VOICES_DIR = PROJECT_ROOT / "baked_voices"


def analyze_conditioning():
    """Analyze the conditioning tokens to understand hallucination source."""
    voice = "samantha"
    voice_dir = BAKED_VOICES_DIR / voice

    print("=" * 70)
    print(f"ANALYZING CONDITIONING TOKENS FOR: {voice}")
    print("=" * 70)
    print()

    # Load the conditioning tokens
    t3_cond_tokens_path = voice_dir / "t3_cond_tokens.npy"
    prompt_token_path = voice_dir / "prompt_token.npy"

    if t3_cond_tokens_path.exists():
        t3_cond = np.load(t3_cond_tokens_path)
        print(f"t3_cond_tokens.npy shape: {t3_cond.shape}")
        print(f"t3_cond_tokens first 50: {t3_cond.flatten()[:50].tolist()}")
        print(f"t3_cond_tokens last 50: {t3_cond.flatten()[-50:].tolist()}")
        print()

    if prompt_token_path.exists():
        prompt = np.load(prompt_token_path)
        print(f"prompt_token.npy shape: {prompt.shape}")
        print(f"prompt_token first 50: {prompt.flatten()[:50].tolist()}")
        print(f"prompt_token last 50: {prompt.flatten()[-50:].tolist()}")
        print()


def analyze_generation():
    """Generate 'Hi.' and analyze the token patterns."""
    from chatterbox.tts import ChatterboxTTS

    print("=" * 70)
    print("GENERATING 'Hi.' TO ANALYZE TOKEN PATTERNS")
    print("=" * 70)
    print()

    voice = "samantha"
    text = "Hi."
    ref_audio_path = BAKED_VOICES_DIR / voice / "ref_audio.wav"

    # Load model
    print("Loading ChatterboxTTS model...")
    device = "mps"
    model = ChatterboxTTS.from_pretrained(device=device)
    print("Model loaded!")
    print()

    # Prepare conditioning (to inspect the tokens)
    print("Preparing conditioning...")
    model.prepare_conditionals(str(ref_audio_path), exaggeration=0.5)

    # Print conditioning token info
    cond_tokens = model.conds.t3.cond_prompt_speech_tokens
    if cond_tokens is not None:
        print(f"cond_prompt_speech_tokens shape: {cond_tokens.shape}")
        cond_flat = cond_tokens[0].tolist()
        print(f"Conditioning tokens (first 50): {cond_flat[:50]}")
        print(f"Conditioning tokens (last 50): {cond_flat[-50:]}")
        print()

        # Check for common patterns
        print("Looking for repeated patterns in conditioning...")
        from collections import Counter
        token_counts = Counter(cond_flat)
        print(f"Most common tokens: {token_counts.most_common(10)}")
        print()

    # Now generate and collect internal state
    print("Generating 'Hi.' with detailed logging...")
    print("-" * 70)

    wav = model.generate(
        text=text,
        audio_prompt_path=str(ref_audio_path),
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.001,  # Near-deterministic
    )

    print("-" * 70)
    print()

    duration_ms = (wav.shape[-1] / model.sr) * 1000
    print(f"Generated audio: {duration_ms:.0f}ms ({wav.shape[-1]} samples)")
    print()


def analyze_uncond_path():
    """
    Test what happens when we generate with NO text (just conditioning).
    This simulates the "uncond" path in CFG to see what the model wants to say.
    """
    from chatterbox.tts import ChatterboxTTS, punc_norm
    import torch.nn.functional as F

    print("=" * 70)
    print("ANALYZING UNCONDITIONED PATH (CONDITIONING ONLY)")
    print("=" * 70)
    print()

    voice = "samantha"
    ref_audio_path = BAKED_VOICES_DIR / voice / "ref_audio.wav"

    # Load model
    print("Loading ChatterboxTTS model...")
    device = "mps"
    model = ChatterboxTTS.from_pretrained(device=device)
    print("Model loaded!")
    print()

    # Prepare conditioning
    model.prepare_conditionals(str(ref_audio_path), exaggeration=0.5)

    # Create minimal text tokens (just SOT/EOT)
    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token

    # Test 1: Just SOT + EOT (empty text)
    print("Test 1: Minimal text (SOT + EOT only)")
    text_tokens = torch.tensor([[sot, eot]], device=device)
    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # For CFG

    print(f"text_tokens: {text_tokens[0].tolist()}")

    with torch.inference_mode():
        speech_tokens = model.t3.inference(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens,
            max_new_tokens=100,  # Short generation
            temperature=0.001,
            cfg_weight=0.5,
        )

    print(f"Generated {speech_tokens.shape[-1]} speech tokens")
    print(f"First 30 tokens: {speech_tokens[0].tolist()[:30]}")
    print()

    # Test 2: Compare "Hi." vs "Hello world."
    print("Test 2: Comparing short vs long text")

    for test_text in ["Hi.", "Hello world."]:
        norm_text = punc_norm(test_text)
        text_tokens = model.tokenizer.text_to_tokens(norm_text).to(device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        print(f"\n'{test_text}':")
        print(f"  Text tokens: {text_tokens[0].tolist()}")
        print(f"  Length: {text_tokens.shape[-1]}")


def check_input_sequence_length():
    """
    The key question: For short text, what's the ratio of
    conditioning tokens to text tokens in the input sequence?
    """
    print("=" * 70)
    print("INPUT SEQUENCE ANALYSIS")
    print("=" * 70)
    print()

    # Config values
    speech_cond_prompt_len = 150  # Raw conditioning tokens before Perceiver
    perceiver_query_tokens = 32   # After Perceiver compression
    speaker_emb_tokens = 1
    emotion_adv_tokens = 1

    total_cond = perceiver_query_tokens + speaker_emb_tokens + emotion_adv_tokens

    print(f"Conditioning tokens:")
    print(f"  - Perceiver output: {perceiver_query_tokens}")
    print(f"  - Speaker embedding: {speaker_emb_tokens}")
    print(f"  - Emotion embedding: {emotion_adv_tokens}")
    print(f"  = Total: {total_cond}")
    print()

    test_texts = [
        ("Hi.", 5),           # SOT + H + i + . + EOT
        ("Hello.", 7),        # SOT + H + e + l + l + o + . + EOT (approx)
        ("Hello world.", 12), # Longer
    ]

    print("Sequence composition:")
    for text, text_len in test_texts:
        total = total_cond + text_len + 1  # +1 for BOS speech token
        cond_ratio = total_cond / total * 100
        text_ratio = text_len / total * 100
        print(f"  '{text}' ({text_len} tokens):")
        print(f"    Conditioning: {total_cond} ({cond_ratio:.1f}%)")
        print(f"    Text: {text_len} ({text_ratio:.1f}%)")
        print(f"    Total input: {total}")
    print()

    print("HYPOTHESIS:")
    print("For 'Hi.', conditioning is 87% of the input sequence.")
    print("The model's attention may be dominated by conditioning tokens,")
    print("causing it to 'continue' the reference audio pattern instead of")
    print("following the short text prompt.")


def compare_generated_to_conditioning():
    """
    Compare the tokens generated for 'Hi.' to the conditioning tokens
    to see if the model is copying from conditioning.
    """
    from chatterbox.tts import ChatterboxTTS, punc_norm
    import torch.nn.functional as F

    print("=" * 70)
    print("COMPARING GENERATED TOKENS TO CONDITIONING")
    print("=" * 70)
    print()

    voice = "samantha"
    ref_audio_path = BAKED_VOICES_DIR / voice / "ref_audio.wav"

    # Load conditioning tokens from file
    t3_cond_tokens_path = BAKED_VOICES_DIR / voice / "t3_cond_tokens.npy"
    cond_tokens_saved = np.load(t3_cond_tokens_path).flatten().tolist()

    print(f"Conditioning tokens (from file):")
    print(f"  First 30: {cond_tokens_saved[:30]}")
    print()

    # Load model
    print("Loading ChatterboxTTS model...")
    device = "mps"
    model = ChatterboxTTS.from_pretrained(device=device)
    print("Model loaded!")
    print()

    # Prepare conditioning
    model.prepare_conditionals(str(ref_audio_path), exaggeration=0.5)

    # Get live conditioning tokens
    live_cond = model.conds.t3.cond_prompt_speech_tokens[0].tolist()
    print(f"Live conditioning tokens:")
    print(f"  First 30: {live_cond[:30]}")
    print()

    # Generate with minimal output but capture tokens
    text = "Hi."
    norm_text = punc_norm(text)
    text_tokens = model.tokenizer.text_to_tokens(norm_text).to(device)

    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    text_tokens = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)
    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # For CFG

    print(f"Text tokens: {text_tokens[0].tolist()}")
    print()

    print("Generating (checking for token copying)...")
    print("-" * 70)

    with torch.inference_mode():
        speech_tokens = model.t3.inference(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens,
            max_new_tokens=100,
            temperature=0.001,
            cfg_weight=0.5,
        )

    print("-" * 70)
    print()

    # Analyze results
    gen_tokens = speech_tokens[0].tolist()
    print(f"Generated tokens: {gen_tokens[:50]}")
    print()

    # Check for matches with conditioning
    print("Checking for conditioning token matches:")
    cond_set = set(cond_tokens_saved[:30])  # First 30 conditioning tokens
    matches = []
    for i, tok in enumerate(gen_tokens[:30]):
        if tok in cond_set:
            # Find where in conditioning
            try:
                cond_idx = cond_tokens_saved.index(tok)
                matches.append((i, tok, cond_idx))
            except ValueError:
                pass

    for gen_idx, tok, cond_idx in matches:
        print(f"  Gen[{gen_idx}] = {tok} matches Cond[{cond_idx}]")

    if matches:
        print()
        print("CONCLUSION: Model IS copying tokens from conditioning!")
    else:
        print()
        print("No direct token matches found.")


if __name__ == "__main__":
    analyze_conditioning()
    check_input_sequence_length()
    compare_generated_to_conditioning()
    # analyze_uncond_path()  # Uncomment to test uncond path
