#!/usr/bin/env python3
"""
Diagnose the "Hi." hallucination by examining:
1. What tokens are actually generated
2. What the attention is looking at
3. Whether tokens match conditioning patterns
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
import torch.nn.functional as F
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
BAKED_VOICES_DIR = PROJECT_ROOT / "baked_voices"


def diagnose():
    from chatterbox.tts import ChatterboxTTS, punc_norm

    print("=" * 70)
    print("DIAGNOSING 'Hi.' HALLUCINATION")
    print("=" * 70)
    print()

    device = "mps"
    model = ChatterboxTTS.from_pretrained(device=device)

    voice = "samantha"
    ref_audio_path = BAKED_VOICES_DIR / voice / "ref_audio.wav"
    model.prepare_conditionals(str(ref_audio_path), exaggeration=0.5)

    # Get conditioning tokens
    cond_tokens = model.conds.t3.cond_prompt_speech_tokens[0].cpu().numpy()
    print(f"Conditioning tokens ({len(cond_tokens)} tokens):")
    print(f"  First 20: {cond_tokens[:20].tolist()}")
    print(f"  Last 20: {cond_tokens[-20:].tolist()}")
    print()

    # Prepare text
    text = "Hi."
    norm_text = punc_norm(text)
    text_tokens = model.tokenizer.text_to_tokens(norm_text).to(device)

    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    text_tokens_padded = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens_padded = F.pad(text_tokens_padded, (0, 1), value=eot)

    print(f"Text: '{text}' -> '{norm_text}'")
    print(f"Text tokens: {text_tokens_padded[0].tolist()}")
    print(f"  SOT={sot}, EOT={eot}")
    print()

    # Calculate sequence lengths
    cond_len = len(cond_tokens)
    text_len = text_tokens_padded.shape[-1]
    total_input = cond_len + text_len + 1  # +1 for BOS speech token

    print(f"Input sequence breakdown:")
    print(f"  Conditioning: {cond_len} tokens ({100*cond_len/total_input:.1f}%)")
    print(f"  Text: {text_len} tokens ({100*text_len/total_input:.1f}%)")
    print(f"  BOS: 1 token")
    print(f"  Total: {total_input} tokens")
    print()

    # Now generate with token logging using direct inference call
    print("=" * 70)
    print("GENERATING WITH TOKEN LOGGING")
    print("=" * 70)

    # Use model's inference but with a wrapper to capture tokens
    text_tokens_cfg = torch.cat([text_tokens_padded, text_tokens_padded], dim=0)

    # Patch the model to log generated tokens
    original_inference = model.t3.inference
    generated_tokens = []

    with torch.inference_mode():
        speech_tokens = model.t3.inference(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens_cfg,
            max_new_tokens=100,
            temperature=0.001,
            cfg_weight=0.5,
        )

    generated_tokens = speech_tokens[0].cpu().numpy().tolist()

    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Check how many generated tokens match conditioning
    cond_set = set(cond_tokens.tolist())
    matches = sum(1 for t in generated_tokens if t in cond_set)

    print(f"Generated {len(generated_tokens)} tokens")
    print(f"Tokens matching conditioning: {matches} ({100*matches/len(generated_tokens):.1f}%)")
    print()

    # Show token sequence
    print(f"First 20 generated tokens: {generated_tokens[:20]}")
    print()

    # The key insight: what tokens are generated in frames 0-15 (before actual speech)?
    print("CRITICAL: First 15 tokens (likely the 'now' hallucination):")
    for i, tok in enumerate(generated_tokens[:15]):
        is_cond = "← matches conditioning" if tok in cond_set else ""
        print(f"  Frame {i}: token {tok} {is_cond}")


if __name__ == "__main__":
    diagnose()
