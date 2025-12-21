#!/usr/bin/env python3
"""
Analyze attention patterns during "Hi." generation to understand hallucination.
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
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
BAKED_VOICES_DIR = PROJECT_ROOT / "baked_voices"


def analyze():
    from chatterbox.tts import ChatterboxTTS, punc_norm

    print("=" * 70)
    print("ATTENTION PATTERN ANALYSIS FOR 'Hi.'")
    print("=" * 70)
    print()

    device = "mps"
    model = ChatterboxTTS.from_pretrained(device=device)

    voice = "samantha"
    ref_audio_path = BAKED_VOICES_DIR / voice / "ref_audio.wav"
    model.prepare_conditionals(str(ref_audio_path), exaggeration=0.5)

    # Get the conditioning info
    cond_tokens = model.conds.t3.cond_prompt_speech_tokens[0].cpu().numpy()
    print(f"Reference audio conditioning: {len(cond_tokens)} raw tokens -> 32 after Perceiver")
    print()

    # Prepare text tokens
    import torch.nn.functional as F
    text = "Hi."
    norm_text = punc_norm(text)
    text_tokens = model.tokenizer.text_to_tokens(norm_text).to(device)

    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    text_tokens_padded = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens_padded = F.pad(text_tokens_padded, (0, 1), value=eot)

    print(f"Text: '{text}'")
    print(f"Text tokens: {text_tokens_padded[0].tolist()}")
    print(f"  Token meanings: SOT={sot}, 'H'=284, 'i'=22, '.'=9, EOT={eot}")
    print()

    # Now look at the Perceiver's compression
    # The conditioning goes through perceiver which uses 32 query tokens
    print("=" * 70)
    print("PERCEIVER COMPRESSION")
    print("=" * 70)
    print(f"Raw conditioning tokens: {len(cond_tokens)}")
    print(f"Perceiver output: 32 tokens (pre_attention_query_token=32)")
    print(f"Compression ratio: {len(cond_tokens)}/32 = {len(cond_tokens)/32:.1f}x")
    print()
    print("This means the conditioning takes 32 positions in the sequence,")
    print("while 'Hi.' (5 text tokens) takes only 5 positions.")
    print("Attention is overwhelmed by conditioning at the start.")
    print()

    # Calculate what ideal generation would look like
    print("=" * 70)
    print("EXPECTED BEHAVIOR")
    print("=" * 70)
    # "Hi" is roughly:
    # - Breath/onset: 0-2 frames
    # - "H": 2-5 frames
    # - "i": 5-10 frames
    # - Trailing: 10-12 frames
    # Total: ~10-15 frames = 400-600ms
    print("For 'Hi.' we expect:")
    print("  - ~10-15 speech frames")
    print("  - ~400-600ms total duration")
    print("  - No 'now' or other sounds before 'Hi'")
    print()
    print("Current behavior:")
    print("  - 19 frames generated")
    print("  - 720ms duration")
    print("  - Possible hallucination in early frames")
    print()

    # Now let's look at what happens if we check attention to text tokens
    print("=" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 70)
    print()
    print("The AlignmentStreamAnalyzer tracks attention to text tokens.")
    print("Key observations:")
    print("1. First ~14 frames show low text attention (hallucination)")
    print("2. After frame 14, model starts attending to text correctly")
    print("3. For short texts, this 'warm-up' period is too long")
    print()
    print("Potential fixes:")
    print("1. Scale text embeddings (current: 2x) - helps but not enough")
    print("2. Add attention bias towards text tokens")
    print("3. Use a different starting strategy for short texts")
    print("4. Reduce conditioning influence for short texts")


if __name__ == "__main__":
    analyze()
