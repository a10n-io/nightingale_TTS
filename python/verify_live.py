#!/usr/bin/env python3
"""
Live verification script for Python/PyTorch vs Swift/MLX comparison.

Usage:
    python verify_live.py --text "Hello world" --voice baked_voice
    python verify_live.py --text "Testing different text" --voice baked_voice

This generates intermediate outputs that Swift can compare against.
Change --text or --voice and re-run both Python and Swift to verify they stay in sync.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import json

# Project paths
PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
MODEL_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICE_DIR = PROJECT_ROOT / "baked_voices"
OUTPUT_DIR = PROJECT_ROOT / "verification_outputs" / "live"


def load_model(device="cpu"):
    """Load the Chatterbox model."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    return ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device=device)


def verify_t3_pipeline(text: str, voice_name: str, device: str = "cpu"):
    """Run T3 pipeline and save intermediate outputs for Swift comparison."""
    from chatterbox.mtl_tts import Conditionals

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PYTHON VERIFICATION - T3 PIPELINE")
    print("=" * 80)
    print(f"Text: \"{text}\"")
    print(f"Voice: {voice_name}")
    print(f"Device: {device}")
    print()

    # Load model
    print("Loading model...")
    model = load_model(device)

    # Load voice (try both folder structure and flat structure)
    voice_path = VOICE_DIR / voice_name / "baked_voice.pt"
    if not voice_path.exists():
        voice_path = VOICE_DIR / f"{voice_name}.pt"
    print(f"Loading voice from: {voice_path}")
    model.conds = Conditionals.load(str(voice_path), map_location=device)

    # Access T3 model internals
    t3 = model.t3

    # =========================================================================
    # STEP 1: TEXT TOKENIZATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: TEXT TOKENIZATION")
    print("=" * 80)

    # Tokenize text using MTLTokenizer
    text_tokens = model.tokenizer.encode(text)
    print(f"Text tokens: {text_tokens}")
    print(f"Token count: {len(text_tokens)}")

    # Save as numpy array
    text_tokens_np = np.array(text_tokens, dtype=np.int32)
    np.save(OUTPUT_DIR / "step1_text_tokens.npy", text_tokens_np)

    # =========================================================================
    # STEP 2: T3 CONDITIONING
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: T3 CONDITIONING")
    print("=" * 80)

    # Get conditioning from baked voice
    conds = model.conds.t3

    # Speaker embedding projection
    speaker_emb = conds.speaker_emb.to(device)  # [1, 256]
    speaker_token = t3.cond_enc.spkr_enc(speaker_emb).unsqueeze(1)  # [1, 1, 1024]
    print(f"speaker_token: {list(speaker_token.shape)}")

    # Perceiver resampler on conditioning speech tokens
    cond_speech_tokens = conds.cond_prompt_speech_tokens.to(device)  # [1, N]
    speech_emb = t3.speech_emb(cond_speech_tokens)
    positions = torch.arange(cond_speech_tokens.shape[1], device=device).unsqueeze(0)
    speech_pos_emb = t3.speech_pos_emb(positions)
    cond_speech_emb = speech_emb + speech_pos_emb
    perceiver_out = t3.cond_enc.perceiver(cond_speech_emb)  # [1, 32, 1024]
    print(f"perceiver_out: {list(perceiver_out.shape)}")

    # Emotion adversarial FC
    emotion_value = conds.emotion_adv.to(device)  # [1, 1, 1]
    emotion_token = t3.cond_enc.emotion_adv_fc(emotion_value)  # [1, 1, 1024]
    print(f"emotion_token: {list(emotion_token.shape)}")

    # Final conditioning: concat [speaker, perceiver, emotion]
    final_cond = torch.cat([speaker_token, perceiver_out, emotion_token], dim=1)
    print(f"final_cond: {list(final_cond.shape)}")

    # Save
    np.save(OUTPUT_DIR / "step2_speaker_token.npy", speaker_token.detach().cpu().numpy())
    np.save(OUTPUT_DIR / "step2_perceiver_out.npy", perceiver_out.detach().cpu().numpy())
    np.save(OUTPUT_DIR / "step2_emotion_token.npy", emotion_token.detach().cpu().numpy())
    np.save(OUTPUT_DIR / "step2_final_cond.npy", final_cond.detach().cpu().numpy())

    # =========================================================================
    # SAVE CONFIG
    # =========================================================================
    config = {
        "text": text,
        "voice": voice_name,
        "device": device,
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 80)
    print("VERIFICATION OUTPUTS SAVED")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Files:")
    for f in sorted(OUTPUT_DIR.glob("*.npy")):
        arr = np.load(f)
        print(f"  {f.name}: {arr.shape} ({arr.dtype})")
    print()
    print("Now run Swift verification to compare:")
    print("  cd swift/test_scripts/VerifyLive && swift build && .build/debug/VerifyLive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live verification for Python vs Swift")
    parser.add_argument("--text", "-t", default="Hello world",
                        help="Text to synthesize")
    parser.add_argument("--voice", "-v", default="baked_voice",
                        help="Voice name (without .pt extension)")
    parser.add_argument("--device", "-d", default="cpu",
                        help="Device (cpu, mps, cuda)")
    args = parser.parse_args()

    verify_t3_pipeline(args.text, args.voice, args.device)
