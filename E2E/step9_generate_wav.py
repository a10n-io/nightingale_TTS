#!/Users/a10n/Projects/nightingale_TTS/python/venv/bin/python
"""
E2E Step 9: Generate .wav audio file from full production pipeline.

This script runs the complete TTS pipeline (Steps 1-8) and outputs a .wav file.
Uses the same deterministic settings as verify_e2e.py for reproducibility.

Usage:
    python E2E/step9_generate_wav.py
    python E2E/step9_generate_wav.py --text "Custom text here"
    python E2E/step9_generate_wav.py --voice samantha --lang en
    python E2E/step9_generate_wav.py --output test_audio/my_test.wav
"""

import torch
import torchaudio as ta
import numpy as np
import random
import json
import argparse
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
MODEL_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICE_DIR = PROJECT_ROOT / "baked_voices"
E2E_DIR = PROJECT_ROOT / "E2E"
OUTPUT_DIR = PROJECT_ROOT / "test_audio"

# Deterministic settings (MUST match verify_e2e.py)
SEED = 42
TEMPERATURE = 0.001  # Near-deterministic for E2E testing
TOP_P = 1.0
REPETITION_PENALTY = 2.0
MIN_P = 0.05
CFG_WEIGHT = 0.5


def set_deterministic_seeds(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'manual_seed'):
        torch.mps.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_wav(
    text: str,
    voice: str = "samantha",
    language: str = "en",
    output_path: str = None,
    device: str = "cpu",
) -> Path:
    """
    Generate .wav file using the full E2E pipeline.

    Args:
        text: Text to synthesize
        voice: Voice name (e.g., "samantha", "sujano")
        language: Language code (e.g., "en", "nl")
        output_path: Optional output path (default: test_audio/e2e_<voice>_<timestamp>.wav)
        device: Device to run on (cpu, mps, cuda)

    Returns:
        Path to generated .wav file
    """
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals, punc_norm
    from chatterbox.models.s3gen.s3gen import drop_invalid_tokens
    import torch.nn.functional as F

    print("=" * 80)
    print("E2E STEP 9: GENERATE .WAV FROM FULL PIPELINE")
    print("=" * 80)
    print()

    # Set deterministic seeds
    set_deterministic_seeds(SEED)

    print(f"Settings:")
    print(f"  Seed: {SEED}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Top-P: {TOP_P}")
    print(f"  Repetition Penalty: {REPETITION_PENALTY}")
    print(f"  Min-P: {MIN_P}")
    print(f"  CFG Weight: {CFG_WEIGHT}")
    print(f"  Device: {device}")
    print()

    # Load model
    print("Loading model...")
    model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device=device)
    print("  Model loaded")

    # Load voice
    voice_path = VOICE_DIR / voice / "baked_voice.pt"
    if not voice_path.exists():
        raise FileNotFoundError(f"Voice not found: {voice_path}")
    model.conds = Conditionals.load(str(voice_path), map_location=device)
    print(f"  Voice loaded: {voice}")
    print()

    # =========================================================================
    # Step 1: Tokenization
    # =========================================================================
    print("Step 1: Tokenization")
    text_normalized = punc_norm(text)
    text_tokens = model.tokenizer.text_to_tokens(text_normalized, language_id=language.lower())
    text_tokens = text_tokens.to(device)
    print(f"  Input text: \"{text}\"")
    print(f"  Normalized: \"{text_normalized}\"")
    print(f"  Token count: {text_tokens.shape[1]}")

    # Prepare for T3 (CFG + SOT/EOT padding)
    text_tokens_cfg = torch.cat([text_tokens, text_tokens], dim=0)  # [2, N]
    sot = model.t3.hp.start_text_token  # 255
    eot = model.t3.hp.stop_text_token   # 0
    text_tokens_cfg = F.pad(text_tokens_cfg, (1, 0), value=sot)  # Prepend SOT
    text_tokens_cfg = F.pad(text_tokens_cfg, (0, 1), value=eot)  # Append EOT
    print(f"  CFG tokens shape: {text_tokens_cfg.shape}")
    print()

    # =========================================================================
    # Step 2-3: T3 Conditioning & Generation
    # =========================================================================
    print("Step 2-3: T3 Conditioning & Generation")
    with torch.inference_mode():
        speech_tokens = model.t3.inference(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens_cfg,
            max_new_tokens=1000,
            temperature=TEMPERATURE,
            cfg_weight=CFG_WEIGHT,
            repetition_penalty=REPETITION_PENALTY,
            min_p=MIN_P,
            top_p=TOP_P,
        )
        speech_tokens = speech_tokens[0]
        if speech_tokens.dim() == 1:
            speech_tokens = speech_tokens.unsqueeze(0)
        speech_tokens = drop_invalid_tokens(speech_tokens)

    # Ensure speech_tokens is 2D [1, T]
    if speech_tokens.dim() == 1:
        speech_tokens = speech_tokens.unsqueeze(0)

    token_count = speech_tokens.shape[1] if speech_tokens.dim() == 2 else speech_tokens.shape[0]
    print(f"  Generated speech tokens: {token_count}")
    print()

    # =========================================================================
    # Steps 4-8: S3Gen (Token -> Mel -> Audio)
    # =========================================================================
    # Use the high-level S3Gen.inference() which handles:
    # - Input preparation (Steps 4-5)
    # - Encoder (Step 6)
    # - ODE Solver / Flow Matching (Step 7)
    # - Vocoder (Step 8)
    print("Steps 4-8: S3Gen (Token -> Mel -> Audio)")
    gen_conds = model.conds.gen
    print(f"  Prompt tokens: {gen_conds['prompt_token_len'].item()}")
    print(f"  Speech tokens: {token_count}")

    with torch.inference_mode():
        result = model.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=gen_conds,
            drop_invalid_tokens=False,  # Already dropped above
            n_cfm_timesteps=10,
        )

    # Handle tuple return (audio, source) if present
    if isinstance(result, tuple):
        audio = result[0]
    else:
        audio = result

    print(f"  Audio shape: {audio.shape}")
    print(f"  Audio range: [{audio.min().item():.4f}, {audio.max().item():.4f}]")

    # Calculate duration
    sample_rate = 24000
    duration = audio.shape[1] / sample_rate
    print(f"  Duration: {duration:.2f}s at {sample_rate}Hz")
    print()

    # =========================================================================
    # Step 9: Save .wav
    # =========================================================================
    print("Step 9: Save .wav")

    # Ensure audio is in correct format
    if audio.dim() == 3:
        audio = audio.squeeze(0)  # [1, 1, T] -> [1, T]
    elif audio.dim() == 1:
        audio = audio.unsqueeze(0)  # [T] -> [1, T]

    # Normalize audio to prevent clipping
    audio_max = audio.abs().max()
    if audio_max > 1.0:
        audio = audio / audio_max

    # Determine output path
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"e2e_{voice}_{language}_{timestamp}.wav"
    else:
        output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as WAV
    ta.save(str(output_path), audio.cpu(), sample_rate)
    print(f"  Saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
    print()

    print("=" * 80)
    print("E2E GENERATION COMPLETE")
    print("=" * 80)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="E2E Step 9: Generate .wav from full pipeline")
    parser.add_argument("--text", "-t", default=None,
                        help="Text to synthesize (default: uses first test sentence)")
    parser.add_argument("--voice", "-v", default="samantha",
                        help="Voice name (default: samantha)")
    parser.add_argument("--lang", "-l", default="en",
                        help="Language code (default: en)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path (default: test_audio/e2e_<voice>_<timestamp>.wav)")
    parser.add_argument("--device", "-d", default="cpu",
                        help="Device (cpu, mps, cuda)")
    args = parser.parse_args()

    # Use default test sentence if no text provided
    if args.text is None:
        # Load from test_sentences.json
        with open(E2E_DIR / "test_sentences.json", "r") as f:
            sentences = json.load(f)
        text_key = f"text_{args.lang}"
        if text_key in sentences[0]:
            args.text = sentences[0][text_key]
        else:
            args.text = "Do you think the model can handle the rising intonation at the end of this sentence?"

    generate_wav(
        text=args.text,
        voice=args.voice,
        language=args.lang,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
