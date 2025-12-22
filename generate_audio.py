#!/usr/bin/env python3
"""
Generate audio using Python TTS flow.
Uses baked_voice.safetensors for voice conditioning.
"""
import argparse
import torch
import torchaudio as ta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICE_DIR = PROJECT_ROOT / "baked_voices"
OUTPUT_DIR = PROJECT_ROOT / "test_audio"


def main():
    parser = argparse.ArgumentParser(description="Generate TTS audio")
    parser.add_argument("--text", "-t", type=str, default="Hello, this is a test.",
                        help="Text to synthesize")
    parser.add_argument("--voice", "-v", type=str, default="samantha",
                        help="Voice name (samantha, sujano)")
    parser.add_argument("--language", "-l", type=str, default="en",
                        help="Language code (en, nl, de, etc.)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output WAV file path")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0.0-1.0)")
    parser.add_argument("--exaggeration", type=float, default=0.5,
                        help="Voice exaggeration (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print("=" * 60)
    print("PYTHON TTS GENERATION")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals

    device = "cpu"
    model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device=device)

    # Load voice from safetensors
    voice_path = VOICE_DIR / args.voice / "baked_voice.safetensors"
    if not voice_path.exists():
        print(f"Error: Voice not found at {voice_path}")
        return 1

    model.conds = Conditionals.load(str(voice_path), map_location=device)
    print(f"Voice: {args.voice}")

    # Generate
    print(f"\nText: {args.text}")
    print(f"Language: {args.language}")
    print(f"Temperature: {args.temperature}")
    print("Generating...")

    audio = model.generate(
        text=args.text,
        language_id=args.language,
        exaggeration=args.exaggeration,
        cfg_weight=0.5,
        temperature=args.temperature,
    )

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / f"python_{args.voice}.wav"

    ta.save(str(output_path), audio.cpu(), 24000)

    duration = audio.shape[1] / 24000
    print(f"\nSaved: {output_path}")
    print(f"Duration: {duration:.2f}s")

    return 0


if __name__ == "__main__":
    exit(main())
