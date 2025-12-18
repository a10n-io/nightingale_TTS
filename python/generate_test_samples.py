#!/usr/bin/env python3
"""
Generate test audio samples for a voice using test sentences.
"""

import torch
import torchaudio as ta
import json
from pathlib import Path
from datetime import datetime
import argparse
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Paths
PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
MODEL_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICE_DIR = PROJECT_ROOT / "baked_voices"
OUTPUT_DIR = PROJECT_ROOT / "test_audio"
TEST_SENTENCES = PROJECT_ROOT / "E2E" / "test_sentences.json"

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def generate_samples(voice_name: str, languages: list = ["en", "nl"]):
    """Generate test samples for a voice."""
    print(f"Using device: {device}")
    print(f"Voice: {voice_name}")

    # Load test sentences
    with open(TEST_SENTENCES, "r") as f:
        sentences = json.load(f)

    # Check for voice source (baked voice or ref audio)
    baked_path = VOICE_DIR / voice_name / "baked_voice.pt"
    ref_audio_path = VOICE_DIR / voice_name / "ref_audio.wav"

    # Load model
    print(f"Loading model from: {MODEL_DIR}")
    model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device=device)

    # Prepare conditionals
    if baked_path.exists():
        print(f"Using baked voice: {baked_path}")
        from chatterbox.mtl_tts import Conditionals
        model.conds = Conditionals.load(str(baked_path), map_location=device)
    elif ref_audio_path.exists():
        print(f"Using reference audio: {ref_audio_path}")
        model.prepare_conditionals(str(ref_audio_path), exaggeration=0.5)
    else:
        raise FileNotFoundError(f"No voice found for {voice_name}")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nGenerating {len(sentences)} samples for {len(languages)} languages...\n")

    for sentence in sentences:
        sentence_id = sentence["id"]
        description = sentence["description"]

        for lang in languages:
            text_key = f"text_{lang}"
            if text_key not in sentence:
                continue

            text = sentence[text_key]
            print(f"[{sentence_id}][{lang}] {description}")
            print(f"  Text: {text[:60]}{'...' if len(text) > 60 else ''}")

            # Generate audio
            wav = model.generate(
                text,
                language_id=lang,
                exaggeration=0.5,
                cfg_weight=0.5,
                temperature=0.8,
            )

            # Save with descriptive filename
            filename = f"{voice_name}_{sentence_id}_{lang}_{timestamp}.wav"
            output_path = OUTPUT_DIR / filename
            ta.save(str(output_path), wav, model.sr)
            print(f"  ✓ Saved: {filename}\n")

    print("=" * 60)
    print("Generation complete!")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test audio samples")
    parser.add_argument("--voice", "-v", required=True,
                        help="Voice name (e.g., 'samantha', 'sujano')")
    parser.add_argument("--lang", "-l", nargs="+", default=["en", "nl"],
                        help="Languages to generate (default: en nl)")
    args = parser.parse_args()

    generate_samples(args.voice, args.lang)
