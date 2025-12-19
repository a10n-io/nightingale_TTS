#!/usr/bin/env python3
"""
Generate test audio files for E2E test sentences.
Uses the Python implementation which has been verified to match Swift bit-for-bit.
"""
import json
import torch
import numpy as np
from pathlib import Path
import scipy.io.wavfile as wavfile

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
MODEL_DIR = PROJECT_ROOT / "models" / "chatterbox"
OUTPUT_DIR = PROJECT_ROOT / "test_audio"
TEST_SENTENCES_PATH = PROJECT_ROOT / "E2E" / "test_sentences.json"

def main():
    print("=" * 80)
    print("TEST AUDIO GENERATION (Python/PyTorch - verified Swift-equivalent)")
    print("=" * 80)
    print()

    # Load test sentences
    with open(TEST_SENTENCES_PATH) as f:
        test_sentences = json.load(f)

    print(f"Loaded {len(test_sentences)} test sentences")
    print()

    # Load model
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    print("Loading Chatterbox model...")
    model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device="cpu")
    print("✅ Model loaded")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    voices = ["samantha", "sujano"]
    total_generated = 0
    total_tests = len(voices) * len(test_sentences) * 2  # 2 languages per sentence

    for voice in voices:
        print("=" * 80)
        print(f"VOICE: {voice}")
        print("=" * 80)
        print()

        # Get voice reference audio path
        voice_ref_path = PROJECT_ROOT / "baked_voices" / voice / "ref_audio.wav"

        for sentence in test_sentences:
            # English
            print(f"[{total_generated + 1}/{total_tests}] {voice} | {sentence['id']} | en")
            print(f"  Text: \"{sentence['text_en']}\"")

            result_en = model.generate(
                sentence['text_en'],
                language_id="en",
                audio_prompt_path=str(voice_ref_path),
                temperature=0.001,
                exaggeration=0.5
            )
            # Result is a tensor, convert to numpy
            audio_en = result_en.cpu().numpy().flatten()

            # Save as WAV (24kHz, mono, float32)
            output_file_en = OUTPUT_DIR / f"python_{voice}_{sentence['id']}_en.wav"
            wavfile.write(str(output_file_en), 24000, audio_en)

            duration_en = len(audio_en) / 24000
            print(f"  ✅ Generated {len(audio_en)} samples")
            print(f"  Duration: {duration_en:.2f}s @ 24kHz")
            print(f"  Saved: {output_file_en.name}")
            print()
            total_generated += 1

            # Dutch
            print(f"[{total_generated + 1}/{total_tests}] {voice} | {sentence['id']} | nl")
            print(f"  Text: \"{sentence['text_nl']}\"")

            result_nl = model.generate(
                sentence['text_nl'],
                language_id="nl",
                audio_prompt_path=str(voice_ref_path),
                temperature=0.001,
                exaggeration=0.5
            )
            # Result is a tensor, convert to numpy
            audio_nl = result_nl.cpu().numpy().flatten()

            # Save as WAV
            output_file_nl = OUTPUT_DIR / f"python_{voice}_{sentence['id']}_nl.wav"
            wavfile.write(str(output_file_nl), 24000, audio_nl)

            duration_nl = len(audio_nl) / 24000
            print(f"  ✅ Generated {len(audio_nl)} samples")
            print(f"  Duration: {duration_nl:.2f}s @ 24kHz")
            print(f"  Saved: {output_file_nl.name}")
            print()
            total_generated += 1

    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print()
    print(f"✅ Successfully generated {total_generated}/{total_tests} audio files")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # List all generated files
    print("Generated files:")
    for wav_file in sorted(OUTPUT_DIR.glob("*.wav")):
        size_kb = wav_file.stat().st_size // 1024
        print(f"  {wav_file.name} ({size_kb} KB)")

if __name__ == "__main__":
    main()
