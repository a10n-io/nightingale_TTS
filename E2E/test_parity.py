#!/usr/bin/env python3
"""
Test script for Python/Swift parity verification.
Uses identical text, parameters, and voice as Swift GenerateAudio test.
"""
import sys
sys.path.insert(0, str(__file__).replace('/E2E/test_parity.py', '/python/chatterbox/src'))

import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file, save_file
import soundfile as sf

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICES_DIR = PROJECT_ROOT / "baked_voices"
OUTPUT_DIR = PROJECT_ROOT / "test_audio"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("PYTHON/SWIFT PARITY TEST")
print("=" * 80)

# Test text - MUST MATCH Swift test
TEST_TEXT = "Wow! I absolutely cannot believe that it worked on the first try!"
print(f"\nTest text: \"{TEST_TEXT}\"")

# Parameters - MUST MATCH Swift
TEMPERATURE = 0.0001  # Near-zero for deterministic (PyTorch crashes at 0)
CFG_WEIGHT = 0.5
REPETITION_PENALTY = 2.0
TOP_P = 1.0
MIN_P = 0.05
MAX_NEW_TOKENS = 1000

print(f"\nParameters:")
print(f"  temperature: {TEMPERATURE}")
print(f"  cfg_weight: {CFG_WEIGHT}")
print(f"  repetition_penalty: {REPETITION_PENALTY}")
print(f"  top_p: {TOP_P}")
print(f"  min_p: {MIN_P}")
print(f"  max_new_tokens: {MAX_NEW_TOKENS}")

# Load model
print("\n" + "=" * 80)
print("Loading ChatterboxMultilingualTTS...")
print("=" * 80)

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)
print("✅ Model loaded")

# Load voice - use samantha (same as Swift)
print("\nLoading voice: samantha")
voice_path = VOICES_DIR / "samantha" / "baked_voice.safetensors"
if voice_path.exists():
    voice_data = load_file(str(voice_path))

    # Extract T3 conditioning
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.mtl_tts import Conditionals

    t3_cond = T3Cond(
        speaker_emb=voice_data["t3.speaker_emb"].to(device),
        cond_prompt_speech_tokens=voice_data["t3.cond_prompt_speech_tokens"].to(device),
        emotion_adv=(torch.ones(1, 1, 1) * 0.5).to(device),  # Default emotion
    )

    # Extract S3Gen conditioning
    prompt_token = voice_data["gen.prompt_token"].to(device)
    prompt_feat = voice_data["gen.prompt_feat"].to(device)
    gen_dict = {
        "embedding": voice_data["gen.embedding"].to(device),
        "prompt_token": prompt_token,
        "prompt_feat": prompt_feat,
        "prompt_token_len": torch.tensor([prompt_token.shape[1]], device=device),
        "prompt_feat_len": torch.tensor([prompt_feat.shape[1]], device=device),
    }

    model.conds = Conditionals(t3_cond, gen_dict)
    print("✅ Voice loaded from baked_voice.safetensors")
else:
    print(f"❌ Voice file not found: {voice_path}")
    sys.exit(1)

# Generate audio
print("\n" + "=" * 80)
print("GENERATING AUDIO")
print("=" * 80)
print(f"Text: \"{TEST_TEXT}\"")

with torch.inference_mode():
    wav = model.generate(
        text=TEST_TEXT,
        language_id="en",
        temperature=TEMPERATURE,
        cfg_weight=CFG_WEIGHT,
        repetition_penalty=REPETITION_PENALTY,
        top_p=TOP_P,
        min_p=MIN_P,
    )

# Convert to numpy
audio = wav.squeeze().cpu().numpy()
duration = len(audio) / 24000.0

print(f"✅ Generated {len(audio)} samples ({duration:.2f}s)")

# Frequency analysis (same as Swift)
def analyze_frequency(samples, sample_rate=24000):
    low_energy = 0
    high_energy = 0
    for freq in range(100, 501, 50):
        real_sum = imag_sum = 0
        for i, sample in enumerate(samples[:10000]):
            angle = 2.0 * np.pi * freq * i / sample_rate
            real_sum += sample * np.cos(angle)
            imag_sum += sample * np.sin(angle)
        low_energy += np.sqrt(real_sum**2 + imag_sum**2)

    for freq in range(5000, 10001, 500):
        real_sum = imag_sum = 0
        for i, sample in enumerate(samples[:10000]):
            angle = 2.0 * np.pi * freq * i / sample_rate
            real_sum += sample * np.cos(angle)
            imag_sum += sample * np.sin(angle)
        high_energy += np.sqrt(real_sum**2 + imag_sum**2)

    return low_energy, high_energy

low_e, high_e = analyze_frequency(audio)
total_e = low_e + high_e
print(f"\nFrequency analysis:")
print(f"  Low freq (100-500 Hz): {100 * low_e / total_e:.1f}%")
print(f"  High freq (5k-10k Hz): {100 * high_e / total_e:.1f}%")
if low_e > high_e:
    print("  ✅ Correct: Low frequency dominant (speech)")
else:
    print("  ⚠️  Warning: High frequency dominant")

# Save audio
output_path = OUTPUT_DIR / "python_parity_test.wav"
sf.write(str(output_path), audio, 24000)
print(f"\n✅ Saved: {output_path}")

# Also save as safetensors for direct comparison
audio_tensor = torch.from_numpy(audio).float()
save_file({"audio": audio_tensor}, str(OUTPUT_DIR / "python_parity_audio.safetensors"))
print(f"✅ Saved tensor: {OUTPUT_DIR / 'python_parity_audio.safetensors'}")

print("\n" + "=" * 80)
print("✅ PYTHON TEST COMPLETE!")
print(f"   Output: {output_path}")
print("=" * 80)
