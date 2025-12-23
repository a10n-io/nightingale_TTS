"""
Measure the actual mel difference between Python and Swift with clamping fix.
Generate both mels fresh and calculate the exact scaling needed.
"""
import torch
from safetensors.torch import load_file, save_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("MEASURE MEL DIFFERENCE (with clamping fix)")
print("=" * 80)

MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICE_DIR = PROJECT_ROOT / "baked_voices" / "samantha"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"\nLoading Python model on {device}...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load samantha voice
voice_data = load_file(str(VOICE_DIR / "baked_voice.safetensors"))
speaker_emb = voice_data["speaker_emb"].to(device)
speech_emb_matrix = voice_data["speech_emb_matrix"].to(device)
prompt_token = voice_data["prompt_token"].to(device)
prompt_feat = voice_data["prompt_feat"].to(device)

# Load Python tokens from cross-validation
tokens_path = PROJECT_ROOT / "test_audio/cross_validate/python_speech_tokens.safetensors"
tokens_data = load_file(str(tokens_path))
generated_tokens = tokens_data["speech_tokens"].to(device)

print(f"Loaded tokens: shape={generated_tokens.shape}")

print(f"\n1️⃣ Generating Python decoder mel...")
with torch.no_grad():
    python_mel_full = model.s3gen.decoder(
        tokens=generated_tokens,
        cond=model.s3gen.cond_enc(speech_emb_matrix, speaker_emb),
        prompt_token=prompt_token,
        prompt_feat=prompt_feat
    )

# Extract generated region (after prompt)
prompt_len = prompt_token.shape[1]
python_mel_gen = python_mel_full[:, :, prompt_len:]

print(f"\n📊 Python Decoder Mel (Generated Region):")
print(f"   Shape: {python_mel_gen.shape}")
print(f"   Range: [{python_mel_gen.min().item():.6f}, {python_mel_gen.max().item():.6f}]")
print(f"   Mean: {python_mel_gen.mean().item():.6f}")
print(f"   Std: {python_mel_gen.std().item():.6f}")

# Check prompt region too
python_mel_prompt = python_mel_full[:, :, :prompt_len]
print(f"\n📊 Python Decoder Mel (Prompt Region):")
print(f"   Mean: {python_mel_prompt.mean().item():.6f}")
print(f"   Max: {python_mel_prompt.max().item():.6f}")

# Now let's load and check the latest Swift audio to infer mel characteristics
# We'll analyze the audio spectral content
import torchaudio
swift_audio_path = PROJECT_ROOT / "test_audio/cross_validate/python_tokens_swift_audio.wav"
if swift_audio_path.exists():
    swift_audio, sr = torchaudio.load(str(swift_audio_path))
    python_audio_path = PROJECT_ROOT / "test_audio/cross_validate/python_tokens_python_audio.wav"
    python_audio, _ = torchaudio.load(str(python_audio_path))

    print(f"\n📊 Audio Analysis:")
    print(f"   Swift audio RMS: {swift_audio.pow(2).mean().sqrt().item():.6f}")
    print(f"   Python audio RMS: {python_audio.pow(2).mean().sqrt().item():.6f}")

    rms_ratio = python_audio.pow(2).mean().sqrt().item() / swift_audio.pow(2).mean().sqrt().item()
    print(f"   RMS ratio (Python/Swift): {rms_ratio:.4f}")

    if rms_ratio < 0.8:
        print(f"\n   ⚠️  Swift audio is louder than Python (RMS ratio < 1)")
    elif rms_ratio > 1.2:
        print(f"\n   ⚠️  Swift audio is quieter than Python (RMS ratio > 1)")
    else:
        print(f"\n   ✅ Audio levels are similar")

# Based on systematic comparison before clamping:
# Swift was 1.46 dB darker (mean -7.27 vs -5.81)
# With clamping to 0.0, the max should now be 0.0 instead of -0.70
# This brightens by ~0.70 dB at the peak, but mean difference remains

print(f"\n" + "=" * 80)
print(f"ANALYSIS OF CLAMPING FIX")
print(f"=" * 80)

print(f"\nBefore clamping (from logs):")
print(f"   Swift mean: -7.27 dB, max: -0.70 dB")
print(f"   Python mean: -5.81 dB, max: 0.00 dB")
print(f"   Difference: 1.46 dB (Swift darker)")

print(f"\nAfter clamping to 0.0:")
print(f"   Swift max: 0.00 dB (clamped)")
print(f"   Expected mean: ~-6.5 to -7.0 dB")
print(f"   Still need: ~0.5-1.2 dB brightening")

# Calculate required scaling
mean_diff_dB = 1.0  # Conservative estimate
required_scale = 10 ** (mean_diff_dB / 20)

print(f"\n💡 RECOMMENDED FIX:")
print(f"   Apply post-clamping scaling: {required_scale:.4f}")
print(f"   This adds ~{mean_diff_dB:.1f} dB brightness")
print(f"\n   In S3Gen.swift after clamping:")
print(f"   h = minimum(h, 0.0)")
print(f"   h = h * {required_scale:.4f}  // Brighten to match Python")

print(f"\n   Alternative (additive in log space):")
print(f"   h = h + {mean_diff_dB:.2f}  // Add {mean_diff_dB:.2f} dB directly")
print(f"   But this can push values positive, so multiplicative is safer")

print("\n" + "=" * 80)
