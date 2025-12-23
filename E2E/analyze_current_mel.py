"""
Analyze current Swift mel output to determine next fix needed.
"""
import torch
from safetensors.torch import load_file, save_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("ANALYZE CURRENT MEL OUTPUT")
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

# Test text
test_text = "Wow! I absolutely cannot believe that it worked on the first try!"

print(f"\nGenerating Python reference mel...")
with torch.no_grad():
    tokens = model.tokenizer.encode([test_text], language_id="en")[0].to(device)
    generated_tokens = model.t3_model(tokens)

    # Run decoder to get mel
    mel = model.s3gen.decoder(
        tokens=generated_tokens,
        cond=model.s3gen.cond_enc(speech_emb_matrix, speaker_emb),
        prompt_token=prompt_token,
        prompt_feat=prompt_feat
    )

# Extract generated region (after prompt)
prompt_len = prompt_token.shape[1]
python_mel_gen = mel[:, :, prompt_len:]

print(f"\n📊 Python Decoder Mel (Generated Region):")
print(f"   Shape: {python_mel_gen.shape}")
print(f"   Range: [{python_mel_gen.min().item():.6f}, {python_mel_gen.max().item():.6f}]")
print(f"   Mean: {python_mel_gen.mean().item():.6f}")
print(f"   Std: {python_mel_gen.std().item():.6f}")

# Save for Swift comparison
output_path = PROJECT_ROOT / "test_audio/cross_validate/python_mel.safetensors"
save_file({"mel": python_mel_gen.cpu()}, str(output_path))
print(f"   ✅ Saved to: {output_path.name}")

# Check if Swift mel exists from latest cross-validation
swift_mel_path = PROJECT_ROOT / "test_audio/cross_validate/swift_mel.safetensors"
if swift_mel_path.exists():
    swift_data = load_file(str(swift_mel_path))
    swift_mel_full = swift_data["mel"]

    # Extract generated region
    if swift_mel_full.shape[2] > prompt_len:
        swift_mel_gen = swift_mel_full[:, :, prompt_len:]

        print(f"\n📊 Swift Decoder Mel (Generated Region, WITH CLAMPING):")
        print(f"   Shape: {swift_mel_gen.shape}")
        print(f"   Range: [{swift_mel_gen.min().item():.6f}, {swift_mel_gen.max().item():.6f}]")
        print(f"   Mean: {swift_mel_gen.mean().item():.6f}")
        print(f"   Std: {swift_mel_gen.std().item():.6f}")

        # Calculate difference
        mean_diff = python_mel_gen.mean().item() - swift_mel_gen.mean().item()
        max_diff = python_mel_gen.max().item() - swift_mel_gen.max().item()

        print(f"\n🔍 Analysis:")
        print(f"   Mean difference: {mean_diff:.6f} dB (Python - Swift)")
        print(f"   Max difference: {max_diff:.6f} dB")

        if swift_mel_gen.max().item() == 0.0:
            print(f"   ✅ Clamping is working (max = 0.0)")
        else:
            print(f"   ⚠️  Clamping not active or mel naturally < 0.0")

        if abs(mean_diff) > 0.5:
            print(f"\n❌ ISSUE: Swift mel is {mean_diff:.2f} dB {'darker' if mean_diff > 0 else 'brighter'} than Python")
            print(f"   This causes {'mumbled words' if mean_diff > 0 else 'overly bright'} speech")

            # Calculate required scaling
            # In log space, adding a constant = multiplying in linear space
            # We want to add mean_diff dB to Swift output
            required_scale = 10 ** (mean_diff / 20)  # Convert dB to linear scale
            print(f"\n💡 RECOMMENDATION:")
            print(f"   Apply scaling factor: {required_scale:.4f}")
            print(f"   After finalProj, use: h = h * {required_scale:.4f}")
            print(f"   This will brighten Swift mel to match Python")
        else:
            print(f"\n✅ Mel brightness is close to Python (within 0.5 dB)")
    else:
        print(f"\n⚠️  Swift mel doesn't have generated region (too short)")
else:
    print(f"\n⚠️  Swift mel not found at {swift_mel_path}")
    print(f"   Run cross-validation to generate it for comparison")

print("\n" + "=" * 80)
