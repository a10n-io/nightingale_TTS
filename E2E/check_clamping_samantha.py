"""
Generate and check Samantha mel with clamping fix.
"""
import torch
from safetensors.torch import load_file, save_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("CHECK SAMANTHA MEL WITH CLAMPING")
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

print(f"\nGenerating Python mel for: {test_text}")
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
generated_mel = mel[:, :, prompt_len:]

print(f"\n📊 Python Decoder Mel (Generated Region):")
print(f"   Shape: {generated_mel.shape}")
print(f"   Range: [{generated_mel.min().item():.6f}, {generated_mel.max().item():.6f}]")
print(f"   Mean: {generated_mel.mean().item():.6f}")

# Check characteristics
max_val = generated_mel.max().item()
if max_val > 0:
    print(f"\n   ⚠️  Has positive values (max={max_val:.6f})")
elif max_val == 0.0:
    print(f"\n   ✅ Max is exactly 0.0 (suggests clamping)")
else:
    print(f"\n   ℹ️  Naturally below 0.0 (max={max_val:.6f})")

# Save for comparison
output_path = PROJECT_ROOT / "test_audio/cross_validate/python_mel_samantha.safetensors"
save_file({"mel": generated_mel.cpu()}, str(output_path))
print(f"\n✅ Saved Python mel to: {output_path.name}")

print("\n" + "=" * 80)
print("Now check Swift mel by comparing with this reference.")
print("Swift mel should have max ≤ 0.0 after clamping fix.")
print("=" * 80)
