"""
Compare Python and Swift sujano mel outputs to find the divergence.
"""
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from safetensors.torch import save_file

print("=" * 80)
print("GENERATE PYTHON SUJANO MEL FOR COMPARISON")
print("=" * 80)

MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

sujano_ref = str(PROJECT_ROOT / "baked_voices/sujano/ref_audio.wav")
print(f"\nPreparing conditionals with: {sujano_ref}")
model.prepare_conditionals(sujano_ref)

test_text = "Wow! I absolutely cannot believe that it worked on the first try!"
print(f"\nGenerating for: {test_text}")

# Get speech tokens
tokens = model.tokenizer.encode([test_text], language_id="en")[0]
print(f"Speech tokens: {tokens.shape}")

# Generate mel using S3Gen
with torch.no_grad():
    mel = model.s3gen.generate_mel(
        tokens.to(device).unsqueeze(0),
        model.conds.s3gen_dict
    )

print(f"\n📊 Python Sujano Mel:")
print(f"   Shape: {mel.shape}")
print(f"   Range: [{mel.min().item():.6f}, {mel.max().item():.6f}]")
print(f"   Mean: {mel.mean().item():.6f}, Std: {mel.std().item():.6f}")

# Save for comparison
output_path = PROJECT_ROOT / "test_audio/python_sujano_mel.safetensors"
save_file({"mel": mel.cpu()}, str(output_path))
print(f"\n✅ Saved: {output_path}")

print("\n" + "=" * 80)
