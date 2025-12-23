"""
Save mel spectrograms from decoder (before vocoder) for comparison.
"""
import torch
from safetensors.torch import load_file, save_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("SAVE DECODER MEL OUTPUTS")
print("=" * 80)

MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# Load model
print("\nLoading ChatterboxMultilingualTTS...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load Python tokens
python_tokens_path = PROJECT_ROOT / "test_audio/cross_validate/python_speech_tokens.safetensors"
python_tokens_data = load_file(str(python_tokens_path))
python_tokens = python_tokens_data["speech_tokens"].to(device)

print(f"\nPython tokens shape: {python_tokens.shape}")
print(f"Python tokens: {python_tokens[:20].tolist()}")

# Load voice for reference dict
voice_path = PROJECT_ROOT / "baked_voices/samantha/baked_voice.safetensors"
voice_data = load_file(str(voice_path))

ref_dict = {
    "embedding": voice_data["gen.embedding"].to(device),
    "prompt_token": voice_data["gen.prompt_token"].to(device),
    "prompt_feat": voice_data["gen.prompt_feat"].to(device),
    "prompt_token_len": torch.tensor([voice_data["gen.prompt_token"].shape[1]], device=device),
    "prompt_feat_len": torch.tensor([voice_data["gen.prompt_feat"].shape[1]], device=device),
}

# Run decoder to get mel (before vocoder)
print("\n🔍 Running decoder (flow matching) to get mel...")
with torch.inference_mode():
    # Get mel from decoder - use flow_inference to get mel before vocoder
    mel = model.s3gen.flow_inference(
        speech_tokens=python_tokens,
        ref_dict=ref_dict,
        n_cfm_timesteps=10,
        finalize=True,
    )

print(f"\n📊 Decoder output mel:")
print(f"   Shape: {mel.shape}")
print(f"   Range: [{mel.min().item():.6f}, {mel.max().item():.6f}]")
print(f"   Mean: {mel.mean().item():.6f}, Std: {mel.std().item():.6f}")

# Check a few mel channels
print(f"\n📊 Mel channel statistics:")
for i in [0, 20, 40, 60, 79]:
    channel = mel[0, i, :]
    print(f"   Channel {i}: mean={channel.mean().item():.6f}, std={channel.std().item():.6f}")

# Save mel
output_path = PROJECT_ROOT / "test_audio/cross_validate/python_mel.safetensors"
save_file({"mel": mel.cpu()}, str(output_path))
print(f"\n✅ Saved Python mel to: {output_path}")

print("\n" + "=" * 80)
