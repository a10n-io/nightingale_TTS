"""
Generate and save RAW Python mel spectrogram for forensic comparison.
No modifications, no clamping - pure decoder output.
"""
import torch
from safetensors.torch import load_file, save_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("GENERATE RAW PYTHON MEL")
print("=" * 80)

MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICE_DIR = PROJECT_ROOT / "baked_voices" / "samantha"
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "forensic"
OUTPUT_DIR.mkdir(exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"\nLoading Python model on {device}...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load samantha voice
voice_data = load_file(str(VOICE_DIR / "baked_voice.safetensors"))
speaker_emb = voice_data["t3.speaker_emb"].to(device)
speech_emb_matrix = voice_data["gen.embedding"].to(device)
prompt_token = voice_data["gen.prompt_token"].to(device)
prompt_feat = voice_data["gen.prompt_feat"].to(device)

# Load Python tokens
tokens_path = PROJECT_ROOT / "test_audio/cross_validate/python_speech_tokens.safetensors"
tokens_data = load_file(str(tokens_path))
generated_tokens = tokens_data["speech_tokens"].to(device)

print(f"\n📊 Inputs:")
print(f"   tokens: {generated_tokens.shape}")
print(f"   speaker_emb: {speaker_emb.shape}")
print(f"   speech_emb_matrix: {speech_emb_matrix.shape}")
print(f"   prompt_token: {prompt_token.shape}")
print(f"   prompt_feat: {prompt_feat.shape}")

print(f"\n🔬 Generating RAW Python mel...")
with torch.no_grad():
    # Prepare inputs in the format expected by flow.inference
    token_len = torch.tensor([generated_tokens.shape[0]], dtype=torch.long, device=device)
    prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.long, device=device)
    prompt_feat_len = torch.tensor([prompt_feat.shape[1]], dtype=torch.long, device=device)

    # Use flow.inference to get the mel output (returns tuple, first element is mel)
    mel_full, _ = model.s3gen.flow.inference(
        token=generated_tokens.unsqueeze(0),  # Add batch dimension
        token_len=token_len,
        prompt_token=prompt_token,
        prompt_token_len=prompt_token_len,
        prompt_feat=prompt_feat,
        prompt_feat_len=prompt_feat_len,
        embedding=speech_emb_matrix,
        finalize=speaker_emb,
        n_timesteps=10
    )

# Note: mel_full contains ONLY the generated region (not the prompt)
# The prompt is used for conditioning but not included in the output
# For 98 tokens, we get 98 * 2 = 196 mel frames
mel_gen = mel_full

print(f"\n📊 Python Mel (RAW - no modifications):")
print(f"   Generated region shape: {mel_gen.shape}")
print(f"   Range: [{mel_gen.min().item():.8f}, {mel_gen.max().item():.8f}]")
print(f"   Mean: {mel_gen.mean().item():.8f}")
print(f"   Std: {mel_gen.std().item():.8f}")
print(f"")
print(f"   Note: This is ONLY the generated region (98 tokens → 196 frames)")
print(f"         Prompt region is used for conditioning but not in output")

# Channel-by-channel stats
print(f"\n   Per-channel statistics:")
for i in [0, 20, 40, 60, 79]:
    chan = mel_gen[0, i, :]
    print(f"     Ch{i:2d}: mean={chan.mean().item():8.5f}, std={chan.std().item():7.5f}, range=[{chan.min().item():8.5f}, {chan.max().item():7.5f}]")

# Save
save_file({
    "mel_gen": mel_gen.cpu()
}, str(OUTPUT_DIR / "python_mel_raw.safetensors"))

print(f"\n✅ Saved to: forensic/python_mel_raw.safetensors")
print("=" * 80)
