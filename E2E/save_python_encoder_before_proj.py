"""
Save Python encoder output BEFORE encoder_proj to compare with Swift.
"""
import torch
from safetensors.torch import load_file, save_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("SAVE PYTHON ENCODER OUTPUT (BEFORE encoder_proj)")
print("=" * 80)

MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICE_DIR = PROJECT_ROOT / "baked_voices" / "samantha"
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "forensic"

device = "cpu"

print(f"\nLoading model...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load voice and tokens
voice_data = load_file(str(VOICE_DIR / "baked_voice.safetensors"))
prompt_token = voice_data["gen.prompt_token"].to(device)
tokens_data = load_file(str(PROJECT_ROOT / "test_audio/cross_validate/python_speech_tokens.safetensors"))
tokens = tokens_data["speech_tokens"].to(device)

# Hook encoder to capture output BEFORE encoder_proj
encoder = model.s3gen.flow.encoder
saved_encoder_output_before_proj = None

original_forward = encoder.forward

def hooked_forward(xs, xs_lens, *args, **kwargs):
    global saved_encoder_output_before_proj
    output, masks = original_forward(xs, xs_lens, *args, **kwargs)
    saved_encoder_output_before_proj = output.detach().cpu().contiguous()
    print(f"[HOOK] Captured encoder output BEFORE encoder_proj:")
    print(f"  Shape: {output.shape}")
    print(f"  Mean: {output.mean().item():.6f}")
    print(f"  Std: {output.std().item():.6f}")
    return output, masks

encoder.forward = hooked_forward

# Run inference
print(f"\nRunning inference...")
token_len = torch.tensor([tokens.shape[0]], dtype=torch.long, device=device)
prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.long, device=device)
prompt_feat = voice_data["gen.prompt_feat"].to(device)
prompt_feat_len = torch.tensor([prompt_feat.shape[1]], dtype=torch.long, device=device)

with torch.no_grad():
    _ = model.s3gen.flow.inference(
        token=tokens.unsqueeze(0),
        token_len=token_len,
        prompt_token=prompt_token,
        prompt_token_len=prompt_token_len,
        prompt_feat=prompt_feat,
        prompt_feat_len=prompt_feat_len,
        embedding=voice_data["gen.embedding"].to(device),
        finalize=voice_data["t3.speaker_emb"].to(device),
        n_timesteps=10
    )

if saved_encoder_output_before_proj is not None:
    print(f"\n✅ Saving encoder output BEFORE encoder_proj...")
    save_file({
        "encoder_before_proj": saved_encoder_output_before_proj
    }, str(OUTPUT_DIR / "python_encoder_before_proj.safetensors"))
    print(f"  Shape: {saved_encoder_output_before_proj.shape}")
    print(f"  Mean: {saved_encoder_output_before_proj.mean().item():.6f}")
    print(f"  Std: {saved_encoder_output_before_proj.std().item():.6f}")
else:
    print(f"\n❌ Failed to capture")

print(f"\n" + "=" * 80)
