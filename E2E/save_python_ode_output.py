"""
Save Python ODE output (before finalProj) for forensic comparison.
We need to hook into the decoder to save the intermediate state.
"""
import torch
from safetensors.torch import load_file, save_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("SAVE PYTHON ODE OUTPUT")
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

# Monkey-patch both encoder_proj and final_proj to capture intermediate outputs
decoder = model.s3gen.flow.decoder
estimator = decoder.estimator
flow = model.s3gen.flow

# Hook encoder_proj to capture encoder output
saved_encoder_output = None
original_encoder_proj_forward = flow.encoder_proj.forward

def hooked_encoder_proj_forward(input):
    """Hook into encoder_proj to save encoder output"""
    global saved_encoder_output
    output = original_encoder_proj_forward(input)
    saved_encoder_output = output.detach().cpu().contiguous()
    print(f"   [HOOKED] Captured encoder output (after encoder_proj):")
    print(f"     Shape: {output.shape}")
    print(f"     Range: [{output.min().item():.6f}, {output.max().item():.6f}]")
    print(f"     Mean: {output.mean().item():.6f}")
    return output

flow.encoder_proj.forward = hooked_encoder_proj_forward

# Hook final_proj to capture ODE output
saved_ode_output = None
original_final_proj_forward = estimator.final_proj.forward

def hooked_final_proj_forward(input):
    """Hook into final_proj to save its input (which is the ODE output)"""
    global saved_ode_output
    saved_ode_output = input.detach().cpu().contiguous()
    print(f"   [HOOKED] Captured ODE output before final_proj:")
    print(f"     Shape: {input.shape}")
    print(f"     Range: [{input.min().item():.6f}, {input.max().item():.6f}]")
    print(f"     Mean: {input.mean().item():.6f}")
    return original_final_proj_forward(input)

estimator.final_proj.forward = hooked_final_proj_forward

print(f"\n🔬 Running inference with hooked final_proj...")

# Prepare inputs
token_len = torch.tensor([generated_tokens.shape[0]], dtype=torch.long, device=device)
prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.long, device=device)
prompt_feat_len = torch.tensor([prompt_feat.shape[1]], dtype=torch.long, device=device)

with torch.no_grad():
    mel_output, _ = model.s3gen.flow.inference(
        token=generated_tokens.unsqueeze(0),
        token_len=token_len,
        prompt_token=prompt_token,
        prompt_token_len=prompt_token_len,
        prompt_feat=prompt_feat,
        prompt_feat_len=prompt_feat_len,
        embedding=speech_emb_matrix,
        finalize=speaker_emb,
        n_timesteps=10
    )

if saved_encoder_output is not None:
    print(f"\n✅ Saving encoder output...")
    save_file({
        "encoder_output": saved_encoder_output
    }, str(OUTPUT_DIR / "python_encoder_output.safetensors"))
    print(f"   Saved to: {OUTPUT_DIR / 'python_encoder_output.safetensors'}")
    print(f"   Shape: {saved_encoder_output.shape}")
    print(f"   Range: [{saved_encoder_output.min().item():.8f}, {saved_encoder_output.max().item():.8f}]")
    print(f"   Mean: {saved_encoder_output.mean().item():.8f}")
else:
    print(f"\n❌ Failed to capture encoder output")

if saved_ode_output is not None:
    print(f"\n✅ Saving ODE output (before final_proj)...")
    save_file({
        "ode_output": saved_ode_output
    }, str(OUTPUT_DIR / "python_ode_output.safetensors"))
    print(f"   Saved to: {OUTPUT_DIR / 'python_ode_output.safetensors'}")
    print(f"   Shape: {saved_ode_output.shape}")
    print(f"   Range: [{saved_ode_output.min().item():.8f}, {saved_ode_output.max().item():.8f}]")
    print(f"   Mean: {saved_ode_output.mean().item():.8f}")
else:
    print(f"\n❌ Failed to capture ODE output")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
