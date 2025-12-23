"""
Trace Python decoder pipeline step-by-step to understand exact behavior.
Save intermediate outputs at each critical stage.
"""
import torch
from safetensors.torch import load_file, save_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("TRACE PYTHON DECODER - STEP BY STEP")
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

# We need to instrument the decoder to save intermediate outputs
# Let's access the decoder directly and trace through it

print(f"\n🔬 Tracing decoder internals...")

# Get the flow decoder
decoder = model.s3gen.flow.decoder
estimator = decoder.estimator

print(f"\n1️⃣ Checking finalProj weights:")
print(f"   finalProj type: {type(estimator.final_proj).__name__}")
print(f"   finalProj weight shape: {estimator.final_proj.weight.shape}")
print(f"   finalProj bias shape: {estimator.final_proj.bias.shape if estimator.final_proj.bias is not None else 'None'}")
print(f"   finalProj weight range: [{estimator.final_proj.weight.min().item():.8f}, {estimator.final_proj.weight.max().item():.8f}]")
print(f"   finalProj weight mean: {estimator.final_proj.weight.mean().item():.8f}")
print(f"   finalProj weight std: {estimator.final_proj.weight.std().item():.8f}")
if estimator.final_proj.bias is not None:
    print(f"   finalProj bias range: [{estimator.final_proj.bias.min().item():.8f}, {estimator.final_proj.bias.max().item():.8f}]")
    print(f"   finalProj bias mean: {estimator.final_proj.bias.mean().item():.8f}")
    print(f"   finalProj bias std: {estimator.final_proj.bias.std().item():.8f}")

# Save finalProj weights for comparison
save_file({
    "final_proj_weight": estimator.final_proj.weight.cpu(),
    "final_proj_bias": estimator.final_proj.bias.cpu() if estimator.final_proj.bias is not None else torch.zeros(1)
}, str(OUTPUT_DIR / "python_finalproj_weights.safetensors"))
print(f"\n   ✅ Saved finalProj weights")

print(f"\n2️⃣ Checking decoder architecture:")
print(f"   Decoder type: {type(decoder).__name__}")
print(f"   Decoder attributes: {[attr for attr in dir(decoder) if not attr.startswith('_')][:20]}")

# Check if we can access ODE solver parameters
if hasattr(decoder, 'estimator'):
    print(f"\n   Estimator (velocity network):")
    print(f"   Type: {type(decoder.estimator).__name__}")

print(f"\n3️⃣ Decoder configuration:")
if hasattr(decoder, 'decoder_conf'):
    print(f"   decoder_conf: {decoder.decoder_conf}")

print(f"\n4️⃣ Running inference to get intermediate outputs...")

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

print(f"\n5️⃣ Final mel output:")
print(f"   Shape: {mel_output.shape}")
print(f"   Range: [{mel_output.min().item():.8f}, {mel_output.max().item():.8f}]")
print(f"   Mean: {mel_output.mean().item():.8f}")
print(f"   Std: {mel_output.std().item():.8f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("1. Compare finalProj weights between Python and Swift")
print("2. Instrument ODE solver to save intermediate states")
print("3. Find where 0.91 dB divergence originates")
print("=" * 80)
