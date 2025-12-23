"""
First Layer Probe - Systematic encoder debugging.
Dump outputs at three critical checkpoints to isolate where divergence starts:
1. speech_emb_matrix (raw weights)
2. x (after embedding lookup)
3. x (after positional encoding added)
"""
import torch
from safetensors.torch import load_file, save_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("FIRST LAYER PROBE - Python Encoder")
print("=" * 80)

MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICE_DIR = PROJECT_ROOT / "baked_voices" / "samantha"
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "forensic"
OUTPUT_DIR.mkdir(exist_ok=True)

device = "cpu"  # Use CPU for exact reproducibility

print(f"\nLoading Python model on {device}...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load samantha voice
voice_data = load_file(str(VOICE_DIR / "baked_voice.safetensors"))
speech_emb_matrix = voice_data["gen.embedding"].to(device)
prompt_token = voice_data["gen.prompt_token"].to(device)
prompt_feat = voice_data["gen.prompt_feat"].to(device)

# Load Python tokens
tokens_path = PROJECT_ROOT / "test_audio/cross_validate/python_speech_tokens.safetensors"
tokens_data = load_file(str(tokens_path))
generated_tokens = tokens_data["speech_tokens"].to(device)

print(f"\n📊 Inputs:")
print(f"   tokens: {generated_tokens.shape}")
print(f"   speech_emb_matrix: {speech_emb_matrix.shape}")

# ==============================================================================
# CHECKPOINT 1: speech_emb_matrix (raw weights)
# ==============================================================================
print(f"\n" + "=" * 80)
print("CHECKPOINT 1: speech_emb_matrix (raw weights)")
print("=" * 80)
print(f"   Shape: {speech_emb_matrix.shape}")
print(f"   Range: [{speech_emb_matrix.min().item():.8f}, {speech_emb_matrix.max().item():.8f}]")
print(f"   Mean: {speech_emb_matrix.mean().item():.8f}")
print(f"   Std: {speech_emb_matrix.std().item():.8f}")
print(f"\n   First 10 values [0, 0, :10]:")
print(f"   {speech_emb_matrix[0, 0, :10].tolist()}")
print(f"\n   Sample from middle [0, 256, :10]:")
print(f"   {speech_emb_matrix[0, 256, :10].tolist()}")

# ==============================================================================
# CHECKPOINT 2: After embedding lookup
# ==============================================================================
# Hook into encoder to capture after embedding lookup
encoder = model.s3gen.flow.encoder
saved_after_embed = None

original_embed_forward = encoder.embed.forward

def hooked_embed_forward(xs, masks):
    """Hook to capture output after embedding lookup"""
    global saved_after_embed
    xs, pos_emb, masks = original_embed_forward(xs, masks)
    saved_after_embed = xs.detach().cpu().contiguous()
    return xs, pos_emb, masks

encoder.embed.forward = hooked_embed_forward

# ==============================================================================
# CHECKPOINT 3: After positional encoding
# ==============================================================================
# Hook to capture after pre_lookahead (which includes pos encoding)
saved_after_pos = None

original_pre_lookahead_forward = encoder.pre_lookahead_layer.forward

def hooked_pre_lookahead_forward(xs):
    """Hook to capture output after positional encoding"""
    global saved_after_pos
    # Pre-lookahead comes after pos encoding in the Python encoder
    # Actually, looking at the code, pos_emb is returned by embed
    # and used in forward_layers. Let me capture after forward_layers[0] instead
    output = original_pre_lookahead_forward(xs)
    saved_after_pos = output.detach().cpu().contiguous()
    return output

encoder.pre_lookahead_layer.forward = hooked_pre_lookahead_forward

print(f"\n" + "=" * 80)
print("Running encoder with hooks...")
print("=" * 80)

# Prepare inputs
token = generated_tokens.unsqueeze(0)  # [1, T]
token_len = torch.tensor([generated_tokens.shape[0]], dtype=torch.long, device=device)

# Create masks
T = token.size(1)
from chatterbox.models.s3gen.utils.mask import make_pad_mask
masks = ~make_pad_mask(token_len, T).unsqueeze(1)  # (B, 1, T)

# Expand speech_emb_matrix to include prompt
# speech_emb_matrix is [1, prompt_len, dim]
# We need to gather from token indices
# Actually, the embedding lookup happens inside the encoder
# Let me look at how the encoder is called in flow.inference

# Looking at the flow code, encoder is called with xs (speech features)
# But for our case, we're using token-based generation
# Let me just run the full inference and capture the hooks

with torch.no_grad():
    prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.long, device=device)
    prompt_feat_len = torch.tensor([prompt_feat.shape[1]], dtype=torch.long, device=device)

    # Run full inference to trigger hooks
    _ = model.s3gen.flow.inference(
        token=token,
        token_len=token_len,
        prompt_token=prompt_token,
        prompt_token_len=prompt_token_len,
        prompt_feat=prompt_feat,
        prompt_feat_len=prompt_feat_len,
        embedding=speech_emb_matrix,
        finalize=voice_data["t3.speaker_emb"].to(device),
        n_timesteps=10
    )

# ==============================================================================
# Report results
# ==============================================================================
print(f"\n" + "=" * 80)
print("CHECKPOINT 2: After embedding lookup")
print("=" * 80)
if saved_after_embed is not None:
    print(f"   Shape: {saved_after_embed.shape}")
    print(f"   Range: [{saved_after_embed.min().item():.8f}, {saved_after_embed.max().item():.8f}]")
    print(f"   Mean: {saved_after_embed.mean().item():.8f}")
    print(f"   Std: {saved_after_embed.std().item():.8f}")
    print(f"\n   First 10 values [0, 0, :10]:")
    print(f"   {saved_after_embed[0, 0, :10].tolist()}")
    print(f"\n   Position 500 [0, 500, :10]:")
    if saved_after_embed.shape[1] > 500:
        print(f"   {saved_after_embed[0, 500, :10].tolist()}")
else:
    print(f"   ❌ Failed to capture")

print(f"\n" + "=" * 80)
print("CHECKPOINT 3: After positional encoding (pre_lookahead)")
print("=" * 80)
if saved_after_pos is not None:
    print(f"   Shape: {saved_after_pos.shape}")
    print(f"   Range: [{saved_after_pos.min().item():.8f}, {saved_after_pos.max().item():.8f}]")
    print(f"   Mean: {saved_after_pos.mean().item():.8f}")
    print(f"   Std: {saved_after_pos.std().item():.8f}")
    print(f"\n   First 10 values [0, 0, :10]:")
    print(f"   {saved_after_pos[0, 0, :10].tolist()}")
    print(f"\n   Position 500 [0, 500, :10]:")
    if saved_after_pos.shape[1] > 500:
        print(f"   {saved_after_pos[0, 500, :10].tolist()}")
else:
    print(f"   ❌ Failed to capture")

# Save all checkpoints
print(f"\n" + "=" * 80)
print("Saving checkpoints...")
print("=" * 80)

save_file({
    "checkpoint1_speech_emb_matrix": speech_emb_matrix.cpu(),
    "checkpoint2_after_embed": saved_after_embed if saved_after_embed is not None else torch.zeros(1),
    "checkpoint3_after_pos": saved_after_pos if saved_after_pos is not None else torch.zeros(1),
}, str(OUTPUT_DIR / "python_encoder_first_layer_probe.safetensors"))

print(f"\n✅ Saved to: {OUTPUT_DIR / 'python_encoder_first_layer_probe.safetensors'}")
print(f"\n" + "=" * 80)
