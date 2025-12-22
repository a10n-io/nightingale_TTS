#!/usr/bin/env python3
"""Bisect decoder to find where Swift diverges from Python."""
import torch
from pathlib import Path
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import numpy as np
from einops import pack, rearrange

# Load model
MODELS_DIR = Path("models/chatterbox")
device = "cpu"  # Force CPU for reproducibility

print("Loading Chatterbox model...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Generate EXACT same inputs as Swift test
L_total = 696
L_pm = 500

# Helper function for bisection debugging - print tensor statistics
def debug_stats(x, name):
    mean = x.mean().item()
    std = x.std().item()
    min_val = x.min().item()
    max_val = x.max().item()
    print(f"🔍 [{name}] Shape: {tuple(x.shape)} | Mean: {mean:.4f} | Std: {std:.4f} | Range: [{min_val:.4f}, {max_val:.4f}]")

# === DETERMINISTIC SYNTHESIS (matches Swift exactly) ===
print("--- 🧪 GENERATING DETERMINISTIC INPUTS ---")

def get_deterministic_tensor(shape):
    """Generates identical data to Swift based on index."""
    count = np.prod(shape)
    flat = np.arange(count, dtype=np.float32)
    data = np.sin(flat / 10.0)  # Non-linear but deterministic
    return torch.from_numpy(data).reshape(shape).to(device)

# Create deterministic inputs
mu = get_deterministic_tensor((1, 80, L_total))
conds = get_deterministic_tensor((1, 80, L_pm))
conds = torch.cat([conds, torch.zeros(1, 80, L_total - L_pm, device=device)], dim=2)
speaker_emb = get_deterministic_tensor((1, 80))

decoder = model.s3gen.flow.decoder.estimator

# ODE parameters
n_timesteps = 10
timesteps = torch.cos(torch.linspace(0, np.pi / 2, n_timesteps + 1, device=device)) ** 2
xt = get_deterministic_tensor((1, L_total, 80))

# Run one ODE step
t_curr = timesteps[0]
t_batch = t_curr.unsqueeze(0).expand(2)

x_transposed = xt.transpose(1, 2)
x_batch = torch.cat([x_transposed, x_transposed], dim=0)
mu_batch = torch.cat([mu, mu], dim=0)
conds_batch = torch.cat([conds, torch.zeros_like(conds)], dim=0)
speaker_batch = speaker_emb.expand(2, -1)
mask_batch = torch.ones(2, 1, L_total, device=device)

print("\n" + "="*80)
print("PYTHON DECODER BISECTION")
print("="*80)

# We need to manually trace through decoder.forward() to insert checkpoints
from chatterbox.models.s3gen.decoder import add_optional_chunk_mask, mask_to_bias

# 1. Time embedding
t = decoder.time_embeddings(t_batch).to(t_batch.dtype)
t = decoder.time_mlp(t)

# 2. Concatenate inputs
h = pack([x_batch, mu_batch], "b * t")[0]
if speaker_batch is not None:
    from einops import repeat
    spks_expanded = repeat(speaker_batch, "b c -> b c t", t=h.shape[-1])
    h = pack([h, spks_expanded], "b * t")[0]
if conds_batch is not None:
    h = pack([h, conds_batch], "b * t")[0]

# ===== MICRO-BISECTION CHECKPOINT A: After input concatenation =====
debug_stats(h, "CHECKPOINT_A_input_concat")

# 3. Down blocks
skips = []
masks = [mask_batch]
for i, (resnet, transformer_blocks, downsample) in enumerate(decoder.down_blocks):
    mask_down = masks[-1]

    # Manually trace through first ResNet for internal checkpoints
    if i == 0:
        # Trace through CausalResNetBlock manually
        h_resnet = resnet.block1(h, mask_down)

        # ===== DEBUG: After block1 =====
        debug_stats(h_resnet, "DEBUG_after_block1")

        # Time embedding injection (mlp is Sequential(Mish(), Linear()))
        t_emb = resnet.mlp(t)  # Already includes Mish inside mlp
        t_emb = t_emb.unsqueeze(-1)

        # ===== DEBUG: Time embedding =====
        debug_stats(t_emb, "DEBUG_tEmb")

        h_resnet = h_resnet + t_emb

        # ===== MICRO-BISECTION CHECKPOINT B1: After Time Embedding =====
        debug_stats(h_resnet, "CHECKPOINT_B1_Norm1_Temb")

        # Block2
        h_resnet = resnet.block2(h_resnet, mask_down)

        # ===== MICRO-BISECTION CHECKPOINT B2: After Branch Output =====
        debug_stats(h_resnet, "CHECKPOINT_B2_Branch_Output")

        # Skip connection (res_conv)
        skip_resnet = resnet.res_conv(h * mask_down)

        # ===== MICRO-BISECTION CHECKPOINT B3: Skip Connection =====
        debug_stats(skip_resnet, "CHECKPOINT_B3_Skip")

        # Final sum
        h = h_resnet + skip_resnet

        # ===== MICRO-BISECTION CHECKPOINT B4: Final Sum =====
        debug_stats(h, "CHECKPOINT_B4_Final_Sum")
    else:
        h = resnet(h, mask_down, t)

    # ===== MICRO-BISECTION CHECKPOINT B: After first ResNet =====
    if i == 0:
        debug_stats(h, "CHECKPOINT_B_after_first_resnet")

    # Transformer blocks
    h = rearrange(h, "b c t -> b t c").contiguous()
    attn_mask = add_optional_chunk_mask(h, mask_down.bool(), False, False, 0, decoder.static_chunk_size, -1)
    attn_mask = mask_to_bias(attn_mask == 1, h.dtype)
    for tfmr_block in transformer_blocks:
        h = tfmr_block(hidden_states=h, attention_mask=attn_mask, timestep=t)
    h = rearrange(h, "b t c -> b c t").contiguous()

    # ===== MICRO-BISECTION CHECKPOINT C: After first transformers =====
    if i == 0:
        debug_stats(h, "CHECKPOINT_C_after_first_transformers")

    # Store skip
    skips.append(h)

    # Downsample
    h = downsample(h * mask_down)
    masks.append(mask_down[:, :, ::2])

masks = masks[:-1]
mask_mid = masks[-1]

# ===== BISECTION CHECKPOINT 1: After down_blocks =====
debug_stats(h, "CHECKPOINT_1_after_down_blocks")

# 4. Mid blocks
for i, (resnet, transformer_blocks) in enumerate(decoder.mid_blocks):
    h = resnet(h, mask_mid, t)
    h = rearrange(h, "b c t -> b t c").contiguous()
    attn_mask = add_optional_chunk_mask(h, mask_mid.bool(), False, False, 0, decoder.static_chunk_size, -1)
    attn_mask = mask_to_bias(attn_mask == 1, h.dtype)
    for tfmr_block in transformer_blocks:
        h = tfmr_block(hidden_states=h, attention_mask=attn_mask, timestep=t)
    h = rearrange(h, "b t c -> b c t").contiguous()

# ===== BISECTION CHECKPOINT 2: After mid_blocks =====
debug_stats(h, "CHECKPOINT_2_after_mid_blocks")

# 5. Up blocks
for i, (resnet, transformer_blocks, upsample) in enumerate(decoder.up_blocks):
    mask_up = masks.pop()
    skip = skips.pop()

    # Skip connection concatenation
    h_trunc = h[:, :, :skip.shape[-1]]
    h = pack([h_trunc, skip], "b * t")[0]

    # Resnet
    h = resnet(h, mask_up, t)

    # Transformers
    h = rearrange(h, "b c t -> b t c").contiguous()
    attn_mask = add_optional_chunk_mask(h, mask_up.bool(), False, False, 0, decoder.static_chunk_size, -1)
    attn_mask = mask_to_bias(attn_mask == 1, h.dtype)
    for tfmr_block in transformer_blocks:
        h = tfmr_block(hidden_states=h, attention_mask=attn_mask, timestep=t)
    h = rearrange(h, "b t c -> b c t").contiguous()

    # Upsample
    h = upsample(h * mask_up)

# ===== BISECTION CHECKPOINT 3: After up_blocks =====
debug_stats(h, "CHECKPOINT_3_after_up_blocks")

# 6. Final block
h = decoder.final_block(h, mask_up)

# 7. Final projection
output = decoder.final_proj(h * mask_up)

# 8. Final output
output = output * mask_batch

print("\n" + "="*80)
print("Final Output:")
debug_stats(output, "FINAL_OUTPUT")
print("="*80)

print("\n✅ Python bisection complete")
