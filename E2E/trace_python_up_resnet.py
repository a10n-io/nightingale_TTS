#!/usr/bin/env python3
"""Trace Python up.resnet block layer-by-layer to find spatial variation bug."""
import torch
from pathlib import Path
import safetensors.torch as st
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import numpy as np

# Load model
MODELS_DIR = Path("models/chatterbox")
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading Chatterbox model...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load decoder trace to get inputs to up.resnet
trace_path = Path("test_audio/python_decoder_trace.safetensors")
trace = st.load_file(trace_path)

mu = trace["mu"].to(device)  # [1, 80, 696]
conds = trace["conds"].to(device)  # [1, 80, 696]
speaker_emb = trace["spk_cond"].to(device)  # [1, 80]

L_total = mu.shape[2]
L_pm = 500  # Prompt length

print(f"mu shape: {mu.shape}")
print(f"L_total: {L_total}, L_pm: {L_pm}")

# Get flow decoder
decoder = model.s3gen.flow.decoder.estimator

# ODE solver parameters
n_timesteps = 10
cfg_rate = 0.7
timesteps = torch.cos(torch.linspace(0, np.pi / 2, n_timesteps + 1, device=device)) ** 2

# Initialize with noise
xt = torch.randn(1, L_total, 80, device=device)  # [1, 696, 80]

print(f"\n{'='*80}")
print("Python up.resnet Layer-by-Layer Trace")
print(f"{'='*80}")

def check_spatial(h, label):
    """Check spatial variation: prompt vs generated region."""
    if h.shape[2] >= L_pm:
        prompt = h[:, :, :L_pm]
        generated = h[:, :, L_pm:]
        prompt_mean = prompt.mean().item()
        generated_mean = generated.mean().item()
        bias = generated_mean - prompt_mean
        print(f"   {label}: prompt={prompt_mean:.4f}, generated={generated_mean:.4f}, bias={bias:.4f}")
        return bias
    return None

# Run ONE ODE step to get to up.resnet block
t_curr = timesteps[0]
t_next = timesteps[1]
dt = t_next - t_curr

# Prepare inputs
t_batch = t_curr.unsqueeze(0).expand(2)
x_transposed = xt.transpose(1, 2)  # [1, 80, L_total]

# CFG batch
x_batch = torch.cat([x_transposed, x_transposed], dim=0)
mu_batch = torch.cat([mu, mu], dim=0)
conds_batch = torch.cat([conds, torch.zeros_like(conds)], dim=0)
speaker_batch = speaker_emb.expand(2, -1)
mask_batch = torch.ones(2, 1, L_total, device=device)

print("\nRunning decoder forward pass to up.resnet...")

# We need to manually trace through decoder.forward() to capture intermediate values
# From python/chatterbox/src/chatterbox/models/s3gen/decoder.py

from einops import pack, rearrange

# 1. Time embedding
t = decoder.time_embeddings(t_batch).to(t_batch.dtype)  # [2, 320]
t = decoder.time_mlp(t)  # [2, time_emb_dim=1024]
print(f"Time embedding shape: {t.shape}")

# 2. Concatenate inputs: [x, mu]
h = pack([x_batch, mu_batch], "b * t")[0]  # [2, 160, L_total]

# Add speaker embedding
if speaker_batch is not None:
    from einops import repeat
    spks_expanded = repeat(speaker_batch, "b c -> b c t", t=h.shape[-1])  # [2, 80, L_total]
    h = pack([h, spks_expanded], "b * t")[0]  # [2, 240, L_total]

# Add conditioning
if conds_batch is not None:
    h = pack([h, conds_batch], "b * t")[0]  # [2, 320, L_total]

check_spatial(h, "01_concat")

# 3. Down blocks (storing skip connections)
skips = []
masks = [mask_batch]
for i, (resnet, transformer_blocks, downsample) in enumerate(decoder.down_blocks):
    mask_down = masks[-1]
    h = resnet(h, mask_down, t)
    check_spatial(h, f"down_{i}_resnet")

    # Transformer blocks need rearrangement
    h = rearrange(h, "b c t -> b t c").contiguous()
    from chatterbox.models.s3gen.decoder import add_optional_chunk_mask, mask_to_bias
    attn_mask = add_optional_chunk_mask(h, mask_down.bool(), False, False, 0, decoder.static_chunk_size, -1)
    attn_mask = mask_to_bias(attn_mask == 1, h.dtype)
    for tfmr_block in transformer_blocks:
        h = tfmr_block(hidden_states=h, attention_mask=attn_mask, timestep=t)
    h = rearrange(h, "b t c -> b c t").contiguous()
    check_spatial(h, f"down_{i}_tfmrs")

    # Store skip
    skips.append(h)

    # Downsample
    h = downsample(h * mask_down)
    masks.append(mask_down[:, :, ::2])
    check_spatial(h, f"down_{i}_downsample")

masks = masks[:-1]
mask_mid = masks[-1]

# 4. Mid blocks
for i, (resnet, transformer_blocks) in enumerate(decoder.mid_blocks):
    h = resnet(h, mask_mid, t)
    h = rearrange(h, "b c t -> b t c").contiguous()
    attn_mask = add_optional_chunk_mask(h, mask_mid.bool(), False, False, 0, decoder.static_chunk_size, -1)
    attn_mask = mask_to_bias(attn_mask == 1, h.dtype)
    for tfmr_block in transformer_blocks:
        h = tfmr_block(hidden_states=h, attention_mask=attn_mask, timestep=t)
    h = rearrange(h, "b t c -> b c t").contiguous()
    if i % 4 == 0:
        check_spatial(h, f"mid_{i:02d}")

# 5. Up blocks - THIS IS WHERE THE BUG OCCURS
print(f"\n{'='*80}")
print("UP BLOCKS - DETAILED TRACE")
print(f"{'='*80}")

for i, (resnet, transformer_blocks, upsample) in enumerate(decoder.up_blocks):
    print(f"\n--- UP BLOCK {i} ---")
    mask_up = masks.pop()

    # Skip connection concatenation
    skip = skips.pop()

    # Check h and skip separately BEFORE concatenation
    h_trunc = h[:, :, :skip.shape[-1]]
    check_spatial(h_trunc, f"up_{i}_h_before_concat")
    check_spatial(skip, f"up_{i}_skip_before_concat")

    h = pack([h_trunc, skip], "b * t")[0]
    check_spatial(h, f"up_{i}_skip_concat")

    print(f"  Input to resnet: {h.shape}")

    # Manually trace through CausalResNetBlock1D
    # From python/chatterbox/src/chatterbox/models/s3gen/decoder.py
    print(f"  Resnet block:")

    h_before = h.clone()

    # Block 1 (CausalBlock1D with conv + layernorm + mish)
    # resnet.block1(h, mask_up) applies block1.block to h*mask then masks output
    h_res = resnet.block1.block(h * mask_up)  # Sequential in CausalBlock1D
    h_res = h_res * mask_up
    check_spatial(h_res, f"    block1 (conv+ln+mish)")

    # Add time embedding (simple addition, not scale/shift!)
    # From matcha/decoder.py line 58: h += self.mlp(time_emb).unsqueeze(-1)
    t_emb_projected = resnet.mlp(t).unsqueeze(-1)  # [2, dim_out, 1]
    h_res = h_res + t_emb_projected
    check_spatial(h_res, f"    +time_emb")

    # Block 2 (second CausalBlock1D)
    h_res = resnet.block2.block(h_res * mask_up)
    h_res = h_res * mask_up
    check_spatial(h_res, f"    block2 (conv+ln+mish)")

    # Residual connection
    if resnet.res_conv is not None:
        h_skip = resnet.res_conv(h_before * mask_up)
        check_spatial(h_skip, f"    res_conv")
    else:
        h_skip = h_before
        check_spatial(h_skip, f"    res_skip (identity)")

    h = h_res + h_skip
    check_spatial(h, f"    residual_sum")

    bias_after_resnet = check_spatial(h, f"  up_{i}_after_resnet")
    print(f"  *** SPATIAL BIAS AFTER RESNET: {bias_after_resnet:.4f} ***")

    # Transformer blocks
    h = rearrange(h, "b c t -> b t c").contiguous()
    attn_mask = add_optional_chunk_mask(h, mask_up.bool(), False, False, 0, decoder.static_chunk_size, -1)
    attn_mask = mask_to_bias(attn_mask == 1, h.dtype)
    print(f"  Transformer blocks ({len(transformer_blocks)} blocks):")
    for j, tfmr_block in enumerate(transformer_blocks):
        h = tfmr_block(hidden_states=h, attention_mask=attn_mask, timestep=t)
        if j == 0:
            h_temp = rearrange(h, "b t c -> b c t").contiguous()
            check_spatial(h_temp, f"    tfmr_{j}")
    h = rearrange(h, "b t c -> b c t").contiguous()
    check_spatial(h, f"  up_{i}_after_tfmrs")

    # Upsample
    h = upsample(h * mask_up)
    check_spatial(h, f"  up_{i}_upsample")

print(f"\n{'='*80}")
print("FINAL LAYERS")
print(f"{'='*80}")

# 6. Final block (using last mask_up from loop)
h = decoder.final_block(h, mask_up)
check_spatial(h, "final_block")

# 7. Final projection
h = decoder.final_proj(h * mask_up)
check_spatial(h, "final_proj")

# Apply final mask
output = h * mask_batch
check_spatial(output, "final_output")

print(f"\n{'='*80}")
print("TRACE COMPLETE")
print(f"{'='*80}")
