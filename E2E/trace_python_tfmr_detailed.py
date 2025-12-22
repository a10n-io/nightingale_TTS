#!/usr/bin/env python3
"""Trace Python transformer forward pass in detail."""
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

# Generate test inputs directly
L_total = 696
L_pm = 500

def check_spatial(tensor, label):
    """Check spatial bias in [B, C, T] or [B, T, C] format."""
    if tensor.ndim == 3:
        # Determine if [B, C, T] or [B, T, C]
        if tensor.shape[1] == 256 or tensor.shape[1] == 80 or tensor.shape[1] == 320 or tensor.shape[1] == 160 or tensor.shape[1] == 240:  # likely [B, C, T]
            prompt = tensor[0, :, :L_pm]
            generated = tensor[0, :, L_pm:]
        else:  # likely [B, T, C]
            prompt = tensor[0, :L_pm, :]
            generated = tensor[0, L_pm:, :]

        p_mean = prompt.mean().item()
        g_mean = generated.mean().item()
        bias = g_mean - p_mean
        print(f"{label}: prompt={p_mean:.4f}, generated={g_mean:.4f}, bias={bias:.4f}")
        return bias
    return None

# Create synthetic inputs similar to decoder
mu = torch.randn(1, 80, L_total, device=device)
# conds: first 500 frames have values, rest are zeros
conds = torch.randn(1, 80, L_pm, device=device)
conds = torch.cat([conds, torch.zeros(1, 80, L_total - L_pm, device=device)], dim=2)
speaker_emb = torch.randn(1, 80, device=device)

decoder = model.s3gen.flow.decoder.estimator

# ODE parameters
n_timesteps = 10
timesteps = torch.cos(torch.linspace(0, np.pi / 2, n_timesteps + 1, device=device)) ** 2
xt = torch.randn(1, L_total, 80, device=device)

# Run one ODE step to get to transformers
t_curr = timesteps[0]
t_batch = t_curr.unsqueeze(0).expand(2)

x_transposed = xt.transpose(1, 2)
x_batch = torch.cat([x_transposed, x_transposed], dim=0)
mu_batch = torch.cat([mu, mu], dim=0)
conds_batch = torch.cat([conds, torch.zeros_like(conds)], dim=0)
speaker_batch = speaker_emb.expand(2, -1)
mask_batch = torch.ones(2, 1, L_total, device=device)

from einops import pack, rearrange
from chatterbox.models.s3gen.decoder import add_optional_chunk_mask, mask_to_bias

# 1. Time embedding
t = decoder.time_embeddings(t_batch).to(t_batch.dtype)
t = decoder.time_mlp(t)

# 2. Concatenate inputs
print("\nConcatenation components:")
check_spatial(x_batch, "  x_batch")
check_spatial(mu_batch, "  mu_batch")

h = pack([x_batch, mu_batch], "b * t")[0]
check_spatial(h, "  after [x,mu]")

if speaker_batch is not None:
    from einops import repeat
    spks_expanded = repeat(speaker_batch, "b c -> b c t", t=h.shape[-1])
    check_spatial(spks_expanded, "  spks_expanded")
    h = pack([h, spks_expanded], "b * t")[0]
    check_spatial(h, "  after [x,mu,spks]")

if conds_batch is not None:
    check_spatial(conds_batch, "  conds_batch")
    h = pack([h, conds_batch], "b * t")[0]
    check_spatial(h, "  after [x,mu,spks,conds]")

# 3. Down blocks - run through resnet to get to transformers
resnet, transformer_blocks, downsample = decoder.down_blocks[0]
mask_down = mask_batch

print(f"{'='*80}")
print("Python Transformer Forward Pass - Detailed Trace")
print(f"{'='*80}")

# Check h BEFORE resnet
check_spatial(h, "01_concat (input to resnet)")

# Run through resnet
h = resnet(h, mask_down, t)

print(f"\nAfter resnet, h shape: {h.shape}")  # [2, 256, 696] in [B, C, T] format

check_spatial(h, "02_after_resnet (B,C,T)")

# Transpose to [B, T, C] for transformers
h = rearrange(h, "b c t -> b t c").contiguous()
print(f"\nAfter transpose, h shape: {h.shape}")  # [2, 696, 256] in [B, T, C] format
check_spatial(h, "03_after_transpose (B,T,C)")

# Create attention mask
attn_mask = add_optional_chunk_mask(h, mask_down.bool(), False, False, 0, decoder.static_chunk_size, -1)
attn_mask = mask_to_bias(attn_mask == 1, h.dtype)
print(f"attn_mask shape: {attn_mask.shape}")

# Trace through FIRST transformer in detail
tfmr = transformer_blocks[0]
print(f"\n{'='*80}")
print("FIRST TRANSFORMER BLOCK - Layer by layer")
print(f"{'='*80}")

with torch.no_grad():
    # Input
    hidden_states = h
    check_spatial(hidden_states, "tfmr input")

    # 1. norm1
    norm_hidden_states = tfmr.norm1(hidden_states)
    check_spatial(norm_hidden_states, "after norm1")

    # 2. attn1
    attn_output = tfmr.attn1(
        norm_hidden_states,
        attention_mask=attn_mask,
    )
    check_spatial(attn_output, "after attn1")

    # 3. residual 1
    hidden_states = attn_output + hidden_states
    check_spatial(hidden_states, "after residual1")

    # 4. norm3 (feedforward norm)
    norm_hidden_states = tfmr.norm3(hidden_states)
    check_spatial(norm_hidden_states, "after norm3")

    # 5. ff
    ff_output = tfmr.ff(norm_hidden_states)
    check_spatial(ff_output, "after ff")

    # 6. residual 2
    hidden_states = ff_output + hidden_states
    check_spatial(hidden_states, "after residual2 (final)")

    # Run through all 4 transformers
    print(f"\n{'='*80}")
    print("ALL 4 TRANSFORMERS")
    print(f"{'='*80}")
    h_input = h
    check_spatial(h_input, "Input to all tfmrs")

    for i, tfmr_block in enumerate(transformer_blocks):
        h_input = tfmr_block(hidden_states=h_input, attention_mask=attn_mask, timestep=t)
        check_spatial(h_input, f"After tfmr[{i}]")

    # Transpose back
    h_output = rearrange(h_input, "b t c -> b c t").contiguous()
    check_spatial(h_output, "After transpose back (B,C,T)")

print(f"\n{'='*80}")
print("COMPLETE")
print(f"{'='*80}")
