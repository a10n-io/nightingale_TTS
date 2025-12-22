#!/usr/bin/env python3
"""Manually trace first transformer block using state dict."""

import torch
import numpy as np
from pathlib import Path
import sys
from einops import rearrange

# Add chatterbox to path
sys.path.insert(0, str(Path.home() / "Library/Python/3.9/lib/python/site-packages"))
from chatterbox.models.s3gen.matcha.transformer import BasicTransformerBlock

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

print("="*80)
print("MANUAL TRANSFORMER TRACE")
print("="*80)

# Load state dict
model_path = PROJECT_ROOT / "models" / "chatterbox" / "s3gen.pt"
state_dict = torch.load(str(model_path), map_location='cpu')

def get_weight(key):
    return state_dict[f'flow.decoder.estimator.{key}']

# Load inputs
h_after_resnet = torch.from_numpy(np.load(ref_dir / "dec_trace_down0_resnet_out.npy"))  # [1, 256, 696]
mask = torch.from_numpy(np.load(ref_dir / "step7_cond_T.npy")[[0]])  # [1, 1, 696]

print(f"\n📥 Inputs:")
print(f"  h_after_resnet: {h_after_resnet.shape}, range=[{h_after_resnet.min():.4f}, {h_after_resnet.max():.4f}]")
print(f"  mask: {mask.shape}, sum={mask.sum().item()}")

# Prepare for transformer (expects [B, T, C])
h = rearrange(h_after_resnet, "b c t -> b t c")  # [1, 696, 256]
mask_2d = rearrange(mask, "b 1 t -> b t")  # [1, 696]

print(f"  Transformed h: {h.shape}")
print(f"  Transformed mask: {mask_2d.shape}")

# Create transformer block - use default activation (not snake)
# Looking at weights: Q/K/V are [512, 256], so inner_dim=512, with 8 heads that's 64 per head
print(f"\n🔧 Creating transformer block...")
tfmr = BasicTransformerBlock(
    dim=256,
    num_attention_heads=8,  # 512 / 64 = 8 heads
    attention_head_dim=64,
    dropout=0.05,
    activation_fn="gelu",  # Use default GELU instead of snake
)

# Load weights for down_blocks.0.1.0 (first transformer in first down block)
prefix = 'down_blocks.0.1.0'

print(f"Loading weights for {prefix}...")

# norm1
tfmr.norm1.weight.data = get_weight(f'{prefix}.norm1.weight')
tfmr.norm1.bias.data = get_weight(f'{prefix}.norm1.bias')

# attn1 (self-attention)
tfmr.attn1.to_q.weight.data = get_weight(f'{prefix}.attn1.to_q.weight')
tfmr.attn1.to_k.weight.data = get_weight(f'{prefix}.attn1.to_k.weight')
tfmr.attn1.to_v.weight.data = get_weight(f'{prefix}.attn1.to_v.weight')
tfmr.attn1.to_out[0].weight.data = get_weight(f'{prefix}.attn1.to_out.0.weight')
tfmr.attn1.to_out[0].bias.data = get_weight(f'{prefix}.attn1.to_out.0.bias')

# norm3 (FFN norm)
tfmr.norm3.weight.data = get_weight(f'{prefix}.norm3.weight')
tfmr.norm3.bias.data = get_weight(f'{prefix}.norm3.bias')

# ff (feedforward)
tfmr.ff.net[0].proj.weight.data = get_weight(f'{prefix}.ff.net.0.proj.weight')
tfmr.ff.net[0].proj.bias.data = get_weight(f'{prefix}.ff.net.0.proj.bias')
tfmr.ff.net[2].weight.data = get_weight(f'{prefix}.ff.net.2.weight')
tfmr.ff.net[2].bias.data = get_weight(f'{prefix}.ff.net.2.bias')

print("✅ Loaded weights")

# Run forward pass with tracing
tfmr.eval()
with torch.no_grad():
    print(f"\n🔄 Running transformer:")

    # Save input
    np.save(ref_dir / "tfmr_trace_input.npy", h.numpy())
    print(f"  Input: range=[{h.min():.4f}, {h.max():.4f}]")

    # norm1
    h_norm1 = tfmr.norm1(h)
    np.save(ref_dir / "tfmr_trace_after_norm1.npy", h_norm1.numpy())
    print(f"  After norm1: range=[{h_norm1.min():.4f}, {h_norm1.max():.4f}]")

    # Query, Key, Value
    q = tfmr.attn1.to_q(h_norm1)
    np.save(ref_dir / "tfmr_trace_query.npy", q.numpy())
    print(f"  Query: {q.shape}, range=[{q.min():.4f}, {q.max():.4f}]")

    k = tfmr.attn1.to_k(h_norm1)
    np.save(ref_dir / "tfmr_trace_key.npy", k.numpy())
    print(f"  Key: {k.shape}, range=[{k.min():.4f}, {k.max():.4f}]")

    v = tfmr.attn1.to_v(h_norm1)
    np.save(ref_dir / "tfmr_trace_value.npy", v.numpy())
    print(f"  Value: {v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")

    # Reshape for multi-head attention
    B, T, C = h_norm1.shape
    inner_dim = 512  # Q/K/V projection output size
    num_heads = 8
    head_dim = 64  # inner_dim / num_heads = 512 / 8 = 64

    q = q.view(B, T, num_heads, head_dim).transpose(1, 2)  # [B, H, T, D]
    k = k.view(B, T, num_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, num_heads, head_dim).transpose(1, 2)

    # Attention scores
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    np.save(ref_dir / "tfmr_trace_attn_scores_before_mask.npy", scores.numpy())
    print(f"  Scores (before mask): range=[{scores.min():.4f}, {scores.max():.4f}]")

    # Apply mask
    if mask_2d is not None:
        mask_expanded = mask_2d.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        scores = scores.masked_fill(mask_expanded == 0, -1e9)

    np.save(ref_dir / "tfmr_trace_attn_scores.npy", scores.numpy())
    print(f"  Scores (after mask): range=[{scores.min():.4f}, {scores.max():.4f}]")

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    np.save(ref_dir / "tfmr_trace_attn_weights.npy", attn_weights.numpy())
    print(f"  Attention weights: range=[{attn_weights.min():.4f}, {attn_weights.max():.4f}]")

    # Apply to values
    attn_out = torch.matmul(attn_weights, v)  # [B, H, T, D]

    # Reshape back
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, inner_dim)
    np.save(ref_dir / "tfmr_trace_attn_out_before_proj.npy", attn_out.numpy())
    print(f"  Attention out (before proj): range=[{attn_out.min():.4f}, {attn_out.max():.4f}]")

    # Output projection
    attn_out_proj = tfmr.attn1.to_out[0](attn_out)
    np.save(ref_dir / "tfmr_trace_attn_output.npy", attn_out_proj.numpy())
    print(f"  After out proj: range=[{attn_out_proj.min():.4f}, {attn_out_proj.max():.4f}]")

    # Residual 1
    h = h + attn_out_proj
    np.save(ref_dir / "tfmr_trace_after_res1.npy", h.numpy())
    print(f"  After residual 1: range=[{h.min():.4f}, {h.max():.4f}]")

    # norm3
    h_norm3 = tfmr.norm3(h)
    np.save(ref_dir / "tfmr_trace_after_norm3.npy", h_norm3.numpy())
    print(f"  After norm3: range=[{h_norm3.min():.4f}, {h_norm3.max():.4f}]")

    # Feedforward
    ff_out = tfmr.ff(h_norm3)
    np.save(ref_dir / "tfmr_trace_ff_output.npy", ff_out.numpy())
    print(f"  After FF: range=[{ff_out.min():.4f}, {ff_out.max():.4f}]")

    # Residual 2
    h_final = h + ff_out
    np.save(ref_dir / "tfmr_trace_final_output.npy", h_final.numpy())
    print(f"  Final output: range=[{h_final.min():.4f}, {h_final.max():.4f}]")

print("\n✅ Saved all transformer trace files")
print("="*80)
