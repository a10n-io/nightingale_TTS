#!/usr/bin/env python3
"""Trace first transformer block in decoder to find divergence."""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path.home() / "Library/Python/3.9/lib/python/site-packages"))
from chatterbox.models.s3gen.matcha.decoder import ResnetBlock1D
from chatterbox.models.s3gen.matcha.transformer import BasicTransformerBlock

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

print("="*80)
print("TRACE FIRST TRANSFORMER BLOCK")
print("="*80)

# Load model weights
model_path = PROJECT_ROOT / "models" / "chatterbox" / "s3gen.pt"
state_dict = torch.load(str(model_path), map_location='cpu')

def get_weight(key):
    return state_dict[f'flow.decoder.estimator.{key}']

# Load Python reference for ResNet output (input to transformer)
# This is the output of down_blocks[0].resnet
h_after_resnet = torch.from_numpy(np.load(ref_dir / "dec_trace_down0_resnet_out.npy"))
mask = torch.from_numpy(np.load(ref_dir / "step7_cond_T.npy")[[0]])  # [1, 1, 696]

print(f"\n📥 Input to transformer:")
print(f"  h_after_resnet: {h_after_resnet.shape}, range=[{h_after_resnet.min():.4f}, {h_after_resnet.max():.4f}]")
print(f"  mask: {mask.shape}, sum={mask.sum().item()}")

# Create first transformer block
# down_blocks.0.1.0 = first down block, transformers list (index 1), first transformer (index 0)
tfmr = BasicTransformerBlock(
    dim=256,
    num_attention_heads=4,
    attention_head_dim=64,
    dropout=0.05,
    activation_fn="snake"
)

# Load weights for down_blocks.0.1.0 (first transformer in first down block)
prefix = 'down_blocks.0.1.0'

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
# ff.net.0.proj is a Linear layer
# ff.net.2 is another Linear layer
tfmr.ff.net[0].proj.weight.data = get_weight(f'{prefix}.ff.net.0.proj.weight')
tfmr.ff.net[0].proj.bias.data = get_weight(f'{prefix}.ff.net.0.proj.bias')
tfmr.ff.net[2].weight.data = get_weight(f'{prefix}.ff.net.2.weight')
tfmr.ff.net[2].bias.data = get_weight(f'{prefix}.ff.net.2.bias')

print("✅ Loaded transformer weights")

# Run transformer forward pass with intermediate tracing
tfmr.eval()
with torch.no_grad():
    # Transformer expects [B, T, C] but h is [B, C, T]
    h = h_after_resnet.transpose(1, 2)  # [1, 696, 256]
    mask_attn = mask.squeeze(1)  # [1, 696]

    print(f"\n🔄 Transformer forward pass:")
    print(f"  Input h: {h.shape}, range=[{h.min():.4f}, {h.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_input.npy", h.numpy())

    # norm1
    h_norm1 = tfmr.norm1(h)
    print(f"  After norm1: range=[{h_norm1.min():.4f}, {h_norm1.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_after_norm1.npy", h_norm1.detach().numpy())

    # Attention
    # to_q
    q = tfmr.attn1.to_q(h_norm1)
    print(f"  Query: {q.shape}, range=[{q.min():.4f}, {q.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_query.npy", q.detach().numpy())

    # to_k
    k = tfmr.attn1.to_k(h_norm1)
    print(f"  Key: {k.shape}, range=[{k.min():.4f}, {k.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_key.npy", k.detach().numpy())

    # to_v
    v = tfmr.attn1.to_v(h_norm1)
    print(f"  Value: {v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_value.npy", v.detach().numpy())

    # Reshape for multi-head attention
    B, T, C = h_norm1.shape
    num_heads = 4
    head_dim = 64

    q = q.view(B, T, num_heads, head_dim).transpose(1, 2)  # [B, H, T, D]
    k = k.view(B, T, num_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, num_heads, head_dim).transpose(1, 2)

    print(f"  Q reshaped: {q.shape}")
    print(f"  K reshaped: {k.shape}")
    print(f"  V reshaped: {v.shape}")

    # Attention scores
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    print(f"  Attention scores: {scores.shape}, range=[{scores.min():.4f}, {scores.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_attn_scores.npy", scores.detach().numpy())

    # Apply mask if needed
    if mask_attn is not None:
        # Expand mask for attention
        mask_expanded = mask_attn.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        scores = scores.masked_fill(mask_expanded == 0, -1e9)

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    print(f"  Attention weights: range=[{attn_weights.min():.4f}, {attn_weights.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_attn_weights.npy", attn_weights.detach().numpy())

    # Apply attention
    attn_output = torch.matmul(attn_weights, v)  # [B, H, T, D]
    print(f"  Attention output: {attn_output.shape}, range=[{attn_output.min():.4f}, {attn_output.max():.4f}]")

    # Reshape back
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
    print(f"  Attention output reshaped: {attn_output.shape}, range=[{attn_output.min():.4f}, {attn_output.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_attn_output.npy", attn_output.detach().numpy())

    # Output projection
    attn_output = tfmr.attn1.to_out[0](attn_output)
    print(f"  After out_proj: range=[{attn_output.min():.4f}, {attn_output.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_after_out_proj.npy", attn_output.detach().numpy())

    # Residual
    h = h + attn_output
    print(f"  After residual 1: range=[{h.min():.4f}, {h.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_after_res1.npy", h.detach().numpy())

    # norm3 (FFN norm)
    h_norm2 = tfmr.norm3(h)
    print(f"  After norm3: range=[{h_norm2.min():.4f}, {h_norm2.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_after_norm3.npy", h_norm2.detach().numpy())

    # Feedforward
    h_ff = tfmr.ff(h_norm2)
    print(f"  After FF: range=[{h_ff.min():.4f}, {h_ff.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_after_ff.npy", h_ff.detach().numpy())

    # Residual 2
    h = h + h_ff
    print(f"  After residual 2 (final): range=[{h.min():.4f}, {h.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_final_output.npy", h.detach().numpy())

print("\n✅ Saved all transformer intermediate values")
print("="*80)
