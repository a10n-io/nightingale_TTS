#!/usr/bin/env python3
"""Trace first transformer block in detail to find Swift divergence."""

import torch
import numpy as np
from pathlib import Path
import sys
from einops import rearrange

# Add chatterbox to path
sys.path.insert(0, str(Path.home() / "Library/Python/3.9/lib/python/site-packages"))
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
MODEL_DIR = PROJECT_ROOT / "models" / "chatterbox"
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

print("="*80)
print("TRANSFORMER BLOCK DETAILED TRACE")
print("="*80)

# Load model
print("\n📦 Loading model...")
model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device="cpu")
print("✅ Model loaded")

# Load inputs - use the output from down block 0 resnet
h_after_resnet = torch.from_numpy(np.load(ref_dir / "dec_trace_down0_resnet_out.npy"))  # [1, 256, 696]
mask = torch.from_numpy(np.load(ref_dir / "step7_cond_T.npy")[[0]])  # [1, 1, 696]
t_emb = torch.from_numpy(np.load(ref_dir / "dec_trace_time_emb.npy"))  # [1, 1024]

print(f"\n📥 Inputs:")
print(f"  h_after_resnet: {h_after_resnet.shape}, range=[{h_after_resnet.min():.4f}, {h_after_resnet.max():.4f}]")
print(f"  mask: {mask.shape}, sum={mask.sum().item()}")
print(f"  t_emb: {t_emb.shape}, range=[{t_emb.min():.4f}, {t_emb.max():.4f}]")

# Get the first transformer block from down_blocks[0]
decoder = model.s3gen.flow.decoder
first_transformer = decoder.estimator.down_blocks[0][1][0]  # [resnet, transformers, downsample] -> transformers[0]

print(f"\n🔧 Transformer block: {type(first_transformer)}")

# Prepare input for transformer (expects [B, T, C])
h = rearrange(h_after_resnet, "b c t -> b t c")  # [1, 696, 256]
mask_2d = rearrange(mask, "b 1 t -> b t")  # [1, 696]

print(f"\n🔄 Running transformer forward pass:")
print(f"  Input h: {h.shape}, range=[{h.min():.4f}, {h.max():.4f}]")
print(f"  Input mask: {mask_2d.shape}")

# Manually step through the transformer to capture intermediates
model.eval()
with torch.no_grad():
    # Save input
    np.save(ref_dir / "tfmr_trace_input.npy", h.numpy())

    # norm1
    h_norm1 = first_transformer.norm1(h)
    print(f"  After norm1: range=[{h_norm1.min():.4f}, {h_norm1.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_after_norm1.npy", h_norm1.numpy())

    # Self-attention (attn1)
    attn_output = first_transformer.attn1(
        h_norm1,
        attention_mask=mask_2d,
    )
    print(f"  After attn1: range=[{attn_output.min():.4f}, {attn_output.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_attn_output.npy", attn_output.numpy())

    # Residual 1
    h = h + attn_output
    print(f"  After residual 1: range=[{h.min():.4f}, {h.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_after_res1.npy", h.numpy())

    # norm3 (FFN norm)
    h_norm3 = first_transformer.norm3(h)
    print(f"  After norm3: range=[{h_norm3.min():.4f}, {h_norm3.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_after_norm3.npy", h_norm3.numpy())

    # Feedforward
    ff_output = first_transformer.ff(h_norm3)
    print(f"  After FF: range=[{ff_output.min():.4f}, {ff_output.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_ff_output.npy", ff_output.numpy())

    # Residual 2
    h_final = h + ff_output
    print(f"  After residual 2 (final): range=[{h_final.min():.4f}, {h_final.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_final_output.npy", h_final.numpy())

    # Also trace attention internals
    print(f"\n📐 Tracing attention internals...")
    # Manually step through attention
    attn = first_transformer.attn1

    # Query, Key, Value projections
    q = attn.to_q(h_norm1)
    print(f"  Query: {q.shape}, range=[{q.min():.4f}, {q.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_query.npy", q.numpy())

    k = attn.to_k(h_norm1)
    print(f"  Key: {k.shape}, range=[{k.min():.4f}, {k.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_key.npy", k.numpy())

    v = attn.to_v(h_norm1)
    print(f"  Value: {v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_value.npy", v.numpy())

    # Reshape for multi-head attention
    B, T, C = h_norm1.shape
    num_heads = attn.heads
    head_dim = attn.dim_head

    q = q.view(B, T, num_heads, head_dim).transpose(1, 2)  # [B, H, T, D]
    k = k.view(B, T, num_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, num_heads, head_dim).transpose(1, 2)

    print(f"  Q reshaped: {q.shape}")
    print(f"  K reshaped: {k.shape}")
    print(f"  V reshaped: {v.shape}")

    # Attention scores
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    print(f"  Attention scores (before mask): {scores.shape}, range=[{scores.min():.4f}, {scores.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_attn_scores_before_mask.npy", scores.numpy())

    # Apply mask
    if mask_2d is not None:
        mask_expanded = mask_2d.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        scores = scores.masked_fill(mask_expanded == 0, -1e9)

    print(f"  Attention scores (after mask): range=[{scores.min():.4f}, {scores.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_attn_scores.npy", scores.numpy())

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    print(f"  Attention weights: range=[{attn_weights.min():.4f}, {attn_weights.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_attn_weights.npy", attn_weights.numpy())

    # Apply attention to values
    attn_out = torch.matmul(attn_weights, v)  # [B, H, T, D]
    print(f"  Attention output (before reshape): {attn_out.shape}, range=[{attn_out.min():.4f}, {attn_out.max():.4f}]")

    # Reshape back
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
    print(f"  Attention output (reshaped): {attn_out.shape}, range=[{attn_out.min():.4f}, {attn_out.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_attn_out_before_proj.npy", attn_out.numpy())

    # Output projection
    attn_out_proj = attn.to_out[0](attn_out)
    print(f"  After out projection: range=[{attn_out_proj.min():.4f}, {attn_out_proj.max():.4f}]")
    np.save(ref_dir / "tfmr_trace_attn_out_proj.npy", attn_out_proj.numpy())

print("\n✅ Saved all transformer trace files")
print("="*80)
