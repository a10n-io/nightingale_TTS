#!/usr/bin/env python3
"""Detailed trace of first transformer block."""
import torch
from pathlib import Path
import safetensors.torch as st
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Load model
MODELS_DIR = Path("models/chatterbox")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load trace
trace_path = Path("test_audio/python_decoder_trace.safetensors")
trace = st.load_file(trace_path)

noise = trace["noise"].to(device)
mu = trace["mu"].to(device)
spk_cond = trace["spk_cond"].to(device)
conds = trace["conds"].to(device)
mask = trace["mask"].to(device)
t = trace["t"].to(device)

decoder = model.s3gen.flow.decoder.estimator

with torch.no_grad():
    # Time embedding
    t_emb = decoder.time_embeddings(t)
    t_emb = decoder.time_mlp(t_emb)

    # Concatenate
    from einops import pack, repeat
    h = pack([noise, mu], "b * t")[0]
    spks_exp = repeat(spk_cond, "b c -> b c t", t=h.shape[-1])
    h = pack([h, spks_exp], "b * t")[0]
    h = pack([h, conds], "b * t")[0]
    print(f"concat: [{h.min():.6f}, {h.max():.6f}], shape={list(h.shape)}")

    # Down block 0: ResNet
    down = decoder.down_blocks[0]
    h = down[0](h, mask=mask, time_emb=t_emb)
    print(f"resnet output: [{h.min():.6f}, {h.max():.6f}], shape={list(h.shape)}")

    # Transpose for transformers
    h = h.transpose(1, 2)  # [B, C, T] -> [B, T, C]
    print(f"after transpose: [{h.min():.6f}, {h.max():.6f}], shape={list(h.shape)}")

    mask_t = mask.squeeze(1)  # [B, T]

    # First transformer block with detailed tracing
    tfmr = down[1][0]

    # Input
    input_h = h
    print(f"\nTransformer[0] INPUT: [{input_h.min():.6f}, {input_h.max():.6f}], mean={input_h.mean():.6f}")

    # Norm1
    norm_h = tfmr.norm1(input_h)
    print(f"After norm1: [{norm_h.min():.6f}, {norm_h.max():.6f}], mean={norm_h.mean():.6f}")

    # Attention
    attn_out = tfmr.attn1(norm_h, attention_mask=mask_t)
    print(f"After attention: [{attn_out.min():.6f}, {attn_out.max():.6f}], mean={attn_out.mean():.6f}")

    # Residual 1
    h_after_attn = attn_out + input_h
    print(f"After residual1 (attn+input): [{h_after_attn.min():.6f}, {h_after_attn.max():.6f}], mean={h_after_attn.mean():.6f}")

    # Norm3 (for FF)
    norm_h2 = tfmr.norm3(h_after_attn)
    print(f"After norm3: [{norm_h2.min():.6f}, {norm_h2.max():.6f}], mean={norm_h2.mean():.6f}")

    # FF
    ff_out = tfmr.ff(norm_h2)
    print(f"After FF: [{ff_out.min():.6f}, {ff_out.max():.6f}], mean={ff_out.mean():.6f}")

    # Residual 2
    final_h = ff_out + h_after_attn
    print(f"After residual2 (ff+h): [{final_h.min():.6f}, {final_h.max():.6f}], mean={final_h.mean():.6f}")

    # Transpose back
    final_h_t = final_h.transpose(1, 2)
    print(f"After transpose back: [{final_h_t.min():.6f}, {final_h_t.max():.6f}], shape={list(final_h_t.shape)}")

print("\n✅ Detailed transformer trace complete!")
