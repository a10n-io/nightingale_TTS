#!/usr/bin/env python3
"""Trace first decoder layer to find divergence."""
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
    print(f"time_emb: [{t_emb.min():.6f}, {t_emb.max():.6f}], mean={t_emb.mean():.6f}")

    # Concatenate
    from einops import pack, repeat
    h = pack([noise, mu], "b * t")[0]
    spks_exp = repeat(spk_cond, "b c -> b c t", t=h.shape[-1])
    h = pack([h, spks_exp], "b * t")[0]
    h = pack([h, conds], "b * t")[0]
    print(f"after concat: [{h.min():.6f}, {h.max():.6f}], mean={h.mean():.6f}")

    # Down block 0: ResNet
    down = decoder.down_blocks[0]
    h_after_resnet = down[0](h, mask=mask, time_emb=t_emb)
    print(f"down[0].resnet: [{h_after_resnet.min():.6f}, {h_after_resnet.max():.6f}], mean={h_after_resnet.mean():.6f}")

    # Down block 0: Transformers
    h_t = h_after_resnet.transpose(1, 2)  # [B, T, C]
    mask_t = mask.squeeze(1)  # [B, T]

    for i, tfmr in enumerate(down[1]):
        output = tfmr(hidden_states=h_t, attention_mask=mask_t)
        h_t = output.sample if hasattr(output, 'sample') else output
        if i == 0:
            print(f"down[0].tfmr[{i}]: [{h_t.min():.6f}, {h_t.max():.6f}], mean={h_t.mean():.6f}")

    h_after_tfmrs = h_t.transpose(1, 2)  # [B, C, T]
    print(f"down[0].transformers: [{h_after_tfmrs.min():.6f}, {h_after_tfmrs.max():.6f}], mean={h_after_tfmrs.mean():.6f}")

print("\n✅ First layer trace complete!")
