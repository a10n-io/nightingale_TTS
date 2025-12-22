#!/usr/bin/env python3
"""Simple Python decoder trace - call decoder and print key intermediate values."""
import torch
from pathlib import Path
import safetensors.torch as st

# Load the saved trace
trace_path = Path("test_audio/python_decoder_trace.safetensors")
trace = st.load_file(trace_path)

print("Loaded Python decoder trace:")
print(f"  decoder_output: range=[{trace['decoder_output'].min():.6f}, {trace['decoder_output'].max():.6f}], mean={trace['decoder_output'].mean():.6f}")

# Now run decoder with instrumentation
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

MODELS_DIR = Path("models/chatterbox")
VOICES_DIR = Path("baked_voices")
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("\nLoading model...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Get decoder
decoder = model.s3gen.flow.decoder.estimator

# Move trace tensors to device
noise = trace["noise"].to(device)
mu = trace["mu"].to(device)
spk_cond = trace["spk_cond"].to(device)
conds = trace["conds"].to(device)
mask = trace["mask"].to(device)
t = trace["t"].to(device)

print(f"\n=== Python Decoder Layer-by-Layer ===")

# Monkey-patch to add logging
original_forward = decoder.forward

def instrumented_forward(x, mask, mu, t, spks=None, cond=None):
    print(f"INPUT x: [{x.min():.6f}, {x.max():.6f}], mean={x.mean():.6f}")
    print(f"INPUT mu: [{mu.min():.6f}, {mu.max():.6f}], mean={mu.mean():.6f}")
    print(f"INPUT spks: [{spks.min():.6f}, {spks.max():.6f}], mean={spks.mean():.6f}")
    print(f"INPUT cond: [{cond.min():.6f}, {cond.max():.6f}], mean={cond.mean():.6f}")

    # Time embeddings
    t_emb = decoder.time_embeddings(t)
    t_emb = decoder.time_mlp(t_emb)
    print(f"time_emb: [{t_emb.min():.6f}, {t_emb.max():.6f}], mean={t_emb.mean():.6f}")

    # Pack x with mu, spks, cond
    from einops import pack, repeat
    x_packed = pack([x, mu], "b * t")[0]
    print(f"after pack x+mu: [{x_packed.min():.6f}, {x_packed.max():.6f}], mean={x_packed.mean():.6f}")

    if spks is not None:
        spks_exp = repeat(spks, "b c -> b c t", t=x_packed.shape[-1])
        x_packed = pack([x_packed, spks_exp], "b * t")[0]
        print(f"after pack +spks: [{x_packed.min():.6f}, {x_packed.max():.6f}], mean={x_packed.mean():.6f}")

    if cond is not None:
        x_packed = pack([x_packed, cond], "b * t")[0]
        print(f"after pack +cond: [{x_packed.min():.6f}, {x_packed.max():.6f}], mean={x_packed.mean():.6f}")

    # Call the real forward (but we won't log internal details)
    result = original_forward(x, mask, mu, t, spks, cond)
    print(f"FINAL OUTPUT: [{result.min():.6f}, {result.max():.6f}], mean={result.mean():.6f}")

    return result

decoder.forward = instrumented_forward

with torch.no_grad():
    output = decoder(
        x=noise,
        mask=mask,
        mu=mu,
        t=t,
        spks=spk_cond,
        cond=conds
    )

print(f"\n✅ Python decoder trace complete!")
print(f"Output: [{output.min():.6f}, {output.max():.6f}], mean={output.mean():.6f}")
