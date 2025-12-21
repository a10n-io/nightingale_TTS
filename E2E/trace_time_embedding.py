#!/usr/bin/env python3
"""Trace time embedding through Python decoder at t=0."""

import torch
import numpy as np
from pathlib import Path
import sys
import math

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
sys.path.insert(0, str(PROJECT_ROOT / "python"))

def main():
    print("=" * 80)
    print("TRACING TIME EMBEDDING AT t=0")
    print("=" * 80)

    # Load Python model
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model_dir = PROJECT_ROOT / "models" / "chatterbox"
    model = ChatterboxMultilingualTTS.from_local(str(model_dir), device="cpu")
    estimator = model.s3gen.flow.decoder.estimator

    # Create t=0
    t = torch.zeros(1)

    # Trace the time embedding computation
    # In DitWrapper.forward():
    #   t = self.time_embeddings(t)  # SinusoidalPosEmb
    #   t = self.time_mlp(t)         # TimestepEmbedding

    # Step 1: Sinusoidal embedding using the model's SinusoidalPosEmb
    sin_emb = estimator.time_embeddings(t)
    print(f"\nSinusoidal embedding (t=0):")
    print(f"  Shape: {sin_emb.shape}")
    print(f"  Min/Max: [{sin_emb.min().item():.6f}, {sin_emb.max().item():.6f}]")
    print(f"  First 10: {sin_emb[0, :10].tolist()}")

    # Step 2: time_mlp (TimestepEmbedding)
    # TimestepEmbedding: linear_1 -> act (silu) -> linear_2
    time_mlp = estimator.time_mlp
    print(f"\nTime MLP structure: {time_mlp}")

    # Trace through time_mlp step by step
    x = sin_emb
    print(f"\nAfter sinusoidal embedding: shape={x.shape}")
    print(f"  Min/Max: [{x.min().item():.6f}, {x.max().item():.6f}]")
    print(f"  [0, :5]: {x[0, :5].tolist()}")

    # Linear 1
    x = time_mlp.linear_1(x)
    print(f"\nAfter Linear 1: shape={x.shape}")
    print(f"  Min/Max: [{x.min().item():.6f}, {x.max().item():.6f}]")
    print(f"  [0, :5]: {x[0, :5].tolist()}")

    # Check Linear 1 weights
    lin1 = time_mlp.linear_1
    print(f"  Linear 1 weight shape: {lin1.weight.shape}")
    print(f"  Linear 1 weight[0, :5]: {lin1.weight[0, :5].tolist()}")
    print(f"  Linear 1 bias[:5]: {lin1.bias[:5].tolist()}")

    # SiLU activation
    x = time_mlp.act(x)
    print(f"\nAfter SiLU: shape={x.shape}")
    print(f"  Min/Max: [{x.min().item():.6f}, {x.max().item():.6f}]")
    print(f"  [0, :5]: {x[0, :5].tolist()}")

    # Linear 2
    x = time_mlp.linear_2(x)
    print(f"\nAfter Linear 2 (final time_emb): shape={x.shape}")
    print(f"  Min/Max: [{x.min().item():.6f}, {x.max().item():.6f}]")
    print(f"  [0, :5]: {x[0, :5].tolist()}")

    # Check Linear 2 weights
    lin2 = time_mlp.linear_2
    print(f"  Linear 2 weight shape: {lin2.weight.shape}")
    print(f"  Linear 2 weight[0, :5]: {lin2.weight[0, :5].tolist()}")
    print(f"  Linear 2 bias[:5]: {lin2.bias[:5].tolist()}")

    # Save time embedding and intermediate values for Swift comparison
    np.save(PROJECT_ROOT / "E2E" / "python_sin_emb_t0.npy", sin_emb.detach().numpy())
    np.save(PROJECT_ROOT / "E2E" / "python_time_emb_t0.npy", x.detach().numpy())
    print(f"\nSaved sinusoidal embedding to python_sin_emb_t0.npy")
    print(f"Saved time embedding to python_time_emb_t0.npy")

if __name__ == "__main__":
    main()
