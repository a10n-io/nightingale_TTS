#!/usr/bin/env python3
"""Trace attention internals in first transformer."""

import torch
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
REF_DIR = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

def main():
    print("=" * 80)
    print("TRACING ATTENTION INTERNALS - FIRST TRANSFORMER")
    print("=" * 80)

    # Load model
    sys.path.insert(0, str(PROJECT_ROOT / "python"))
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model_dir = PROJECT_ROOT / "models" / "chatterbox"
    model = ChatterboxMultilingualTTS.from_local(str(model_dir), device="cpu")
    s3 = model.s3gen

    # Load Step 7a inputs
    mu_T = torch.from_numpy(np.load(REF_DIR / "step7a_mu_T.npy"))
    cond_T = torch.from_numpy(np.load(REF_DIR / "step7a_cond_T.npy"))
    initial_noise = torch.from_numpy(np.load(REF_DIR / "step7a_initial_noise.npy"))
    mask_T = torch.from_numpy(np.load(REF_DIR / "step7a_mask_T.npy"))
    spk_emb = torch.from_numpy(np.load(REF_DIR / "step5_spk_emb.npy"))
    t0 = torch.tensor([0.0])

    estimator = s3.flow.decoder.estimator
    outputs = {}
    hooks = []

    # Hook first transformer's attention component (attn1)
    first_tfmr = estimator.down_blocks[0][1][0]
    attn = first_tfmr.attn1

    # Hook Q, K, V projections
    def to_q_hook(module, input, output):
        outputs["Q"] = output.detach().clone()
    hooks.append(attn.to_q.register_forward_hook(to_q_hook))

    def to_k_hook(module, input, output):
        outputs["K"] = output.detach().clone()
    hooks.append(attn.to_k.register_forward_hook(to_k_hook))

    def to_v_hook(module, input, output):
        outputs["V"] = output.detach().clone()
    hooks.append(attn.to_v.register_forward_hook(to_v_hook))

    # Hook output projection
    def to_out_hook(module, input, output):
        outputs["out"] = output.detach().clone()
    hooks.append(attn.to_out[0].register_forward_hook(to_out_hook))  # to_out[0] is the Linear layer

    # Hook the full attention output
    def attn_hook(module, input, output):
        outputs["attn"] = output.detach().clone()
    hooks.append(attn.register_forward_hook(attn_hook))

    # Also hook norm1 output to verify input to attention
    def norm1_hook(module, input, output):
        outputs["norm1"] = output.detach().clone()
    hooks.append(first_tfmr.norm1.register_forward_hook(norm1_hook))

    print("\nRunning decoder with attention hooks...")

    with torch.no_grad():
        velocity = estimator.forward(
            x=initial_noise,
            mask=mask_T,
            mu=mu_T,
            t=t0,
            spks=spk_emb,
            cond=cond_T
        )

    for h in hooks:
        h.remove()

    print("\n" + "=" * 80)
    print("PYTHON ATTENTION INTERNALS (down_blocks[0][1][0].attn1):")
    print("=" * 80)

    print(f"\nnorm1 output (attention input):")
    v = outputs["norm1"]
    print(f"  shape: {list(v.shape)}")
    print(f"  range: [{v.min().item():.6f}, {v.max().item():.6f}]")

    print(f"\nQ (to_q output):")
    v = outputs["Q"]
    print(f"  shape: {list(v.shape)}")
    print(f"  range: [{v.min().item():.6f}, {v.max().item():.6f}]")

    print(f"\nK (to_k output):")
    v = outputs["K"]
    print(f"  shape: {list(v.shape)}")
    print(f"  range: [{v.min().item():.6f}, {v.max().item():.6f}]")

    print(f"\nV (to_v output):")
    v = outputs["V"]
    print(f"  shape: {list(v.shape)}")
    print(f"  range: [{v.min().item():.6f}, {v.max().item():.6f}]")

    print(f"\nout (to_out[0] output):")
    v = outputs["out"]
    print(f"  shape: {list(v.shape)}")
    print(f"  range: [{v.min().item():.6f}, {v.max().item():.6f}]")

    print(f"\nattn (full attention output):")
    v = outputs["attn"]
    print(f"  shape: {list(v.shape)}")
    print(f"  range: [{v.min().item():.6f}, {v.max().item():.6f}]")

    # Check weight shapes
    print("\n" + "=" * 80)
    print("ATTENTION WEIGHT SHAPES:")
    print("=" * 80)
    print(f"to_q.weight: {list(attn.to_q.weight.shape)}")
    print(f"to_k.weight: {list(attn.to_k.weight.shape)}")
    print(f"to_v.weight: {list(attn.to_v.weight.shape)}")
    print(f"to_out[0].weight: {list(attn.to_out[0].weight.shape)}")

    # Check some weight values
    print("\nFirst 5 values of to_q.weight[0,:]:")
    print(f"  {attn.to_q.weight[0, :5].tolist()}")

if __name__ == "__main__":
    main()
