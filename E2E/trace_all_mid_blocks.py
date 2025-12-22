#!/usr/bin/env python3
"""Trace all mid block outputs in Python decoder using hooks."""

import torch
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
REF_DIR = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

def main():
    print("=" * 80)
    print("TRACING ALL MID BLOCK OUTPUTS IN PYTHON DECODER (using hooks)")
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

    print(f"Inputs loaded: mu_T={mu_T.shape}, noise={initial_noise.shape}")

    estimator = s3.flow.decoder.estimator
    mid_outputs = {}

    # Register hooks on each mid_block's last component (the transformers list)
    hooks = []

    # For down_blocks, capture the output after downsample
    def down_hook(module, input, output):
        mid_outputs["down"] = (output.min().item(), output.max().item())

    # Hook the downsample layer (index 2)
    hooks.append(estimator.down_blocks[0][2].register_forward_hook(down_hook))

    # For mid_blocks, we need to hook the transformer outputs
    # But transformers are in a ModuleList, so hook each transformer
    for i in range(12):
        mid_block = estimator.mid_blocks[i]
        tfmr_list = mid_block[1]  # [resnet, transformers_list]
        last_tfmr = tfmr_list[3]  # Last transformer in the block

        def make_hook(block_id):
            def hook(module, input, output):
                # Output is in [B, T, C] format after transformer
                mid_outputs[f"mid_{block_id}"] = (output.min().item(), output.max().item())
            return hook

        hooks.append(last_tfmr.register_forward_hook(make_hook(i)))

    # Also hook time_mlp
    def time_mlp_hook(module, input, output):
        mid_outputs["time_mlp"] = (output.min().item(), output.max().item())
        mid_outputs["time_mlp_vals"] = output[0, :5].tolist()

    hooks.append(estimator.time_mlp.register_forward_hook(time_mlp_hook))

    print("\nRunning decoder with hooks...")

    with torch.no_grad():
        velocity = estimator.forward(
            x=initial_noise,
            mask=mask_T,
            mu=mu_T,
            t=t0,
            spks=spk_emb,
            cond=cond_T
        )

    # Remove hooks
    for h in hooks:
        h.remove()

    # Print results
    print("\n" + "=" * 80)
    print("PYTHON DECODER INTERMEDIATE VALUES:")
    print("=" * 80)

    print(f"\ntime_mlp: [{mid_outputs['time_mlp'][0]:.4f}, {mid_outputs['time_mlp'][1]:.4f}]")
    print(f"time_mlp[:5]: {mid_outputs['time_mlp_vals']}")

    if "down" in mid_outputs:
        print(f"\nAfter down_blocks: [{mid_outputs['down'][0]:.4f}, {mid_outputs['down'][1]:.4f}]")

    print("\nMid block outputs (after all transformers):")
    for i in range(12):
        key = f"mid_{i}"
        if key in mid_outputs:
            print(f"  mid[{i}]: [{mid_outputs[key][0]:.4f}, {mid_outputs[key][1]:.4f}]")

    print(f"\nFinal velocity: [{velocity.min().item():.4f}, {velocity.max().item():.4f}]")

if __name__ == "__main__":
    main()
