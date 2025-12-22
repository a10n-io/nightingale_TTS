#!/usr/bin/env python3
"""Trace down_blocks more carefully to find where divergence starts."""

import torch
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
REF_DIR = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

def main():
    print("=" * 80)
    print("TRACING DOWN BLOCKS AND MID[0] DETAIL")
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
    outputs = {}
    hooks = []

    # Hook down_blocks[0] components
    down_block = estimator.down_blocks[0]
    resnet = down_block[0]
    tfmr_list = down_block[1]
    downsample = down_block[2]

    # Hook resnet output
    def resnet_hook(module, input, output):
        outputs["down_resnet"] = (output.min().item(), output.max().item())
        outputs["down_resnet_shape"] = list(output.shape)
    hooks.append(resnet.register_forward_hook(resnet_hook))

    # Hook each transformer in down_blocks[0]
    for i, tfmr in enumerate(tfmr_list):
        def make_hook(idx):
            def hook(module, input, output):
                outputs[f"down_tfmr_{idx}"] = (output.min().item(), output.max().item())
            return hook
        hooks.append(tfmr.register_forward_hook(make_hook(i)))

    # Hook downsample
    def downsample_hook(module, input, output):
        outputs["downsample"] = (output.min().item(), output.max().item())
    hooks.append(downsample.register_forward_hook(downsample_hook))

    # Hook mid_blocks[0] components
    mid_block0 = estimator.mid_blocks[0]
    mid_resnet0 = mid_block0[0]
    mid_tfmr_list0 = mid_block0[1]

    def mid_resnet_hook(module, input, output):
        outputs["mid0_resnet"] = (output.min().item(), output.max().item())
    hooks.append(mid_resnet0.register_forward_hook(mid_resnet_hook))

    for i, tfmr in enumerate(mid_tfmr_list0):
        def make_hook(idx):
            def hook(module, input, output):
                outputs[f"mid0_tfmr_{idx}"] = (output.min().item(), output.max().item())
            return hook
        hooks.append(tfmr.register_forward_hook(make_hook(i)))

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

    print("\n" + "=" * 80)
    print("PYTHON DOWN_BLOCKS[0] DETAIL:")
    print("=" * 80)

    print(f"\ndown_resnet: {outputs['down_resnet']} shape={outputs['down_resnet_shape']}")
    for i in range(4):
        key = f"down_tfmr_{i}"
        if key in outputs:
            print(f"down_tfmr[{i}]: {outputs[key]}")
    print(f"downsample: {outputs.get('downsample')}")

    print("\n" + "=" * 80)
    print("PYTHON MID_BLOCKS[0] DETAIL:")
    print("=" * 80)

    print(f"\nmid0_resnet: {outputs.get('mid0_resnet')}")
    for i in range(4):
        key = f"mid0_tfmr_{i}"
        if key in outputs:
            print(f"mid0_tfmr[{i}]: {outputs[key]}")

    print(f"\nFinal velocity: [{velocity.min().item():.4f}, {velocity.max().item():.4f}]")

if __name__ == "__main__":
    main()
