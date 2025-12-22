#!/usr/bin/env python3
"""Trace Python decoder block outputs to compare with Swift."""

import torch
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
sys.path.insert(0, str(PROJECT_ROOT / "python"))

def main():
    print("=" * 80)
    print("TRACING PYTHON DECODER BLOCK OUTPUTS")
    print("=" * 80)

    # Load Python model
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model_dir = PROJECT_ROOT / "models" / "chatterbox"
    model = ChatterboxMultilingualTTS.from_local(str(model_dir), device="cpu")
    estimator = model.s3gen.flow.decoder.estimator

    # Load reference inputs
    ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"
    initial_noise = torch.from_numpy(np.load(ref_dir / "step7a_initial_noise.npy"))
    mu_T = torch.from_numpy(np.load(ref_dir / "step7a_mu_T.npy"))
    cond_T = torch.from_numpy(np.load(ref_dir / "step7a_cond_T.npy"))
    mask_T = torch.from_numpy(np.load(ref_dir / "step7a_mask_T.npy"))
    spk_emb = torch.from_numpy(np.load(ref_dir / "step5_spk_emb.npy"))

    # Create t=0
    t = torch.zeros(1)

    # Put model in eval mode
    model.s3gen.flow.decoder.eval()

    # Register hooks to capture intermediate values
    outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if out is not None:
                outputs[name] = {
                    'shape': list(out.shape),
                    'min': out.min().item(),
                    'max': out.max().item(),
                    'mean': out.mean().item(),
                    'first_5': out.flatten()[:5].tolist() if out.numel() >= 5 else out.flatten().tolist()
                }
        return hook

    # Add hooks
    hooks = []

    # Time MLP
    hooks.append(estimator.time_mlp.register_forward_hook(make_hook('time_mlp')))

    # Down blocks
    for i, (resnet, tfmrs, downsample) in enumerate(estimator.down_blocks):
        hooks.append(resnet.register_forward_hook(make_hook(f'down[{i}].resnet')))
        for j, tfmr in enumerate(tfmrs):
            hooks.append(tfmr.register_forward_hook(make_hook(f'down[{i}].tfmr[{j}]')))
        hooks.append(downsample.register_forward_hook(make_hook(f'down[{i}].downsample')))

    # Mid blocks
    for i, (resnet, tfmrs) in enumerate(estimator.mid_blocks):
        hooks.append(resnet.register_forward_hook(make_hook(f'mid[{i}].resnet')))
        for j, tfmr in enumerate(tfmrs):
            hooks.append(tfmr.register_forward_hook(make_hook(f'mid[{i}].tfmr[{j}]')))

    # Up blocks
    for i, (resnet, tfmrs, upsample) in enumerate(estimator.up_blocks):
        hooks.append(resnet.register_forward_hook(make_hook(f'up[{i}].resnet')))
        for j, tfmr in enumerate(tfmrs):
            hooks.append(tfmr.register_forward_hook(make_hook(f'up[{i}].tfmr[{j}]')))
        hooks.append(upsample.register_forward_hook(make_hook(f'up[{i}].upsample')))

    # Final
    hooks.append(estimator.final_block.register_forward_hook(make_hook('final_block')))
    hooks.append(estimator.final_proj.register_forward_hook(make_hook('final_proj')))

    # Run decoder
    with torch.no_grad():
        velocity = estimator(
            x=initial_noise,
            mask=mask_T,
            mu=mu_T,
            t=t,
            spks=spk_emb,
            cond=cond_T
        )

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print results
    print("\nBlock outputs:")
    for name, data in sorted(outputs.items()):
        print(f"\n{name}:")
        print(f"  shape: {data['shape']}")
        print(f"  min/max: [{data['min']:.6f}, {data['max']:.6f}]")
        print(f"  mean: {data['mean']:.6f}")
        print(f"  first_5: {[f'{x:.6f}' for x in data['first_5']]}")

    print(f"\n\nFinal velocity:")
    print(f"  shape: {list(velocity.shape)}")
    print(f"  min/max: [{velocity.min().item():.6f}, {velocity.max().item():.6f}]")

if __name__ == "__main__":
    main()
