#!/usr/bin/env python3
"""Trace Python decoder output at specific position (0, 8, 438)."""

import torch
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
sys.path.insert(0, str(PROJECT_ROOT / "python"))

def main():
    print("=" * 80)
    print("TRACING PYTHON DECODER AT POSITION (0, 8, 438)")
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

    # Register hooks to capture intermediate values at position (0, 8, 438)
    # Python decoder expects x: [B, C=80, T=696]
    # Position (0, 8, 438) means batch=0, channel=8, time=438

    outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if out.dim() == 3:  # [B, C, T]
                outputs[name] = {
                    'shape': out.shape,
                    'min': out.min().item(),
                    'max': out.max().item(),
                    'pos_8_438': out[0, 8, 438].item() if out.shape[1] > 8 and out.shape[2] > 438 else None
                }
        return hook

    # Add hooks to mid_blocks[11]
    hooks = []
    mid_block_11 = estimator.mid_blocks[11]

    # Hook resnet
    hooks.append(mid_block_11[0].register_forward_hook(make_hook('mid11_resnet')))

    # Hook transformers
    for i, tfmr in enumerate(mid_block_11[1]):
        hooks.append(tfmr.register_forward_hook(make_hook(f'mid11_tfmr{i}')))
        hooks.append(tfmr.ff.register_forward_hook(make_hook(f'mid11_tfmr{i}_ff')))

    # Run decoder
    with torch.no_grad():
        velocity = model.s3gen.flow.decoder.estimator(
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

    print("\nIntermediate outputs:")
    for name, data in sorted(outputs.items()):
        print(f"  {name}:")
        print(f"    shape: {data['shape']}")
        print(f"    min/max: [{data['min']:.4f}, {data['max']:.4f}]")
        if data['pos_8_438'] is not None:
            print(f"    [0, 8, 438]: {data['pos_8_438']:.6f}")

    print(f"\nFinal velocity at (0, 8, 438): {velocity[0, 8, 438].item():.6f}")

if __name__ == "__main__":
    main()
