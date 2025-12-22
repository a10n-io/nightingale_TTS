#!/usr/bin/env python3
"""Trace decoder FF values at mid_blocks[11].transformers[3] to compare with Swift."""

import torch
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
REF_DIR = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

def main():
    print("=" * 80)
    print("TRACING DECODER FF VALUES - MID BLOCKS 11, TRANSFORMER 3")
    print("=" * 80)

    # Load model
    sys.path.insert(0, str(PROJECT_ROOT / "python"))
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model_dir = PROJECT_ROOT / "models" / "chatterbox"
    model = ChatterboxMultilingualTTS.from_local(str(model_dir), device="cpu")
    s3 = model.s3gen

    # Load Step 7a inputs (same as Swift)
    print("\nLoading Step 7a reference inputs...")
    mu_T = torch.from_numpy(np.load(REF_DIR / "step7a_mu_T.npy"))
    cond_T = torch.from_numpy(np.load(REF_DIR / "step7a_cond_T.npy"))
    initial_noise = torch.from_numpy(np.load(REF_DIR / "step7a_initial_noise.npy"))
    mask_T = torch.from_numpy(np.load(REF_DIR / "step7a_mask_T.npy"))
    spk_emb = torch.from_numpy(np.load(REF_DIR / "step5_spk_emb.npy"))

    print(f"mu_T: {mu_T.shape}")
    print(f"cond_T: {cond_T.shape}")
    print(f"initial_noise: {initial_noise.shape}")
    print(f"mask_T: {mask_T.shape}")
    print(f"spk_emb: {spk_emb.shape}")

    # Get the decoder estimator
    estimator = s3.flow.decoder.estimator
    t0 = torch.tensor([0.0])

    # Hook to capture FF values
    ff_inputs = {}
    ff_outputs = {}

    def make_ff_hook(block_id, tfmr_id, layer_id):
        def hook(module, input, output):
            key = f"b{block_id}_t{tfmr_id}_l{layer_id}"
            ff_inputs[key] = input[0].detach().clone()
            ff_outputs[key] = output.detach().clone()
        return hook

    # Register hooks on mid_blocks[11].transformers[3].ff layers
    mid_block_11 = estimator.mid_blocks[11]
    transformer_3 = mid_block_11[1][3]  # mid_block is [resnet, transformers_list]
    ff = transformer_3.ff

    # ff.net is ModuleList: [GELU(with proj), Dropout, Linear]
    # Hook on GELU.proj (first linear)
    hooks = []
    hooks.append(ff.net[0].proj.register_forward_hook(make_ff_hook(11, 3, 0)))
    # Hook on final linear
    hooks.append(ff.net[2].register_forward_hook(make_ff_hook(11, 3, 1)))

    # Also hook norm2 (before ff input)
    def norm2_hook(module, input, output):
        ff_inputs["norm2_output"] = output.detach().clone()
    hooks.append(transformer_3.norm3.register_forward_hook(norm2_hook))  # Python uses norm3 (not norm2)

    print("\n" + "=" * 80)
    print("Running decoder forward pass with hooks...")
    print("=" * 80)

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
    print("PYTHON MID_BLOCKS[11].TRANSFORMERS[3].FF VALUES:")
    print("=" * 80)

    if "norm2_output" in ff_inputs:
        v = ff_inputs["norm2_output"]
        print(f"\nnorm2 output (ff input to GELU.proj):")
        print(f"  shape: {list(v.shape)}")
        print(f"  range: [{v.min().item():.6f}, {v.max().item():.6f}]")
        print(f"  mean: {v.mean().item():.6f}")

    for key in sorted(ff_inputs.keys()):
        if key.startswith("b11_t3"):
            v_in = ff_inputs[key]
            v_out = ff_outputs[key]
            print(f"\n{key}:")
            print(f"  input: range=[{v_in.min().item():.6f}, {v_in.max().item():.6f}], mean={v_in.mean().item():.6f}")
            print(f"  output: range=[{v_out.min().item():.6f}, {v_out.max().item():.6f}], mean={v_out.mean().item():.6f}")

    print(f"\nvelocity output: range=[{velocity.min().item():.4f}, {velocity.max().item():.4f}]")

    # Save debug values
    if "norm2_output" in ff_inputs:
        np.save(REF_DIR / "debug_ff_norm2_output.npy", ff_inputs["norm2_output"].numpy())
    for key, v in ff_outputs.items():
        if key.startswith("b11_t3"):
            np.save(REF_DIR / f"debug_ff_{key}_output.npy", v.numpy())

    print(f"\nSaved debug FF values to {REF_DIR}")

if __name__ == "__main__":
    main()
