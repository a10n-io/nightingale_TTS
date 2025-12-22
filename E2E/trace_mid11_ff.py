#!/usr/bin/env python3
"""Trace mid[11] transformer FF layers to compare with Swift."""

import torch
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
sys.path.insert(0, str(PROJECT_ROOT / "python"))

def main():
    print("=" * 80)
    print("TRACING MID[11] FF LAYERS")
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

    print(f"\nInputs loaded:")
    print(f"  initial_noise: {initial_noise.shape}")
    print(f"  mu_T: {mu_T.shape}")
    print(f"  cond_T: {cond_T.shape}")
    print(f"  mask_T: {mask_T.shape}")
    print(f"  spk_emb: {spk_emb.shape}")

    # Run decoder with hooks to capture intermediates
    mid_block_11 = estimator.mid_blocks[11]
    transformers = mid_block_11[1]  # ModuleList of transformers

    # We need to run the decoder up to mid_block[11]
    # This is complex, so let's just check the FF weights directly

    print("\n" + "=" * 40)
    print("MID[11] FF LAYER WEIGHTS:")
    print("=" * 40)

    for tfmr_idx in [2, 3]:
        tfmr = transformers[tfmr_idx]
        ff = tfmr.ff
        # ff.net is Sequential: [GEGLU, Dropout, Linear]
        # GEGLU: Linear + GELU activation
        # ff.net[0] is GEGLU with proj layer
        # ff.net[2] is output Linear

        print(f"\nTransformer[{tfmr_idx}]:")

        # GEGLU inner
        geglu = ff.net[0]
        proj_w = geglu.proj.weight  # [out*2, in]
        proj_b = geglu.proj.bias if geglu.proj.bias is not None else None
        print(f"  ff.net[0] (GEGLU) proj.weight: {proj_w.shape}")
        print(f"    first 5: {proj_w.flatten()[:5].tolist()}")
        print(f"    sum: {proj_w.sum().item():.6f}")

        # Output linear
        out_linear = ff.net[2]
        out_w = out_linear.weight
        out_b = out_linear.bias if out_linear.bias is not None else None
        print(f"  ff.net[2] (Linear) weight: {out_w.shape}")
        print(f"    first 5: {out_w.flatten()[:5].tolist()}")
        print(f"    sum: {out_w.sum().item():.6f}")

    # Also check what the safetensors has
    print("\n" + "=" * 40)
    print("SAFETENSORS WEIGHTS:")
    print("=" * 40)

    from safetensors import safe_open

    st_path = PROJECT_ROOT / "models" / "mlx" / "python_flow_weights.safetensors"
    with safe_open(st_path, framework="pt", device="cpu") as f:
        for key in sorted(f.keys()):
            if "mid_blocks_11" in key and ("layers.0" in key or "layers.1" in key) and "transformer_2" in key:
                t = f.get_tensor(key)
                print(f"  {key}: {t.shape}")
                print(f"    first 5: {t.flatten()[:5].tolist()}")

if __name__ == "__main__":
    main()
