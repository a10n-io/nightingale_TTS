#!/usr/bin/env python3
"""Debug Step 7a decoder internals - save intermediate values."""

import torch
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
MODEL_DIR = PROJECT_ROOT / "models" / "chatterbox"

def main():
    ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

    print("=" * 80)
    print("DEBUG STEP 7a INTERNALS")
    print("=" * 80)

    # Load the model
    sys.path.insert(0, str(PROJECT_ROOT / "python"))
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device="cpu")
    s3 = model.s3gen
    estimator = s3.flow.decoder.estimator

    # Load inputs
    mu = torch.from_numpy(np.load(ref_dir / "step6_mu.npy"))
    x_cond = torch.from_numpy(np.load(ref_dir / "step6_x_cond.npy"))
    spk_emb = torch.from_numpy(np.load(ref_dir / "step5_spk_emb.npy"))

    L_total = mu.shape[2]
    mask = torch.ones(1, 1, L_total)

    torch.manual_seed(0)
    x = torch.randn(1, 80, L_total, dtype=mu.dtype)
    t = torch.tensor([0.0])

    print(f"x: {list(x.shape)}")
    print(f"mu: {list(mu.shape)}")
    print(f"spk_emb: {list(spk_emb.shape)}")
    print(f"cond: {list(x_cond.shape)}")
    print(f"t: {t.item()}")

    # Manually trace through the decoder
    print("\n" + "=" * 80)
    print("TRACING DECODER INTERNALS")
    print("=" * 80)

    # Time embedding
    t_emb = estimator.time_embeddings(t).to(t.dtype)
    t_emb = estimator.time_mlp(t_emb)
    print(f"\nTime embedding:")
    print(f"  t_emb shape: {list(t_emb.shape)}")
    print(f"  t_emb[:5]: {t_emb[0, :5].tolist()}")
    print(f"  t_emb mean: {t_emb.mean().item():.6f}")

    # Input concatenation
    from einops import pack, repeat
    h = pack([x, mu], "b * t")[0]
    print(f"\nAfter pack [x, mu]: h shape = {list(h.shape)}")

    if spk_emb is not None:
        spks_tiled = repeat(spk_emb, "b c -> b c t", t=h.shape[-1])
        h = pack([h, spks_tiled], "b * t")[0]
        print(f"After pack [h, spks_tiled]: h shape = {list(h.shape)}")

    if x_cond is not None:
        h = pack([h, x_cond], "b * t")[0]
        print(f"After pack [h, cond]: h shape = {list(h.shape)}")

    print(f"\nFinal h shape: {list(h.shape)}")
    print(f"h[0, :5, 0]: {h[0, :5, 0].tolist()}")
    print(f"h[0, 80:85, 0]: {h[0, 80:85, 0].tolist()}")
    print(f"h[0, 160:165, 0]: {h[0, 160:165, 0].tolist()}")
    print(f"h[0, 240:245, 0]: {h[0, 240:245, 0].tolist()}")
    print(f"h mean: {h.mean().item():.6f}")

    # Save intermediate values
    np.save(ref_dir / "debug_t_emb.npy", t_emb.detach().cpu().numpy())
    np.save(ref_dir / "debug_h_concat.npy", h.detach().cpu().numpy())

    print(f"\nSaved debug_t_emb.npy and debug_h_concat.npy")

    # First ResNet block
    print("\n" + "=" * 80)
    print("FIRST RESNET BLOCK")
    print("=" * 80)

    resnet, transformer_blocks, downsample = estimator.down_blocks[0]
    mask_down = mask

    h_after_resnet = resnet(h, mask_down, t_emb)
    print(f"After ResNet: h shape = {list(h_after_resnet.shape)}")
    print(f"h_after_resnet[0, :5, 0]: {h_after_resnet[0, :5, 0].tolist()}")
    print(f"h_after_resnet mean: {h_after_resnet.mean().item():.6f}")

    np.save(ref_dir / "debug_h_after_resnet.npy", h_after_resnet.detach().cpu().numpy())
    print(f"Saved debug_h_after_resnet.npy")


if __name__ == "__main__":
    main()
