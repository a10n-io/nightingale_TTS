#!/usr/bin/env python3
"""Comprehensive decoder trace to find Swift divergence."""

import torch
import numpy as np
from pathlib import Path
import sys

# Add chatterbox to path
sys.path.insert(0, str(Path.home() / "Library/Python/3.9/lib/python/site-packages"))

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

print("="*80)
print("COMPREHENSIVE DECODER TRACE")
print("="*80)

# Load full model to get decoder
model_path = PROJECT_ROOT / "models" / "chatterbox" / "s3gen.pt"
state_dict = torch.load(str(model_path), map_location='cpu')

# Just use the decoder (estimator) directly from state dict
# We don't need to load the full model, just manually run the decoder forward pass
print("✅ Loaded state dict")

# Load inputs
x = torch.from_numpy(np.load(ref_dir / "step7_step1_x_before.npy")[[0]])  # [1, 80, 696]
mu = torch.from_numpy(np.load(ref_dir / "step7_mu_T.npy")[[0]])           # [1, 80, 696]
spk_emb = torch.from_numpy(np.load(ref_dir / "step7_spk_emb.npy")[[0]])   # [1, 80]
x_cond = torch.from_numpy(np.load(ref_dir / "step6_x_cond.npy"))          # [1, 80, 696]
mask = torch.from_numpy(np.load(ref_dir / "step7_cond_T.npy")[[0]])       # [1, 1, 696]
t = torch.from_numpy(np.load(ref_dir / "step7_step1_t.npy"))              # []

print(f"\n📥 Inputs:")
print(f"  x: {x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")
print(f"  mu: {mu.shape}, range=[{mu.min():.4f}, {mu.max():.4f}]")
print(f"  spk_emb: {spk_emb.shape}, range=[{spk_emb.min():.4f}, {spk_emb.max():.4f}]")
print(f"  x_cond: {x_cond.shape}, range=[{x_cond.min():.4f}, {x_cond.max():.4f}]")
print(f"  mask: {mask.shape}, sum={mask.sum().item()}/{mask.numel()}")
print(f"  t: {t.shape}, value={t.item():.6f}")

# Patch decoder to trace
from einops import pack, rearrange

traces = {}

def save_trace(name, tensor):
    traces[name] = tensor.detach().clone()
    print(f"  {name}: range=[{tensor.min():.4f}, {tensor.max():.4f}]")

with torch.no_grad():
    # Time embedding
    t_emb_raw = decoder.estimator.time_embeddings(t)
    save_trace("time_emb_raw", t_emb_raw)
    t_emb = decoder.estimator.time_mlp(t_emb_raw)
    save_trace("time_emb", t_emb)

    # Concatenate inputs
    spks_expanded = spk_emb.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    h = pack([x, mu, spks_expanded, x_cond], "b * t")[0]
    save_trace("h_concat", h)

    # Down blocks
    hiddens = []
    masks = [mask]
    for block_idx, (resnet, transformer_blocks, downsample) in enumerate(decoder.estimator.down_blocks):
        print(f"\n🔽 Down Block {block_idx}")
        mask_down = masks[-1]

        # ResNet
        h_before_resnet = h.clone()
        save_trace(f"down{block_idx}_before_resnet", h_before_resnet)

        # Detailed ResNet trace
        h_block1 = resnet.block1(h, mask_down)
        save_trace(f"down{block_idx}_resnet_block1", h_block1)

        t_mlp_out = resnet.mlp(t_emb)
        save_trace(f"down{block_idx}_resnet_mlp", t_mlp_out)

        h_with_time = h_block1 + t_mlp_out.unsqueeze(-1)
        save_trace(f"down{block_idx}_resnet_with_time", h_with_time)

        h_block2 = resnet.block2(h_with_time, mask_down)
        save_trace(f"down{block_idx}_resnet_block2", h_block2)

        h_res_conv = resnet.res_conv(h * mask_down)
        save_trace(f"down{block_idx}_resnet_res_conv", h_res_conv)

        h = h_block2 + h_res_conv
        save_trace(f"down{block_idx}_after_resnet", h)

        # Transformers
        h = rearrange(h, "b c t -> b t c")
        mask_down = rearrange(mask_down, "b 1 t -> b t")
        for tfmr_idx, transformer_block in enumerate(transformer_blocks):
            h_before_tfmr = h.clone()
            h = transformer_block(
                hidden_states=h,
                attention_mask=mask_down,
                timestep=t_emb,
            )
            save_trace(f"down{block_idx}_tfmr{tfmr_idx}", h)
        h = rearrange(h, "b t c -> b c t")
        mask_down = rearrange(mask_down, "b t -> b 1 t")

        hiddens.append(h)
        h = downsample(h * mask_down)
        save_trace(f"down{block_idx}_after_downsample", h)
        masks.append(mask_down[:, :, ::2])

    # Mid blocks
    masks = masks[:-1]
    mask_mid = masks[-1]

    for block_idx, (resnet, transformer_blocks) in enumerate(decoder.estimator.mid_blocks):
        print(f"\n⏺ Mid Block {block_idx}")

        # ResNet
        h_before_resnet = h.clone()
        save_trace(f"mid{block_idx}_before_resnet", h_before_resnet)

        h_block1 = resnet.block1(h, mask_mid)
        save_trace(f"mid{block_idx}_resnet_block1", h_block1)

        t_mlp_out = resnet.mlp(t_emb)
        h_with_time = h_block1 + t_mlp_out.unsqueeze(-1)
        save_trace(f"mid{block_idx}_resnet_with_time", h_with_time)

        h_block2 = resnet.block2(h_with_time, mask_mid)
        save_trace(f"mid{block_idx}_resnet_block2", h_block2)

        h_res_conv = resnet.res_conv(h * mask_mid)
        save_trace(f"mid{block_idx}_resnet_res_conv", h_res_conv)

        h = h_block2 + h_res_conv
        save_trace(f"mid{block_idx}_after_resnet", h)

        # Transformers
        h = rearrange(h, "b c t -> b t c")
        mask_mid = rearrange(mask_mid, "b 1 t -> b t")
        for tfmr_idx, transformer_block in enumerate(transformer_blocks):
            h = transformer_block(
                hidden_states=h,
                attention_mask=mask_mid,
                timestep=t_emb,
            )
            save_trace(f"mid{block_idx}_tfmr{tfmr_idx}", h)
        h = rearrange(h, "b t c -> b c t")
        mask_mid = rearrange(mask_mid, "b t -> b 1 t")

    save_trace("before_up_blocks", h)

    # Up blocks
    for block_idx, (resnet, transformer_blocks, upsample) in enumerate(decoder.estimator.up_blocks):
        print(f"\n🔼 Up Block {block_idx}")
        mask_up = masks.pop()
        hidden = hiddens.pop()

        # Concatenate skip connection
        h_concat = pack([h, hidden], "b * t")[0]
        save_trace(f"up{block_idx}_after_concat", h_concat)

        # ResNet
        h_before_resnet = h_concat.clone()
        save_trace(f"up{block_idx}_before_resnet", h_before_resnet)

        h_block1 = resnet.block1(h_concat, mask_up)
        save_trace(f"up{block_idx}_resnet_block1", h_block1)

        t_mlp_out = resnet.mlp(t_emb)
        h_with_time = h_block1 + t_mlp_out.unsqueeze(-1)
        save_trace(f"up{block_idx}_resnet_with_time", h_with_time)

        h_block2 = resnet.block2(h_with_time, mask_up)
        save_trace(f"up{block_idx}_resnet_block2", h_block2)

        h_res_conv = resnet.res_conv(h_concat * mask_up)
        save_trace(f"up{block_idx}_resnet_res_conv", h_res_conv)

        h = h_block2 + h_res_conv
        save_trace(f"up{block_idx}_after_resnet", h)

        # Transformers
        h = rearrange(h, "b c t -> b t c")
        mask_up = rearrange(mask_up, "b 1 t -> b t")
        for tfmr_idx, transformer_block in enumerate(transformer_blocks):
            h = transformer_block(
                hidden_states=h,
                attention_mask=mask_up,
                timestep=t_emb,
            )
            save_trace(f"up{block_idx}_tfmr{tfmr_idx}", h)
        h = rearrange(h, "b t c -> b c t")
        mask_up = rearrange(mask_up, "b t -> b 1 t")

        h = upsample(h * mask_up)
        save_trace(f"up{block_idx}_after_upsample", h)

    # Final blocks
    print(f"\n🏁 Final Blocks")
    h_before_final = h.clone()
    save_trace("before_final_block", h_before_final)

    h = decoder.estimator.final_block(h, mask_up)
    save_trace("after_final_block", h)

    output = decoder.estimator.final_proj(h * mask_up)
    save_trace("after_final_proj", output)

    output = output * mask
    save_trace("final_output", output)

# Save all traces
print(f"\n💾 Saving {len(traces)} traces...")
for name, tensor in traces.items():
    np.save(ref_dir / f"decoder_trace_{name}.npy", tensor.numpy())

print(f"\n✅ Saved {len(traces)} trace files")
print("="*80)
