#!/usr/bin/env python3
"""Trace decoder transformers to find Swift divergence."""

import torch
import numpy as np
from pathlib import Path
import sys

# Add chatterbox to path
sys.path.insert(0, str(Path.home() / "Library/Python/3.9/lib/python/site-packages"))
from chatterbox.models.s3gen import S3GenModel

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

print("="*80)
print("DECODER TRANSFORMER TRACE")
print("="*80)

# Load full model
print("\n📦 Loading S3Gen model...")
model_path = PROJECT_ROOT / "models" / "chatterbox" / "s3gen.pt"
model = S3GenModel.load_from_checkpoint(str(model_path), map_location='cpu')
model.eval()
decoder = model.flow.decoder
print("✅ Loaded model")

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
print(f"  mask: {mask.shape}, sum={mask.sum().item()}/{mask.numel()}")
print(f"  t: {t.shape}, value={t.item():.6f}")

# Patch transformer blocks to trace
from einops import pack, rearrange

traces = {}

def save_trace(name, tensor):
    traces[name] = tensor.detach().clone()
    print(f"  {name}: {tensor.shape}, range=[{tensor.min():.4f}, {tensor.max():.4f}]")

# Hook to trace transformer internal operations
def create_transformer_hook(block_name):
    def hook(module, input, output):
        save_trace(block_name, output)
    return hook

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
        h = resnet(h, t_emb, mask_down)
        save_trace(f"down{block_idx}_after_resnet", h)

        # Transformers - trace the first one in detail
        h = rearrange(h, "b c t -> b t c")
        mask_down_2d = rearrange(mask_down, "b 1 t -> b t")

        for tfmr_idx, transformer_block in enumerate(transformer_blocks):
            if block_idx == 0 and tfmr_idx == 0:
                # Detailed trace of first transformer
                print(f"  📐 Tracing transformer {block_idx}.{tfmr_idx} in detail...")

                # Save input
                h_in = h.clone()
                save_trace(f"tfmr_input", h_in)

                # Manually step through the transformer
                # norm1
                h_norm1 = transformer_block.norm1(h)
                save_trace(f"tfmr_after_norm1", h_norm1)

                # Self-attention
                attn_output = transformer_block.attn1(
                    h_norm1,
                    attention_mask=mask_down_2d,
                )
                save_trace(f"tfmr_attn_output", attn_output)

                # Residual 1
                h = h + attn_output
                save_trace(f"tfmr_after_res1", h)

                # norm3 (FFN norm)
                h_norm3 = transformer_block.norm3(h)
                save_trace(f"tfmr_after_norm3", h_norm3)

                # Feedforward
                ff_output = transformer_block.ff(h_norm3)
                save_trace(f"tfmr_ff_output", ff_output)

                # Residual 2
                h = h + ff_output
                save_trace(f"tfmr_final_output", h)
            else:
                # Just run normally
                h = transformer_block(
                    hidden_states=h,
                    attention_mask=mask_down_2d,
                    timestep=t_emb,
                )

            save_trace(f"down{block_idx}_tfmr{tfmr_idx}", h)

        h = rearrange(h, "b t c -> b c t")
        mask_down = rearrange(mask_down_2d, "b t -> b 1 t")

        hiddens.append(h)
        h = downsample(h * mask_down)
        save_trace(f"down{block_idx}_after_downsample", h)
        masks.append(mask_down[:, :, ::2])

# Save all traces
print(f"\n💾 Saving {len(traces)} traces...")
for name, tensor in traces.items():
    np.save(ref_dir / f"decoder_trace_{name}.npy", tensor.numpy())

print(f"\n✅ Saved {len(traces)} trace files")
print("="*80)
