#!/usr/bin/env python3
"""Trace Python decoder to check spatial bias before/after finalProj."""
import torch
from pathlib import Path
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import numpy as np

# Load model
MODELS_DIR = Path("models/chatterbox")
device = "cpu"

print("Loading Chatterbox model...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate test inputs
L_total = 696
L_pm = 500

mu = torch.randn(1, 80, L_total, device=device)
conds = torch.randn(1, 80, L_pm, device=device)
conds = torch.cat([conds, torch.zeros(1, 80, L_total - L_pm, device=device)], dim=2)
speaker_emb = torch.randn(1, 80, device=device)

decoder = model.s3gen.flow.decoder.estimator

# ODE parameters
timesteps = torch.cos(torch.linspace(0, np.pi / 2, 11, device=device)) ** 2
xt = torch.randn(1, L_total, 80, device=device)

# Run one ODE step
t_curr = timesteps[0]
t_batch = t_curr.unsqueeze(0).expand(2)

x_transposed = xt.transpose(1, 2)
x_batch = torch.cat([x_transposed, x_transposed], dim=0)
mu_batch = torch.cat([mu, mu], dim=0)
conds_batch = torch.cat([conds, torch.zeros_like(conds)], dim=0)
speaker_batch = speaker_emb.expand(2, -1)
mask_batch = torch.ones(2, 1, L_total, device=device)

print("Running decoder...")
with torch.no_grad():
    output = decoder(
        x=x_batch,
        mask=mask_batch,
        mu=mu_batch,
        t=t_batch,
        spks=speaker_batch,
        cond=conds_batch
    )

# Check spatial bias
def check_spatial(tensor, label):
    if tensor.ndim == 3 and tensor.shape[2] >= L_pm:
        prompt = tensor[0, :, :L_pm]
        generated = tensor[0, :, L_pm:]
        p_mean = prompt.mean().item()
        g_mean = generated.mean().item()
        bias = g_mean - p_mean
        print(f"{label}: prompt={p_mean:.4f}, generated={g_mean:.4f}, bias={bias:.4f}")

        # Also check per-channel for a few channels
        print(f"  Per-channel biases:")
        for c in [0, 20, 40, 60, 79]:
            p_c = prompt[c, :].mean().item()
            g_c = generated[c, :].mean().item()
            bias_c = g_c - p_c
            print(f"    ch{c:2d}: bias={bias_c:.4f}")

check_spatial(output, "Python decoder output")

print("\n✅ Done")
