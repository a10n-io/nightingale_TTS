#!/usr/bin/env python3
"""Save a decoder trace with synthetic inputs for Swift comparison."""
import torch
from pathlib import Path
import safetensors.torch as st
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import numpy as np

# Load model
MODELS_DIR = Path("models/chatterbox")
device = "cpu"  # Force CPU for compatibility

print("Loading Chatterbox model...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Generate EXACT same inputs as trace_python_tfmr_detailed.py
L_total = 696
L_pm = 500

# Set fixed seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create synthetic inputs
mu = torch.randn(1, 80, L_total, device=device)
# conds: first 500 frames have values, rest are zeros
conds = torch.randn(1, 80, L_pm, device=device)
conds = torch.cat([conds, torch.zeros(1, 80, L_total - L_pm, device=device)], dim=2)
speaker_emb = torch.randn(1, 80, device=device)

decoder = model.s3gen.flow.decoder.estimator

# ODE parameters
n_timesteps = 10
timesteps = torch.cos(torch.linspace(0, np.pi / 2, n_timesteps + 1, device=device)) ** 2
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

print(f"Output shape: {output.shape}")
print(f"Output range: [{output.min().item()}, {output.max().item()}]")
print(f"Output mean: {output.mean().item()}")

# Save trace (make all tensors contiguous)
trace_dict = {
    "noise": x_batch.cpu().contiguous(),
    "mu": mu_batch.cpu().contiguous(),
    "spk_cond": speaker_batch.cpu().contiguous(),
    "conds": conds_batch.cpu().contiguous(),
    "mask": mask_batch.cpu().contiguous(),
    "t": t_batch.cpu().contiguous(),
    "decoder_output": output.cpu().contiguous()
}

output_path = Path("test_audio/python_decoder_trace.safetensors")
output_path.parent.mkdir(exist_ok=True)
st.save_file(trace_dict, output_path)
print(f"\n✅ Saved trace to {output_path}")

# Also save as individual NPY files for MLX compatibility
import numpy as np
npy_dir = Path("test_audio/decoder_trace_npy")
npy_dir.mkdir(exist_ok=True)
for key, value in trace_dict.items():
    np.save(npy_dir / f"{key}.npy", value.numpy())
print(f"✅ Also saved as NPY files to {npy_dir}")

# Also save as raw binary for easier Swift loading
bin_dir = Path("test_audio/decoder_trace_bin")
bin_dir.mkdir(exist_ok=True)
for key, value in trace_dict.items():
    # Save shape as text
    np_arr = value.numpy()
    with open(bin_dir / f"{key}.shape", "w") as f:
        f.write(" ".join(map(str, np_arr.shape)))
    # Save data as float32 binary
    np_arr.astype(np.float32).tofile(bin_dir / f"{key}.bin")
print(f"✅ Also saved as binary files to {bin_dir}")

# Also print component spatial biases for verification
L_pm = 500

def check_spatial(tensor, label):
    """Check spatial bias in [B, C, T] format."""
    if tensor.ndim == 3 and tensor.shape[2] >= L_pm:
        prompt = tensor[0, :, :L_pm]
        generated = tensor[0, :, L_pm:]
        p_mean = prompt.mean().item()
        g_mean = generated.mean().item()
        bias = g_mean - p_mean
        print(f"{label}: prompt={p_mean:.4f}, generated={g_mean:.4f}, bias={bias:.4f}")

print("\n Spatial bias check:")
check_spatial(x_batch, "x_batch")
check_spatial(mu_batch, "mu_batch")
check_spatial(conds_batch, "conds_batch")
check_spatial(output, "output")
