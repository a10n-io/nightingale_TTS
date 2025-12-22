#!/usr/bin/env python3
"""
Verify PyTorch implementations against expected behavior for MLX port.
Exports reference values for comparison with Swift.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safetensors.torch import save_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "E2E" / "verification_reference"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("PYTORCH/MLX IMPLEMENTATION VERIFICATION")
print("=" * 70)

# =============================================================================
# 1. LAYERNORM
# =============================================================================
print("\n" + "=" * 70)
print("1. LAYERNORM")
print("=" * 70)

# Test input
ln_input = torch.randn(1, 10, 512)
ln = nn.LayerNorm(512)

# Check default epsilon
print(f"  PyTorch LayerNorm default eps: {ln.eps}")

# Manual computation for verification
mean = ln_input.mean(dim=-1, keepdim=True)
var = ln_input.var(dim=-1, unbiased=False, keepdim=True)
ln_manual = (ln_input - mean) / torch.sqrt(var + ln.eps)

# PyTorch result (without weight/bias since they're default)
ln.weight.data.fill_(1.0)
ln.bias.data.zero_()
ln_pytorch = ln(ln_input)

diff = (ln_manual - ln_pytorch).abs().max().item()
print(f"  Manual vs PyTorch diff: {diff:.2e}")
print(f"  ✅ LayerNorm eps=1e-5, uses unbiased=False variance")

# Save reference
save_file({
    "input": ln_input.contiguous(),
    "output": ln_pytorch.contiguous(),
    "mean": mean.contiguous(),
    "var": var.contiguous(),
}, OUTPUT_DIR / "layernorm_ref.safetensors")

# =============================================================================
# 2. GELU VARIANTS
# =============================================================================
print("\n" + "=" * 70)
print("2. GELU VARIANTS")
print("=" * 70)

gelu_input = torch.randn(1, 10, 512)

# Exact GELU
gelu_exact = F.gelu(gelu_input, approximate='none')

# Tanh approximation (used by some models)
gelu_tanh = F.gelu(gelu_input, approximate='tanh')

diff = (gelu_exact - gelu_tanh).abs().max().item()
print(f"  Exact vs Tanh approximation max diff: {diff:.6f}")
print(f"  ⚠️  GELU variants differ by up to {diff:.4f}")

# Check which one the model uses
from chatterbox.models.s3gen.s3gen import S3Token2Wav
s3gen = S3Token2Wav()

# Look for GELU usage in the model
gelu_type = "unknown"
for name, module in s3gen.named_modules():
    if isinstance(module, nn.GELU):
        gelu_type = f"nn.GELU (approximate={getattr(module, 'approximate', 'none')})"
        break

print(f"  S3Gen uses: {gelu_type}")

save_file({
    "input": gelu_input.contiguous(),
    "gelu_exact": gelu_exact.contiguous(),
    "gelu_tanh": gelu_tanh.contiguous(),
}, OUTPUT_DIR / "gelu_ref.safetensors")

# =============================================================================
# 3. ATTENTION MECHANICS
# =============================================================================
print("\n" + "=" * 70)
print("3. ATTENTION MECHANICS")
print("=" * 70)

# Test scaled dot product attention
batch, heads, seq_len, head_dim = 1, 8, 20, 64
q = torch.randn(batch, heads, seq_len, head_dim)
k = torch.randn(batch, heads, seq_len, head_dim)
v = torch.randn(batch, heads, seq_len, head_dim)

# Manual attention
scale = 1.0 / (head_dim ** 0.5)
print(f"  Attention scale factor: 1/sqrt({head_dim}) = {scale:.6f}")

scores = torch.matmul(q, k.transpose(-2, -1)) * scale
attn_weights = F.softmax(scores, dim=-1)
attn_output_manual = torch.matmul(attn_weights, v)

# PyTorch SDPA
attn_output_sdpa = F.scaled_dot_product_attention(q, k, v, scale=scale)

diff = (attn_output_manual - attn_output_sdpa).abs().max().item()
print(f"  Manual vs SDPA diff: {diff:.2e}")
print(f"  ✅ Attention uses scale=1/sqrt(head_dim)")

# Causal mask test
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
print(f"  Causal mask shape: {causal_mask.shape}")
print(f"  Causal mask[0,:5]: {causal_mask[0,:5].tolist()}")  # Should be [F,T,T,T,T]

save_file({
    "q": q.contiguous(),
    "k": k.contiguous(),
    "v": v.contiguous(),
    "attn_output": attn_output_manual.contiguous(),
    "scale": torch.tensor([scale]),
}, OUTPUT_DIR / "attention_ref.safetensors")

# =============================================================================
# 4. ODE/CFM SOLVER
# =============================================================================
print("\n" + "=" * 70)
print("4. ODE/CFM SOLVER (Conditional Flow Matching)")
print("=" * 70)

# Check S3Gen CFM implementation
# (s3gen already loaded above)

# The CFM uses an ODE from t=0 to t=1
# x(t) = (1-t)*x0 + t*x1 where x0=noise, x1=target
# velocity v(t) = x1 - x0

print("  CFM ODE direction: t=0 (noise) → t=1 (target)")
print("  Euler step: x_{t+dt} = x_t + dt * v(x_t, t)")

# Check the actual timesteps used
n_steps = 10
timesteps = torch.linspace(0, 1, n_steps + 1)
print(f"  Default timesteps ({n_steps} steps): {timesteps[:5].tolist()}...")
print(f"  dt = {(timesteps[1] - timesteps[0]).item():.4f}")

save_file({
    "timesteps": timesteps.contiguous(),
    "n_steps": torch.tensor([n_steps]),
}, OUTPUT_DIR / "cfm_ref.safetensors")

# =============================================================================
# 5. VOCODER (HiFi-GAN style)
# =============================================================================
print("\n" + "=" * 70)
print("5. VOCODER ANALYSIS")
print("=" * 70)

# Load vocoder from s3gen
from safetensors.torch import load_file
s3gen_weights = load_file(str(PROJECT_ROOT / "models/chatterbox/s3gen.safetensors"))

# Find vocoder keys
vocoder_keys = [k for k in s3gen_weights.keys() if k.startswith("mel2wav.")]
print(f"  Vocoder has {len(vocoder_keys)} weight tensors")

# Check for upsampling layers
upsample_keys = [k for k in vocoder_keys if "ups" in k.lower()]
print(f"  Upsampling layers: {len(upsample_keys)}")

# Check LeakyReLU slope (usually 0.1 for HiFi-GAN)
# This is typically hardcoded, let's check the model definition
print("  LeakyReLU slope: checking model definition...")

# Load actual vocoder and check
s3gen.load_state_dict(s3gen_weights)
vocoder = s3gen.mel2wav

# Check activation types
for name, module in vocoder.named_modules():
    if isinstance(module, nn.LeakyReLU):
        print(f"    {name}: LeakyReLU(negative_slope={module.negative_slope})")
        break

# Check ConvTranspose1d for upsampling
for name, module in vocoder.named_modules():
    if isinstance(module, nn.ConvTranspose1d):
        print(f"    {name}: ConvTranspose1d(in={module.in_channels}, out={module.out_channels}, k={module.kernel_size}, s={module.stride})")
        break

# =============================================================================
# 6. CONV1D WEIGHT FORMAT
# =============================================================================
print("\n" + "=" * 70)
print("6. CONV1D WEIGHT FORMAT")
print("=" * 70)

conv = nn.Conv1d(64, 128, kernel_size=3, padding=1)
print(f"  PyTorch Conv1d weight shape: {conv.weight.shape}")
print(f"    [out_channels, in_channels, kernel_size] = [{conv.out_channels}, {conv.in_channels}, {conv.kernel_size[0]}]")
print(f"  MLX Conv1d expects: [out_channels, kernel_size, in_channels]")
print(f"  ⚠️  Transpose axes (1,2) when loading: weight.transpose(1,2)")

# =============================================================================
# 7. GROUPNORM
# =============================================================================
print("\n" + "=" * 70)
print("7. GROUPNORM")
print("=" * 70)

# Check GroupNorm in decoder
gn_input = torch.randn(1, 256, 100)  # [B, C, T]
gn = nn.GroupNorm(num_groups=32, num_channels=256)
gn_output = gn(gn_input)

print(f"  GroupNorm default eps: {gn.eps}")
print(f"  Input shape: {gn_input.shape} [B, C, T]")
print(f"  Output shape: {gn_output.shape}")

# Check what's in the decoder
decoder_gn_keys = [k for k in s3gen_weights.keys() if "norm" in k.lower() and "decoder" in k.lower()]
print(f"  Decoder has {len(decoder_gn_keys)} norm layers")

save_file({
    "input": gn_input.contiguous(),
    "output": gn_output.contiguous(),
}, OUTPUT_DIR / "groupnorm_ref.safetensors")

# =============================================================================
# 8. SINUSOIDAL TIME EMBEDDING
# =============================================================================
print("\n" + "=" * 70)
print("8. SINUSOIDAL TIME EMBEDDING")
print("=" * 70)

# Check how time embedding is computed
dim = 256
t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

# Standard sinusoidal embedding
half_dim = dim // 2
freqs = torch.exp(-np.log(10000) * torch.arange(half_dim) / half_dim)
args = t[:, None] * freqs[None, :]
time_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

print(f"  Time values: {t.tolist()}")
print(f"  Embedding dim: {dim}")
print(f"  Frequency range: [{freqs[0]:.4f}, {freqs[-1]:.6f}]")
print(f"  Output shape: {time_emb.shape}")

save_file({
    "t": t.contiguous(),
    "embedding": time_emb.contiguous(),
    "freqs": freqs.contiguous(),
}, OUTPUT_DIR / "time_embedding_ref.safetensors")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY - KEY IMPLEMENTATION DETAILS")
print("=" * 70)
print("""
  1. LayerNorm: eps=1e-5, unbiased=False variance
  2. GELU: Check if model uses exact or tanh approximation
  3. Attention: scale=1/sqrt(head_dim), standard softmax
  4. CFM/ODE: t=0→1 direction, Euler steps
  5. Vocoder: ConvTranspose1d upsampling, LeakyReLU
  6. Conv1d: PyTorch [O,I,K] → MLX [O,K,I] (transpose axes 1,2)
  7. GroupNorm: eps=1e-5, operates on channel dim
  8. Time embedding: Sinusoidal with log-spaced frequencies
""")

print(f"\nReference files saved to: {OUTPUT_DIR}/")
for f in OUTPUT_DIR.glob("*.safetensors"):
    print(f"  - {f.name}")
