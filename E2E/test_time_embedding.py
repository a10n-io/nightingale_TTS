import torch
import numpy as np

print("=" * 60)
print("🐍 PYTHON GOLDEN VALUES - Time Embedding Test")
print("=" * 60)

# 1. Setup Deterministic Inputs
t_val = 0.5 
t = torch.tensor([t_val], dtype=torch.float32).reshape(1)

# 2. Run the Embedding (Manually, to see inside)
model_dim = 320  # From decoder init: in_channels
half_dim = model_dim // 2
scale = 1000.0  # The default in SinusoidalPosEmb

# Replicate the exact math:
emb_scale = np.log(10000) / (half_dim - 1)
emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb_scale)
raw_input = scale * t.unsqueeze(1) * emb.unsqueeze(0) 
raw_sin = raw_input.sin()
raw_cos = raw_input.cos()
features = torch.cat((raw_sin, raw_cos), dim=-1)

# 3. Print the "Golden Fingerprint"
print(f"t_input:       {t.item():.6f}")
print(f"scale_factor:  {scale}")
print(f"Raw Input [0]: {raw_input[0, 0].item():.6f} (Should be ~0.5 * 1000 * 1 = 500)")
print(f"Raw Input [-1]:{raw_input[0, -1].item():.6f} (Should be small)")
print(f"Sin Feature[0]:{raw_sin[0, 0].item():.6f}")
print(f"Cos Feature[0]:{raw_cos[0, 0].item():.6f}")
print(f"Features shape: {features.shape}")
print(f"Features[0, :5]: {features[0, :5].tolist()}")
print(f"Features[0, 160:165]: {features[0, 160:165].tolist()}")
print(f"Features mean: {features.mean().item():.6f}")
print(f"Features std:  {features.std().item():.6f} (Target: ~0.707)")
print("=" * 60)
