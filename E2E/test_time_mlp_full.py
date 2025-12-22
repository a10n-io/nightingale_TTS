import torch
import numpy as np
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from pathlib import Path

print("=" * 60)
print("🐍 PYTHON - Full TimeMLP Test")
print("=" * 60)

# Load model
model = ChatterboxMultilingualTTS.from_local(Path('/Users/a10n/Projects/nightingale_TTS/models/chatterbox'), device='cpu')
decoder = model.s3gen.flow.decoder.estimator

# Test with t=0.5
t = torch.tensor([0.5], dtype=torch.float32)

# Run through time_embeddings (sinusoidal)
t_emb = decoder.time_embeddings(t)
print(f"After sinusoidal embedding:")
print(f"  Shape: {t_emb.shape}")
print(f"  Mean: {t_emb.mean().item():.6f}")
print(f"  Std: {t_emb.std().item():.6f}")
print(f"  Range: [{t_emb.min().item():.4f}, {t_emb.max().item():.4f}]")
print(f"  [:5]: {t_emb[0, :5].tolist()}")

# Run through time_mlp
t_mlp_out = decoder.time_mlp(t_emb)
print(f"\nAfter time_mlp (Linear→SiLU→Linear):")
print(f"  Shape: {t_mlp_out.shape}")
print(f"  Mean: {t_mlp_out.mean().item():.6f}")
print(f"  Std: {t_mlp_out.std().item():.6f}")
print(f"  Range: [{t_mlp_out.min().item():.4f}, {t_mlp_out.max().item():.4f}]")
print(f"  [:5]: {t_mlp_out[0, :5].tolist()}")
print("=" * 60)
