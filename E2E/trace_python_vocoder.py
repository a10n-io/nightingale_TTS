"""Trace Python vocoder intermediate outputs."""
import sys
sys.path.insert(0, "/Users/a10n/Projects/nightingale_TTS/python/chatterbox/src")

import numpy as np
import torch
from pathlib import Path

OUTPUT_DIR = Path("/Users/a10n/Projects/nightingale_TTS/E2E/reference_outputs/samantha/expressive_surprise_en")

print("Loading model...")
from chatterbox.tts import ChatterboxTTS
model = ChatterboxTTS.from_pretrained(device="mps")

# Load test mel
mel = np.load(OUTPUT_DIR / "test_mel_BCT.npy")
mel_tensor = torch.from_numpy(mel).to("mps")
print(f"Test mel shape: {mel_tensor.shape}")

# Get vocoder
vocoder = model.s3gen.mel2wav

# Trace through vocoder
with torch.no_grad():
    print("\n--- Tracing Python vocoder ---")
    
    # 1. F0 prediction
    f0 = vocoder.f0_predictor(mel_tensor)  # [B, 80, T] -> [B, T]
    print(f"F0: shape={f0.shape}, range=[{f0.min():.6f}, {f0.max():.6f}]")
    print(f"F0 first 20: {f0[0, :20].cpu().numpy()}")
    
    # 2. F0 upsampling
    f0_up = vocoder.f0_upsamp(f0[:, None]).transpose(1, 2)  # [B, T_high, 1]
    print(f"F0 upsampled: shape={f0_up.shape}")
    
    # 3. Source generation
    s, noise, uv = vocoder.m_source(f0_up)
    print(f"Source: shape={s.shape}, range=[{s.min():.6f}, {s.max():.6f}]")
    print(f"Source first 20: {s[0, :20, 0].cpu().numpy()}")
    
    # Check mSource linear weights
    m_linear_w = vocoder.m_source.l_linear.weight
    m_linear_b = vocoder.m_source.l_linear.bias if vocoder.m_source.l_linear.bias is not None else None
    print(f"m_source.l_linear.weight: shape={m_linear_w.shape}")
    print(f"  values: {m_linear_w.squeeze().cpu().numpy()}")
    if m_linear_b is not None:
        print(f"m_source.l_linear.bias: {m_linear_b.cpu().numpy()}")
    
    # 4. Conv pre output
    # Note: In Python, decode() is called with mel transposed [B, C, T]
    # First transpose mel from [B, C, T] to [B, T, C] for comparison
    mel_for_conv = mel_tensor  # Actually, Python expects [B, 80, T] for conv_pre
    x = vocoder.conv_pre(mel_for_conv)
    print(f"After conv_pre: shape={x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")
    
    # Check conv_pre weight shape
    print(f"conv_pre.weight.shape: {vocoder.conv_pre.weight.shape}")
    
    # 5. Full inference
    wav, _ = vocoder.inference(mel_tensor)
    print(f"\nFinal wav: shape={wav.shape}, range=[{wav.min():.4f}, {wav.max():.4f}]")

print("\nDone!")
