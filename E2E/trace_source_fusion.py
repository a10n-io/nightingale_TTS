"""
Trace source fusion processing in detail.
"""
import torch
from safetensors.torch import load_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("TRACE SOURCE FUSION PROCESSING")
print("=" * 80)

MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load mel
mel_path = PROJECT_ROOT / "test_audio/cross_validate/python_mel.safetensors"
mel_data = load_file(str(mel_path))
mel = mel_data["mel"].to(device)

vocoder = model.s3gen.mel2wav

with torch.no_grad():
    # Get source STFT
    f0 = vocoder.f0_predictor(mel)
    s = vocoder.f0_upsamp(f0[:, None]).transpose(1, 2)
    s, _, _ = vocoder.m_source(s)
    s = s.transpose(1, 2)
    s_stft_real, s_stft_imag = vocoder._stft(s.squeeze(1))
    s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

    print(f"\n📊 s_stft:")
    print(f"   Shape: {s_stft.shape}")
    print(f"   Range: [{s_stft.min().item():.6f}, {s_stft.max().item():.6f}]")
    print(f"   Mean: {s_stft.mean().item():.6f}")

    # Process through source_downs[0] and source_resblocks[0]
    print(f"\n--- Source Stage 0 Processing ---")
    si = s_stft  # [B, 18, T]
    print(f"Input to source_downs[0]: shape={si.shape}, mean={si.mean().item():.6f}")

    si = vocoder.source_downs[0](si)
    print(f"After source_downs[0]: shape={si.shape}, mean={si.mean().item():.6f}, range=[{si.min().item():.6f}, {si.max().item():.6f}]")

    si = vocoder.source_resblocks[0](si)
    print(f"After source_resblocks[0]: shape={si.shape}, mean={si.mean().item():.6f}, range=[{si.min().item():.6f}, {si.max().item():.6f}]")

print("\n" + "=" * 80)
