"""
Test if source signals match when we remove random phase initialization
"""
import torch
from safetensors.torch import load_file
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "python" / "chatterbox" / "src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
CROSS_VAL_DIR = PROJECT_ROOT / "test_audio" / "cross_validate"

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading model...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load Python's decoder mel
python_decoder_mel = load_file(str(CROSS_VAL_DIR / "python_decoder_mel_for_swift_vocoder.safetensors"))["mel"]
python_decoder_mel = python_decoder_mel.to(device)

# Prepare for vocoder
mel_input = python_decoder_mel.unsqueeze(0).transpose(1, 2)  # [80, T] -> [1, T, 80]

print("\nRunning vocoder 3 times with random phases...")
sources = []
for i in range(3):
    with torch.no_grad():
        speech_feat = mel_input.transpose(1, 2)  # [1, T, 80] -> [1, 80, T]
        f0 = model.s3gen.mel2wav.f0_predictor(speech_feat)
        f0_up = model.s3gen.mel2wav.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = model.s3gen.mel2wav.m_source(f0_up)
        sources.append(s.squeeze().cpu().numpy())

# Compare correlation between runs
corr_01 = np.corrcoef(sources[0], sources[1])[0, 1]
corr_02 = np.corrcoef(sources[0], sources[2])[0, 1]
corr_12 = np.corrcoef(sources[1], sources[2])[0, 1]

print(f"\nCorrelations between different runs (with random phases):")
print(f"  Run 0 vs Run 1: {corr_01:.6f}")
print(f"  Run 0 vs Run 2: {corr_02:.6f}")
print(f"  Run 1 vs Run 2: {corr_12:.6f}")

print(f"\nMean source value across runs:")
for i, s in enumerate(sources):
    print(f"  Run {i}: mean={s.mean():.6f}, std={s.std():.6f}")

print("\n" + "=" * 80)
if corr_01 < 0.5:
    print("✅ CONFIRMED: Random phases make source non-deterministic")
    print("   Each run produces different source signals due to random phase init")
    print("   This is EXPECTED behavior!")
    print("\n   The 0.0045 Swift vs Python correlation suggests:")
    print("   1. Different random seeds")
    print("   2. OR implementation bug in sine generation/linear combination")
else:
    print("⚠️  UNEXPECTED: Source signals correlate despite random phases")
    print("   This suggests phases might be seeded or cached")
print("=" * 80)
