"""
Save the EXACT tensor that enters source_downs[0] in Python
This is the real/imag concatenation of the source STFT
"""
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path
import sys

# Add chatterbox to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python" / "chatterbox" / "src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
CROSS_VAL_DIR = PROJECT_ROOT / "test_audio" / "cross_validate"
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "stft_dump"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("=" * 80)
print("PYTHON: SAVE STFT INPUT TO source_downs[0]")
print("=" * 80)

# Load model
print(f"\nLoading Python model on {device}...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load Python decoder mel
print("Loading Python decoder mel...")
mel_arrays = load_file(str(CROSS_VAL_DIR / "python_decoder_mel_for_swift_vocoder.safetensors"))
mel_input = mel_arrays["mel"].to(device)  # [80, T]

with torch.no_grad():
    speech_feat = mel_input.unsqueeze(0)  # [80, T] -> [1, 80, T]
    vocoder = model.s3gen.mel2wav

    # F0 prediction
    f0 = vocoder.f0_predictor(speech_feat)

    # F0 upsampling
    f0_up = vocoder.f0_upsamp(f0[:, None]).transpose(1, 2)  # [B, T_high, 1]

    # Source generation
    s, _, _ = vocoder.m_source(f0_up)  # [B, T_high, 1]
    s = s.transpose(1, 2)  # [B, 1, T_high]

    print(f"\nSource signal shape: {s.shape}")
    print(f"Source signal range: [{s.min().item():.6f}, {s.max().item():.6f}]")

    # ========== CRITICAL: RAW STFT OUTPUT ==========
    print("\n" + "=" * 80)
    print("CAPTURING RAW STFT OUTPUT (INPUT TO source_downs[0])")
    print("=" * 80)

    # Get raw STFT
    s_stft_real, s_stft_imag = vocoder._stft(s.squeeze(1))

    # This is the EXACT concatenation that enters source_downs[0]
    s_stft_concat = torch.cat([s_stft_real, s_stft_imag], dim=1)  # [B, n_fft+2, T']

    print(f"\n🔍 PYTHON STFT INPUT TO source_downs[0]:")
    print(f"  Shape: {s_stft_concat.shape}")
    print(f"  Real part (first half) mean: {s_stft_real.mean().item():.8f}")
    print(f"  Imag part (second half) mean: {s_stft_imag.mean().item():.8f}")
    print(f"  Real part range: [{s_stft_real.min().item():.6f}, {s_stft_real.max().item():.6f}]")
    print(f"  Imag part range: [{s_stft_imag.min().item():.6f}, {s_stft_imag.max().item():.6f}]")

    # Print first 10 values of first channel (Real bin 0)
    print(f"\n  Real channel 0, first 10 time steps: {s_stft_real[0, 0, :10].tolist()}")
    print(f"  Imag channel 0, first 10 time steps: {s_stft_imag[0, 0, :10].tolist()}")

    # Save the exact tensor
    save_file({
        "s_stft_real": s_stft_real.cpu().contiguous(),
        "s_stft_imag": s_stft_imag.cpu().contiguous(),
        "s_stft_concat": s_stft_concat.cpu().contiguous(),
        "source_signal": s.cpu().contiguous()
    }, str(OUTPUT_DIR / "python_stft_input.safetensors"))

    print("\n" + "=" * 80)
    print(f"✅ Saved to: {OUTPUT_DIR}/python_stft_input.safetensors")
    print("=" * 80)
    print("\n📋 NEXT: Run Swift script to save the same tensor and compare")
    print("=" * 80)
