"""
Cross-validate vocoder: Feed Python's decoder mel to both Python and Swift vocoders
and verify they produce identical audio.
"""
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "python" / "chatterbox" / "src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "cross_validate"
OUTPUT_DIR.mkdir(exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("=" * 80)
print("VOCODER CROSS-VALIDATION TEST")
print("=" * 80)
print("\nTest: Feed Python's decoder mel to both Python and Swift vocoders")
print("Expected: Both should produce identical audio (correlation ≈ 1.0)")
print("=" * 80)

# Load Python model
print(f"\nLoading Python model on {device}...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load Python's decoder mel output
python_decoder_mel = load_file(str(FORENSIC_DIR / "python_decoder_mel.safetensors"))["mel"]
python_decoder_mel = python_decoder_mel.to(device)

print(f"\nPython decoder mel:")
print(f"  Shape: {python_decoder_mel.shape}")
print(f"  Mean: {python_decoder_mel.mean().item():.6f}")
print(f"  Std: {python_decoder_mel.std().item():.6f}")

# Run Python vocoder on Python mel
print("\n1. Running PYTHON vocoder on Python decoder mel...")
with torch.no_grad():
    mel_input = python_decoder_mel.unsqueeze(0).transpose(1, 2)  # [80, T] -> [1, T, 80]
    batch = {"speech_feat": mel_input}
    python_audio, _ = model.s3gen.mel2wav(batch, device=device)
    python_audio = python_audio.squeeze(0).squeeze(0)

    print(f"   Python vocoder output: {python_audio.shape}")
    print(f"   Mean: {python_audio.mean().item():.8f}")
    print(f"   Std: {python_audio.std().item():.8f}")

    # Save for Swift to process
    save_file(
        {"mel": python_decoder_mel.cpu().contiguous()},
        str(OUTPUT_DIR / "python_decoder_mel_for_swift_vocoder.safetensors")
    )
    print(f"   Saved Python mel to: {OUTPUT_DIR}/python_decoder_mel_for_swift_vocoder.safetensors")

    # Also save Python's audio output
    save_file(
        {"audio": python_audio.cpu().contiguous()},
        str(FORENSIC_DIR / "python_mel_python_vocoder.safetensors")
    )
    print(f"   Saved Python audio to: forensic/python_mel_python_vocoder.safetensors")

print("\n2. Now run Swift vocoder on the same Python mel using SaveVocoderCrossValidation.swift")
print("   (Creating Swift script...)")
