"""
Compare Swift vocoder output with Python vocoder output using the same mel.
"""
import torch
from safetensors.torch import load_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import soundfile as sf
import numpy as np

print("=" * 80)
print("COMPARE SWIFT vs PYTHON VOCODER")
print("=" * 80)

MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model
print(f"\nLoading ChatterboxMultilingualTTS on {device}...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load Python mel
mel_path = PROJECT_ROOT / "test_audio/cross_validate/python_mel.safetensors"
mel_data = load_file(str(mel_path))
mel = mel_data["mel"].to(device)

print(f"\n📊 Python Mel:")
print(f"   Shape: {mel.shape}")
print(f"   Range: [{mel.min().item():.6f}, {mel.max().item():.6f}]")

# Run Python vocoder
print(f"\n🔍 Running Python vocoder...")
vocoder = model.s3gen.mel2wav
with torch.inference_mode():
    audio_python, _ = vocoder.inference(mel)

print(f"\n📊 Python vocoder output:")
print(f"   Shape: {audio_python.shape}")
print(f"   Range: [{audio_python.min().item():.6f}, {audio_python.max().item():.6f}]")
print(f"   Mean: {audio_python.mean().item():.6f}, Std: {audio_python.std().item():.6f}")
print(f"   First 20: {audio_python[0, :20].tolist()}")

# Save Python audio for comparison
audio_python_np = audio_python.squeeze(0).cpu().numpy()
output_dir = PROJECT_ROOT / "test_audio/cross_validate"
python_voc_path = output_dir / "python_mel_python_vocoder.wav"
sf.write(str(python_voc_path), audio_python_np, 24000)
print(f"\n✅ Saved Python vocoder audio to: {python_voc_path}")

# Now load the Swift audio (from VocoderTest or similar)
# We'll need to run Swift first to generate this
print(f"\n⏳ Waiting for Swift vocoder audio...")
print(f"   Run: swift run -c release VocoderShapeTest")
print(f"   Then we can compare the outputs")

print("\n" + "=" * 80)
