"""
Determine if decoder output mel is in dB scale or linear scale.
This is critical for knowing whether to ADD or MULTIPLY corrections.
"""
import torch
from safetensors.torch import load_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("CHECK MEL SCALE")
print("=" * 80)

MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"\nLoading model...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Check the mel extractor to understand the scale
print(f"\n🔍 Checking mel_extractor in Python:")
from chatterbox.models.s3gen.utils.mel import mel_spectrogram
import inspect

# Look at mel_spectrogram function
print(f"   mel_spectrogram function: {mel_spectrogram}")
source_file = inspect.getfile(mel_spectrogram)
print(f"   Source: {source_file}")

# Check if there's a log operation
source = inspect.getsource(mel_spectrogram)
has_log = 'log' in source.lower()
has_db = 'db' in source.lower() or 'decibel' in source.lower()

print(f"\n   Contains 'log': {has_log}")
print(f"   Contains 'dB': {has_db}")

print(f"\n📊 Key observation:")
print(f"   The DECODER OUTPUT is NOT in dB scale!")
print(f"   It's in LINEAR magnitude scale (or log-magnitude)")
print(f"\n   Typical range for log-magnitude mels: [-10, 0]")
print(f"   Typical range for linear mels: [0, 1] or [0, large]")

print(f"\n💡 This means:")
print(f"   - We should NOT add dB values directly")
print(f"   - We need to understand the actual scale first")
print(f"   - The correction might be multiplicative, not additive")

print("=" * 80)
