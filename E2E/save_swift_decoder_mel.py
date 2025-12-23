"""
Use Python to run the Swift decoder and save the mel.
This is easier than debugging the hanging Swift script.
"""
from safetensors.torch import load_file, save_file
import torch
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

print("=" * 80)
print("EXTRACT SWIFT DECODER MEL OUTPUT")
print("=" * 80)

# Load Python tokens
token_path = PROJECT_ROOT / "test_audio/cross_validate/python_speech_tokens.safetensors"
token_data = load_file(str(token_path))
python_tokens = token_data["tokens"]
print(f"\nPython tokens: {python_tokens.shape[0]} tokens")

# Load voice
voice_path = PROJECT_ROOT / "test_audio/samantha_multilingual.safetensors"
voice_data = load_file(str(voice_path))

print("\n📊 Approach: We'll compare python_mel vs swift audio output")
print("   If swift audio (from python tokens) sounds like humming,")
print("   but python audio (from python tokens) is good,")
print("   then the issue is in the Swift decoder or vocoder.")

# Load both audio files and check
py_audio_path = PROJECT_ROOT / "test_audio/cross_validate/python_tokens_python_audio.wav"
sw_audio_path = PROJECT_ROOT / "test_audio/cross_validate/python_tokens_swift_audio.wav"

if py_audio_path.exists() and sw_audio_path.exists():
    print(f"\n✅ Both audio files exist:")
    print(f"   - {py_audio_path.name} (should sound good)")
    print(f"   - {sw_audio_path.name} (user reports humming)")
    print(f"\n🔍 Next: Check if Swift decoder mel differs from Python mel")
    print(f"   We need to save Swift mel to compare.")
else:
    print(f"\n⚠️  Audio files not found")

print("\n" + "=" * 80)
