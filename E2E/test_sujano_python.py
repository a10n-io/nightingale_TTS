"""
Test sujano voice with Python to compare mel brightness.
"""
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("PYTHON SUJANO VOICE TEST")
print("=" * 80)

MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"\nLoading model on {device}...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load sujano voice prompt
sujano_ref = PROJECT_ROOT / "baked_voices/sujano/ref_audio.wav"
print(f"\nPreparing conditionals with reference: {sujano_ref}")
model.prepare_conditionals(audio_prompt_path=str(sujano_ref))

test_text = "Wow! I absolutely cannot believe that it worked on the first try!"
print(f"\nGenerating audio for: {test_text}")

# Generate with temperature matching Swift
audio = model.generate(test_text, language_id="eng", temperature=0.0001)

print(f"\n📊 Python Audio Output:")
print(f"   Shape: {audio.shape}")
print(f"   Range: [{audio.min().item():.6f}, {audio.max().item():.6f}]")
print(f"   Mean: {audio.mean().item():.6f}, Std: {audio.std().item():.6f}")

# Save for comparison
import torchaudio
output_path = PROJECT_ROOT / "test_audio/python_sujano_test.wav"
torchaudio.save(str(output_path), audio.cpu(), 24000)
print(f"\n✅ Saved: {output_path}")

print("\n" + "=" * 80)
