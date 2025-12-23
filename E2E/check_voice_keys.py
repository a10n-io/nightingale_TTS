"""Quick check of baked voice keys."""
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
VOICE_DIR = PROJECT_ROOT / "baked_voices" / "samantha"

voice_data = load_file(str(VOICE_DIR / "baked_voice.safetensors"))
print("Keys in baked_voice.safetensors:")
for key in voice_data.keys():
    print(f"  {key}: {voice_data[key].shape}")
