"""
Save Python vocoder output (audio waveform) for comparison with Swift
"""
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "python" / "chatterbox" / "src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "forensic"
OUTPUT_DIR.mkdir(exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading Python model on {device}...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load the decoder mel output from Python
decoder_mel = load_file(str(FORENSIC_DIR / "python_decoder_mel.safetensors"))["mel"]
decoder_mel = decoder_mel.to(device)

print(f"\nPython decoder mel input to vocoder:")
print(f"  Shape: {decoder_mel.shape}")
print(f"  Mean: {decoder_mel.mean().item():.6f}")
print(f"  Std: {decoder_mel.std().item():.6f}")
print(f"  Range: [{decoder_mel.min().item():.6f}, {decoder_mel.max().item():.6f}]")

# Run vocoder
with torch.no_grad():
    # mel2wav expects a batch dict with 'speech_feat' in [B, T, C] format
    mel_input = decoder_mel.unsqueeze(0).transpose(1, 2)  # [80, T] -> [1, T, 80]
    batch = {"speech_feat": mel_input}
    audio, f0 = model.s3gen.mel2wav(batch, device=device)

    # Remove batch dimension
    audio = audio.squeeze(0).squeeze(0)  # [T_audio]

    print(f"\nPython vocoder audio output:")
    print(f"  Shape: {audio.shape}")
    print(f"  Mean: {audio.mean().item():.8f}")
    print(f"  Std: {audio.std().item():.8f}")
    print(f"  Range: [{audio.min().item():.8f}, {audio.max().item():.8f}]")
    print(f"  Sample rate: 24000 Hz")
    print(f"  Duration: {audio.shape[0] / 24000:.2f} seconds")

    # Save
    save_file(
        {"audio": audio.cpu().contiguous()},
        str(OUTPUT_DIR / "python_vocoder_audio.safetensors")
    )
    print(f"\n✅ Saved to: {OUTPUT_DIR}/python_vocoder_audio.safetensors")
