"""
Compare Python and Swift at each stage to isolate the humming issue.
"""
import torch
from safetensors.torch import load_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("COMPARE FULL PIPELINE: Python vs Swift")
print("=" * 80)

MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load Python model
print(f"\nLoading Python model on {device}...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load Python mel (decoder output)
python_mel_path = PROJECT_ROOT / "test_audio/cross_validate/python_mel.safetensors"
python_mel_data = load_file(str(python_mel_path))
python_mel = python_mel_data["mel"].to(device)

print(f"\n📊 Python Decoder Mel:")
print(f"   Shape: {python_mel.shape}")
print(f"   Range: [{python_mel.min().item():.6f}, {python_mel.max().item():.6f}]")
print(f"   Mean: {python_mel.mean().item():.6f}, Std: {python_mel.std().item():.6f}")

# Run Python vocoder on Python mel
print(f"\n--- Running Python Vocoder on Python Mel ---")
with torch.no_grad():
    python_audio = model.s3gen.mel2wav(python_mel)

print(f"Python Audio Output:")
print(f"   Shape: {python_audio.shape}")
print(f"   Range: [{python_audio.min().item():.6f}, {python_audio.max().item():.6f}]")
print(f"   Mean: {python_audio.mean().item():.6f}, Std: {python_audio.std().item():.6f}")
print(f"   First 20 samples: {python_audio[0, :20].tolist()}")

# Check if Swift mel exists
swift_mel_path = PROJECT_ROOT / "test_audio/cross_validate/swift_mel.safetensors"
if swift_mel_path.exists():
    print(f"\n--- Comparing Swift Decoder Mel ---")
    swift_mel_data = load_file(str(swift_mel_path))
    swift_mel = swift_mel_data["mel"].to(device)

    print(f"\n📊 Swift Decoder Mel:")
    print(f"   Shape: {swift_mel.shape}")
    print(f"   Range: [{swift_mel.min().item():.6f}, {swift_mel.max().item():.6f}]")
    print(f"   Mean: {swift_mel.mean().item():.6f}, Std: {swift_mel.std().item():.6f}")

    # Compare mels
    diff = (python_mel - swift_mel).abs()
    print(f"\n📊 Mel Difference (Python - Swift):")
    print(f"   Mean abs diff: {diff.mean().item():.6f}")
    print(f"   Max abs diff: {diff.max().item():.6f}")
    print(f"   Relative diff: {(diff.mean() / python_mel.abs().mean()).item():.4%}")

    # Run Python vocoder on Swift mel
    print(f"\n--- Running Python Vocoder on Swift Mel ---")
    with torch.no_grad():
        swift_mel_python_audio = model.s3gen.mel2wav(swift_mel)

    print(f"Python Vocoder Output (Swift Mel):")
    print(f"   Shape: {swift_mel_python_audio.shape}")
    print(f"   Range: [{swift_mel_python_audio.min().item():.6f}, {swift_mel_python_audio.max().item():.6f}]")
    print(f"   Mean: {swift_mel_python_audio.mean().item():.6f}, Std: {swift_mel_python_audio.std().item():.6f}")

    # Save for listening
    from safetensors.torch import save_file
    import torchaudio

    # Save audio
    output_path = PROJECT_ROOT / "test_audio/cross_validate/swift_mel_python_vocoder.wav"
    torchaudio.save(str(output_path), swift_mel_python_audio.cpu(), 24000)
    print(f"\n✅ Saved Swift mel + Python vocoder audio to: swift_mel_python_vocoder.wav")

    if diff.max().item() > 0.1:
        print(f"\n⚠️  WARNING: Large difference in decoder mels!")
        print(f"   This suggests the decoder (not vocoder) is producing different output")
    else:
        print(f"\n✅ Decoder mels are similar (max diff < 0.1)")
else:
    print(f"\n⚠️  Swift mel not found. Please run SaveSwiftMel first.")

print("\n" + "=" * 80)
