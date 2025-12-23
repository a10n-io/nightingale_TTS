"""
Trace Python vocoder layer by layer and save intermediate outputs
"""
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "python" / "chatterbox" / "src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
CROSS_VAL_DIR = PROJECT_ROOT / "test_audio" / "cross_validate"
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "vocoder_trace"
OUTPUT_DIR.mkdir(exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("=" * 80)
print("PYTHON VOCODER LAYER-BY-LAYER TRACE")
print("=" * 80)

# Load model
print(f"\nLoading Python model on {device}...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load Python's decoder mel output (same input used for Swift)
python_decoder_mel = load_file(str(CROSS_VAL_DIR / "python_decoder_mel_for_swift_vocoder.safetensors"))["mel"]
python_decoder_mel = python_decoder_mel.to(device)

print(f"\nInput mel: {python_decoder_mel.shape} = [80, T]")

# Prepare input for vocoder
mel_input = python_decoder_mel.unsqueeze(0).transpose(1, 2)  # [80, T] -> [1, T, 80]
print(f"Vocoder input: {mel_input.shape} = [B, T, C]")

# Access the vocoder
vocoder = model.s3gen.mel2wav

# Manually trace through vocoder layers
with torch.no_grad():
    # Step 0: Transpose to [B, C, T] for PyTorch Conv1d
    speech_feat = mel_input.transpose(1, 2)  # [1, T, 80] -> [1, 80, T]
    print(f"\nStep 0: Transposed to PyTorch format: {speech_feat.shape} = [B, C, T]")
    save_file({"speech_feat": speech_feat.cpu()}, str(OUTPUT_DIR / "python_0_input.safetensors"))

    # Step 1: F0 Prediction
    f0 = vocoder.f0_predictor(speech_feat)  # [1, T]
    print(f"\nStep 1: F0 prediction: {f0.shape}")
    print(f"  F0 range: [{f0.min().item():.6f}, {f0.max().item():.6f}]")
    print(f"  F0 first 10: {f0[0, :10].tolist()}")
    save_file({"f0": f0.cpu()}, str(OUTPUT_DIR / "python_1_f0.safetensors"))

    # Step 2: F0 Upsampling
    f0_upsampled = vocoder.f0_upsamp(f0[:, None])  # [1, 1, T] -> [1, 1, T_high]
    f0_upsampled = f0_upsampled.transpose(1, 2)  # [1, 1, T_high] -> [1, T_high, 1]
    print(f"\nStep 2: F0 upsampled: {f0_upsampled.shape}")
    print(f"  F0_up range: [{f0_upsampled.min().item():.6f}, {f0_upsampled.max().item():.6f}]")
    save_file({"f0_upsampled": f0_upsampled.cpu()}, str(OUTPUT_DIR / "python_2_f0_upsampled.safetensors"))

    # Step 3: Source Generation
    s, _, _ = vocoder.m_source(f0_upsampled)
    print(f"\nStep 3: Source signal: {s.shape}")
    print(f"  Source range: [{s.min().item():.6f}, {s.max().item():.6f}]")
    print(f"  Source first 10: {s[0, :10, 0].tolist()}")
    save_file({"source": s.cpu()}, str(OUTPUT_DIR / "python_3_source.safetensors"))

    # Step 4: Source transpose for decode
    s = s.transpose(1, 2)  # [1, T_high, 1] -> [1, 1, T_high]
    print(f"\nStep 4: Source transposed: {s.shape}")
    save_file({"source_transposed": s.cpu()}, str(OUTPUT_DIR / "python_4_source_transposed.safetensors"))

    # Step 5: Conv Pre
    x = vocoder.conv_pre(speech_feat)  # [1, 80, T] -> [1, 512, T]
    print(f"\nStep 5: Conv pre: {x.shape}")
    print(f"  Conv pre range: [{x.min().item():.6f}, {x.max().item():.6f}]")
    save_file({"conv_pre": x.cpu()}, str(OUTPUT_DIR / "python_5_conv_pre.safetensors"))

    print("\n" + "=" * 80)
    print("✅ Saved Python vocoder intermediate outputs to:")
    print(f"   {OUTPUT_DIR}")
    print("=" * 80)
