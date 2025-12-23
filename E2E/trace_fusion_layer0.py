"""
Trace Python vocoder FUSION at Layer 0 to identify divergence point
This is the critical diagnostic to determine if the bug is in:
1. Main Path (ups[0])
2. Source Path (source_downs[0])
3. Alignment (x + s being misaligned)
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
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "fusion_trace"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("=" * 80)
print("PYTHON VOCODER FUSION LAYER 0 TRACE")
print("=" * 80)
print("Goal: Capture x (main path) and s (source path) BEFORE fusion")
print("=" * 80)

# Load model
print(f"\nLoading Python model on {device}...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load Python decoder mel
print("Loading Python decoder mel...")
mel_arrays = load_file(str(CROSS_VAL_DIR / "python_decoder_mel_for_swift_vocoder.safetensors"))
mel_input = mel_arrays["mel"].to(device)  # [80, T]

with torch.no_grad():
    # Prepare input matching hifigan.py forward() method
    # From hifigan.py line 451: speech_feat = batch['speech_feat'].transpose(1, 2).to(device)
    # So if batch has [1, T, 80], speech_feat becomes [1, 80, T] (PyTorch Conv1d format)
    speech_feat = mel_input.unsqueeze(0)  # [80, T] -> [1, 80, T]

    print(f"\nInput mel (raw): {mel_input.shape}")
    print(f"Speech feat [B,C,T]: {speech_feat.shape}")

    vocoder = model.s3gen.mel2wav

    # F0 prediction (expects [B, C, T] format for PyTorch Conv1d)
    print("\nStep 1: F0 Prediction...")
    f0 = vocoder.f0_predictor(speech_feat)
    print(f"  F0 shape: {f0.shape}")

    # F0 upsampling
    print("\nStep 2: F0 Upsampling...")
    f0_up = vocoder.f0_upsamp(f0[:, None]).transpose(1, 2)  # [B, T_high, 1]
    print(f"  F0 upsampled shape: {f0_up.shape}")

    # Source generation
    print("\nStep 3: Source Generation...")
    s, _, _ = vocoder.m_source(f0_up)  # [B, T_high, 1]
    s = s.transpose(1, 2)  # [B, 1, T_high]
    print(f"  Source shape: {s.shape}")
    print(f"  Source range: [{s.min().item():.6f}, {s.max().item():.6f}]")

    # ========== CRITICAL: Source STFT Processing ==========
    print("\n" + "=" * 80)
    print("STEP 4: SOURCE STFT PROCESSING")
    print("=" * 80)

    # Source STFT
    s_stft_real, s_stft_imag = vocoder._stft(s.squeeze(1))
    s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)  # [B, n_fft+2, T']
    print(f"  Source STFT shape: {s_stft.shape}")
    print(f"  Source STFT range: [{s_stft.min().item():.6f}, {s_stft.max().item():.6f}]")

    # Conv Pre (main path start)
    print("\nStep 5: Conv Pre (mel path)...")
    x = vocoder.conv_pre(speech_feat)  # [B, 512, T] in PyTorch format
    print(f"  Conv pre shape: {x.shape}")

    # ========== LAYER 0 FUSION TRACE ==========
    print("\n" + "=" * 80)
    print("LAYER 0 FUSION - THE CRITICAL DIAGNOSTIC")
    print("=" * 80)

    i = 0

    # 1. Main Path Upsample
    # Apply leaky ReLU
    x = torch.nn.functional.leaky_relu(x, vocoder.lrelu_slope)

    # Upsample
    x_up = vocoder.ups[i](x)

    print(f"\n🔍 PYTHON FUSION [Layer {i}]")
    print(f"  x_up (Main Path after ups[{i}]): shape={x_up.shape}")
    print(f"    mean={x_up.mean().item():.8f}")
    print(f"    std={x_up.std().item():.8f}")
    print(f"    range=[{x_up.min().item():.6f}, {x_up.max().item():.6f}]")
    print(f"    first 10: {x_up[0, :, 0].tolist()[:10]}")

    # 2. Source Path Downsample
    s_down = vocoder.source_downs[i](s_stft)

    print(f"\n  s_down (Source Path after source_downs[{i}]): shape={s_down.shape}")
    print(f"    mean={s_down.mean().item():.8f}")
    print(f"    std={s_down.std().item():.8f}")
    print(f"    range=[{s_down.min().item():.6f}, {s_down.max().item():.6f}]")
    print(f"    first 10: {s_down[0, :, 0].tolist()[:10]}")

    # Save for comparison with Swift
    save_file({
        "x_up": x_up.cpu().contiguous(),
        "s_down": s_down.cpu().contiguous()
    }, str(OUTPUT_DIR / "python_fusion_layer0_pre.safetensors"))

    # 3. The Fusion
    x_fused = x_up + s_down

    print(f"\n  x_fused (After x + s): shape={x_fused.shape}")
    print(f"    mean={x_fused.mean().item():.8f}")
    print(f"    std={x_fused.std().item():.8f}")
    print(f"    range=[{x_fused.min().item():.6f}, {x_fused.max().item():.6f}]")

    # 4. Source Resblock
    x_after_resblock = vocoder.source_resblocks[i](x_fused)

    print(f"\n  x_after_resblock (After source_resblocks[{i}]): shape={x_after_resblock.shape}")
    print(f"    mean={x_after_resblock.mean().item():.8f}")
    print(f"    std={x_after_resblock.std().item():.8f}")
    print(f"    range=[{x_after_resblock.min().item():.6f}, {x_after_resblock.max().item():.6f}]")

    save_file({
        "x_fused": x_fused.cpu().contiguous(),
        "x_after_resblock": x_after_resblock.cpu().contiguous()
    }, str(OUTPUT_DIR / "python_fusion_layer0_post.safetensors"))

    print("\n" + "=" * 80)
    print("✅ Saved Python Layer 0 fusion trace to:")
    print(f"   {OUTPUT_DIR}/python_fusion_layer0_pre.safetensors")
    print(f"   {OUTPUT_DIR}/python_fusion_layer0_post.safetensors")
    print("=" * 80)
    print("\n📋 NEXT STEP: Run Swift trace and compare:")
    print("   1. If x_up matches (1.0 corr) → Main path OK")
    print("   2. If s_down matches (1.0 corr) → Source path OK")
    print("   3. If both match but x_fused doesn't → ALIGNMENT BUG")
    print("   4. If s_down mismatches → Bug is in STFT or source_downs")
    print("   5. If x_up mismatches → Bug is in ups (ConvTranspose)")
    print("=" * 80)
