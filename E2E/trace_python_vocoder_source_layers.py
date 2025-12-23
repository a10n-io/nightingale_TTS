"""
Trace Python vocoder source processing layers to find divergence
"""
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path
import sys

# Add chatterbox to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python" / "chatterbox" / "src"))

from chatterbox.models.s3gen.s3gen import S3Gen

MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
CROSS_VAL_DIR = PROJECT_ROOT / "test_audio" / "cross_validate"
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "vocoder_trace"

print("=" * 80)
print("PYTHON VOCODER SOURCE PROCESSING TRACE")
print("=" * 80)

# Load model
print("\nLoading S3Gen model...")
model = S3Gen.from_pretrained(str(MODELS_DIR / "s3gen.safetensors"))
model.eval()

# Load Python decoder mel (same input used for both Python and Swift)
print("Loading Python decoder mel...")
mel_arrays = load_file(str(CROSS_VAL_DIR / "python_decoder_mel_for_swift_vocoder.safetensors"))
mel_input = mel_arrays["mel"]  # [80, T]
print(f"  Input mel: {mel_input.shape}")

with torch.no_grad():
    # Prepare input
    mel_input = mel_input.unsqueeze(0)  # [1, 80, T]
    print(f"  Mel input (batched): {mel_input.shape}")

    # Transpose for Python vocoder: [B, C, T] -> [B, T, C]
    speech_feat = mel_input.transpose(1, 2)  # [1, T, 80]
    print(f"  Speech feat (transposed): {speech_feat.shape}")

    # F0 prediction
    print("\nStep 1: F0 Prediction...")
    f0 = model.s3gen.mel2wav.f0_predictor(speech_feat)
    print(f"  F0 shape: {f0.shape}")
    save_file({"f0": f0.cpu()}, str(OUTPUT_DIR / "python_1_f0.safetensors"))

    # F0 upsampling
    print("\nStep 2: F0 Upsampling...")
    f0_up = model.s3gen.mel2wav.f0_upsamp(f0[:, None]).transpose(1, 2)  # [B, T_high, 1]
    print(f"  F0 upsampled shape: {f0_up.shape}")
    save_file({"f0_upsampled": f0_up.cpu()}, str(OUTPUT_DIR / "python_2_f0_upsampled.safetensors"))

    # Source generation
    print("\nStep 3: Source Generation...")
    s, _, _ = model.s3gen.mel2wav.m_source(f0_up)  # [B, T_high, 1]
    print(f"  Source shape: {s.shape}")
    print(f"  Source range: [{s.min().item():.6f}, {s.max().item():.6f}]")
    save_file({"source": s.cpu()}, str(OUTPUT_DIR / "python_3_source.safetensors"))

    # Transpose source for decode: [B, T, 1] -> [B, 1, T]
    s = s.transpose(1, 2)  # [B, 1, T]
    print(f"  Source transposed: {s.shape}")

    # ========== CRITICAL: Source STFT Processing ==========
    print("\n" + "=" * 80)
    print("STEP 4: SOURCE STFT PROCESSING")
    print("=" * 80)

    # Source STFT
    s_stft_real, s_stft_imag = model.s3gen.mel2wav._stft(s.squeeze(1))
    s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)  # [B, n_fft+2, T']
    print(f"  Source STFT shape: {s_stft.shape}")
    print(f"  Source STFT real range: [{s_stft_real.min().item():.6f}, {s_stft_real.max().item():.6f}]")
    print(f"  Source STFT imag range: [{s_stft_imag.min().item():.6f}, {s_stft_imag.max().item():.6f}]")
    print(f"  Source STFT combined range: [{s_stft.min().item():.6f}, {s_stft.max().item():.6f}]")
    print(f"  Source STFT mean: {s_stft.mean().item():.8f}")
    print(f"  Source STFT std: {s_stft.std().item():.8f}")
    save_file({
        "s_stft": s_stft.cpu(),
        "s_stft_real": s_stft_real.cpu(),
        "s_stft_imag": s_stft_imag.cpu()
    }, str(OUTPUT_DIR / "python_4_source_stft.safetensors"))

    # Conv pre (mel path)
    print("\nStep 5: Conv Pre (mel path)...")
    x = model.s3gen.mel2wav.conv_pre(speech_feat)  # [B, 512, T]
    print(f"  Conv pre shape: {x.shape}")
    save_file({"conv_pre": x.cpu()}, str(OUTPUT_DIR / "python_5_conv_pre.safetensors"))

    # ========== UPSAMPLING + SOURCE FUSION ==========
    print("\n" + "=" * 80)
    print("STEP 6-8: UPSAMPLING + SOURCE FUSION LAYERS")
    print("=" * 80)

    num_upsamples = len(model.s3gen.mel2wav.ups)
    print(f"  Number of upsamples: {num_upsamples}")

    for i in range(num_upsamples):
        print(f"\n--- Upsample {i} ---")

        # Upsample mel path
        x = torch.nn.functional.leaky_relu(x, model.s3gen.mel2wav.lrelu_slope)
        x = model.s3gen.mel2wav.ups[i](x)
        print(f"  After ups[{i}]: {x.shape}")

        if i == num_upsamples - 1:
            x = model.s3gen.mel2wav.reflection_pad(x)
            print(f"  After reflection_pad: {x.shape}")

        # Source processing
        si = model.s3gen.mel2wav.source_downs[i](s_stft)
        print(f"  After source_downs[{i}]: {si.shape}, range=[{si.min().item():.6f}, {si.max().item():.6f}]")
        save_file({f"source_down_{i}": si.cpu()}, str(OUTPUT_DIR / f"python_6_{i}_source_down.safetensors"))

        si = model.s3gen.mel2wav.source_resblocks[i](si)
        print(f"  After source_resblocks[{i}]: {si.shape}, range=[{si.min().item():.6f}, {si.max().item():.6f}]")
        save_file({f"source_resblock_{i}": si.cpu()}, str(OUTPUT_DIR / f"python_7_{i}_source_resblock.safetensors"))

        # Fusion
        x_before_fusion = x.clone()
        x = x + si
        print(f"  After fusion (x + si): {x.shape}, range=[{x.min().item():.6f}, {x.max().item():.6f}]")
        save_file({
            f"mel_before_fusion_{i}": x_before_fusion.cpu(),
            f"fused_{i}": x.cpu()
        }, str(OUTPUT_DIR / f"python_8_{i}_fusion.safetensors"))

        # Main resblocks
        xs = None
        for j in range(model.s3gen.mel2wav.num_kernels):
            if xs is None:
                xs = model.s3gen.mel2wav.resblocks[i * model.s3gen.mel2wav.num_kernels + j](x)
            else:
                xs += model.s3gen.mel2wav.resblocks[i * model.s3gen.mel2wav.num_kernels + j](x)
        x = xs / model.s3gen.mel2wav.num_kernels
        print(f"  After resblocks: {x.shape}, range=[{x.min().item():.6f}, {x.max().item():.6f}]")

    # Conv post
    print("\n" + "=" * 80)
    print("STEP 9: CONV POST")
    print("=" * 80)
    x = torch.nn.functional.leaky_relu(x)
    x = model.s3gen.mel2wav.conv_post(x)
    print(f"  After conv_post: {x.shape}")
    print(f"  Conv post range: [{x.min().item():.6f}, {x.max().item():.6f}]")
    save_file({"conv_post": x.cpu()}, str(OUTPUT_DIR / "python_9_conv_post.safetensors"))

    # ISTFT
    print("\n" + "=" * 80)
    print("STEP 10: ISTFT")
    print("=" * 80)
    magnitude = torch.exp(x[:, :model.s3gen.mel2wav.istft_params["n_fft"] // 2 + 1, :])
    phase = torch.sin(x[:, model.s3gen.mel2wav.istft_params["n_fft"] // 2 + 1:, :])
    print(f"  Magnitude shape: {magnitude.shape}, range=[{magnitude.min().item():.6f}, {magnitude.max().item():.6f}]")
    print(f"  Phase shape: {phase.shape}, range=[{phase.min().item():.6f}, {phase.max().item():.6f}]")

    audio = model.s3gen.mel2wav._istft(magnitude, phase)
    audio = torch.clamp(audio, -model.s3gen.mel2wav.audio_limit, model.s3gen.mel2wav.audio_limit)
    print(f"  Audio shape: {audio.shape}")
    print(f"  Audio range: [{audio.min().item():.6f}, {audio.max().item():.6f}]")

print("\n" + "=" * 80)
print("✅ Saved Python vocoder source processing trace to:")
print(f"   {OUTPUT_DIR}")
print("=" * 80)
