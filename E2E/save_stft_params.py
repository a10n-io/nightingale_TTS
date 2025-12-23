"""
Save magnitude and phase before ISTFT for comparison.
"""
import torch
from safetensors.torch import load_file, save_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("SAVE STFT PARAMETERS (before ISTFT)")
print("=" * 80)

MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model
print(f"\nLoading model on {device}...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load Python mel
mel_path = PROJECT_ROOT / "test_audio/cross_validate/python_mel.safetensors"
mel_data = load_file(str(mel_path))
mel = mel_data["mel"].to(device)

print(f"\n📊 Mel: {mel.shape}")

# Run vocoder up to conv_post to get STFT parameters
vocoder = model.s3gen.mel2wav
with torch.inference_mode():
    # F0 prediction
    f0 = vocoder.f0_predictor(mel)
    s = vocoder.f0_upsamp(f0[:, None]).transpose(1, 2)
    s, _, _ = vocoder.m_source(s)
    s = s.transpose(1, 2)

    # Source STFT
    s_stft_real, s_stft_imag = vocoder._stft(s.squeeze(1))
    s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

    # Main path
    x = vocoder.conv_pre(mel)
    for i in range(vocoder.num_upsamples):
        x = torch.nn.functional.leaky_relu(x, vocoder.lrelu_slope)
        x = vocoder.ups[i](x)

        if i == vocoder.num_upsamples - 1:
            x = vocoder.reflection_pad(x)

        # Source fusion
        si = vocoder.source_downs[i](s_stft)
        si = vocoder.source_resblocks[i](si)
        x = x + si

        # ResBlocks
        xs = None
        for j in range(vocoder.num_kernels):
            res_out = vocoder.resblocks[i * vocoder.num_kernels + j](x)
            xs = res_out if xs is None else xs + res_out
        x = xs / vocoder.num_kernels

    # Final conv_post
    x = torch.nn.functional.leaky_relu(x)
    x = vocoder.conv_post(x)  # [B, 18, T]

    print(f"\n📊 conv_post output (STFT params):")
    print(f"   Shape: {x.shape}")
    print(f"   Range: [{x.min().item():.6f}, {x.max().item():.6f}]")

    # Extract magnitude and phase
    n_fft_half = vocoder.istft_params["n_fft"] // 2 + 1  # 9
    magnitude = torch.exp(x[:, :n_fft_half, :])
    phase = torch.sin(x[:, n_fft_half:, :])

    print(f"\n📊 Magnitude:")
    print(f"   Shape: {magnitude.shape}")
    print(f"   Range: [{magnitude.min().item():.6f}, {magnitude.max().item():.6f}]")
    print(f"   Mean: {magnitude.mean().item():.6f}")

    print(f"\n📊 Phase (sin of network output):")
    print(f"   Shape: {phase.shape}")
    print(f"   Range: [{phase.min().item():.6f}, {phase.max().item():.6f}]")
    print(f"   Mean: {phase.mean().item():.6f}")

    # Save for Swift comparison
    output_dir = PROJECT_ROOT / "test_audio/cross_validate"
    save_file({
        "magnitude": magnitude.cpu(),
        "phase": phase.cpu(),
        "conv_post_output": x.cpu(),
    }, str(output_dir / "python_stft_params.safetensors"))

    print(f"\n✅ Saved STFT parameters to: python_stft_params.safetensors")

print("\n" + "=" * 80)
