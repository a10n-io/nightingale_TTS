"""
Trace Python vocoder step-by-step to compare with Swift.
"""
import torch
from safetensors.torch import load_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("PYTHON VOCODER TRACE")
print("=" * 80)

MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# Load model
print("\nLoading ChatterboxMultilingualTTS...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load Python mel from cross-validation
mel_path = PROJECT_ROOT / "test_audio/cross_validate/python_mel.safetensors"
mel_data = load_file(str(mel_path))
mel = mel_data["mel"].to(device)  # Should be [B, 80, T]

print(f"\n📊 Input Mel:")
print(f"   Shape: {mel.shape}")
print(f"   Range: [{mel.min().item():.6f}, {mel.max().item():.6f}]")
print(f"   Mean: {mel.mean().item():.6f}, Std: {mel.std().item():.6f}")

# Get vocoder
vocoder = model.s3gen.mel2wav
print(f"\n🔍 Vocoder type: {type(vocoder).__name__}")

# Trace through vocoder
with torch.no_grad():
    # Step 0: F0 prediction
    print(f"\n--- Step 0: F0 Prediction ---")
    f0 = vocoder.f0_predictor(mel)
    print(f"   F0 shape: {f0.shape}")
    print(f"   F0 range: [{f0.min().item():.6f}, {f0.max().item():.6f}]")
    print(f"   F0 mean: {f0.mean().item():.6f}")

    # Step 0b: F0 -> Source
    print(f"\n--- Step 0b: F0 -> Source ---")
    s = vocoder.f0_upsamp(f0[:, None]).transpose(1, 2)
    print(f"   After f0_upsamp: {s.shape}")
    s, _, _ = vocoder.m_source(s)
    s = s.transpose(1, 2)
    print(f"   After m_source: {s.shape}")
    print(f"   Source range: [{s.min().item():.6f}, {s.max().item():.6f}]")
    print(f"   Source mean: {s.mean().item():.6f}")

    # Step 0c: Source STFT
    print(f"\n--- Step 0c: Source STFT ---")
    s_stft_real, s_stft_imag = vocoder._stft(s.squeeze(1))
    s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)
    print(f"   s_stft shape: {s_stft.shape}")
    print(f"   s_stft range: [{s_stft.min().item():.6f}, {s_stft.max().item():.6f}]")

    # Step 1: conv_pre on mel
    x = mel
    print(f"\n--- Step 1: conv_pre ---")
    x = vocoder.conv_pre(x)
    print(f"   Shape: {x.shape}")
    print(f"   Range: [{x.min().item():.6f}, {x.max().item():.6f}]")
    print(f"   Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")

    # Upsampling stages
    for i in range(vocoder.num_upsamples):
        print(f"\n--- Upsampling Stage {i} ---")
        x = torch.nn.functional.leaky_relu(x, vocoder.lrelu_slope)

        print(f"   Before ups[{i}]: {x.shape}")
        x = vocoder.ups[i](x)
        print(f"   After ups[{i}]: {x.shape}")
        print(f"   Range: [{x.min().item():.6f}, {x.max().item():.6f}]")

        if i == vocoder.num_upsamples - 1:
            x = vocoder.reflection_pad(x)
            print(f"   After reflection_pad: {x.shape}")

        # Source fusion
        si = vocoder.source_downs[i](s_stft)
        si = vocoder.source_resblocks[i](si)
        print(f"   Source fusion add (should be ~zeros): mean={si.mean().item():.6f}")
        x = x + si

        # ResBlocks
        xs = None
        for j in range(vocoder.num_kernels):
            res_out = vocoder.resblocks[i * vocoder.num_kernels + j](x)
            xs = res_out if xs is None else xs + res_out
        x = xs / vocoder.num_kernels
        print(f"   After ResBlocks: {x.shape}")
        print(f"   Range: [{x.min().item():.6f}, {x.max().item():.6f}]")
        print(f"   Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")

    # Final conv
    print(f"\n--- Step 2: conv_post ---")
    x = torch.nn.functional.leaky_relu(x)
    x = vocoder.conv_post(x)
    print(f"   Shape: {x.shape}")
    print(f"   Expected: [B, n_fft+2, T] = [B, 18, T]")
    print(f"   Range: [{x.min().item():.6f}, {x.max().item():.6f}]")

    # Extract magnitude and phase
    print(f"\n--- Step 3: Extract magnitude and phase ---")
    n_fft_half = vocoder.istft_params["n_fft"] // 2 + 1  # 9
    magnitude = torch.exp(x[:, :n_fft_half, :])
    phase = torch.sin(x[:, n_fft_half:, :])
    print(f"   Magnitude shape: {magnitude.shape}")
    print(f"   Magnitude range: [{magnitude.min().item():.6f}, {magnitude.max().item():.6f}]")
    print(f"   Phase shape: {phase.shape}")
    print(f"   Phase range: [{phase.min().item():.6f}, {phase.max().item():.6f}]")

    # ISTFT
    print(f"\n--- Step 4: ISTFT ---")
    print(f"   ISTFT params: {vocoder.istft_params}")
    x = vocoder._istft(magnitude, phase)
    print(f"   Shape: {x.shape}")
    print(f"   Range: [{x.min().item():.6f}, {x.max().item():.6f}]")
    print(f"   Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")

    # Clamp
    x = torch.clamp(x, -vocoder.audio_limit, vocoder.audio_limit)
    print(f"\n--- Final Audio ---")
    print(f"   Shape: {x.shape}")
    print(f"   Range: [{x.min().item():.6f}, {x.max().item():.6f}]")
    print(f"   Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")
    print(f"   First 20 samples: {x[0, :20].tolist()}")

print("\n" + "=" * 80)
