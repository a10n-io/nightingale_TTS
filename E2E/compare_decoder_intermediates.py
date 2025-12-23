"""
Compare all decoder intermediate values between Python and Swift.
"""
import torch
from safetensors.torch import load_file
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"

print("=" * 80)
print("COMPARING DECODER INTERMEDIATE VALUES")
print("=" * 80)

# 1. Compare mu (encoder output)
python_mu = load_file(str(FORENSIC_DIR / "python_decoder_mu.safetensors"))["mu"]
swift_mu = load_file(str(FORENSIC_DIR / "swift_decoder_mu.safetensors"))["mu"]

print("\n1. MU (Encoder output):")
print(f"   Python: shape={python_mu.shape}, mean={python_mu.mean().item():.8f}")
print(f"   Swift:  shape={swift_mu.shape}, mean={swift_mu.mean().item():.8f}")

mu_corr = np.corrcoef(python_mu.flatten().numpy(), swift_mu.flatten().numpy())[0, 1]
print(f"   Correlation: {mu_corr:.10f}")

# 2. Compare spk (speaker embedding)
python_spk = load_file(str(FORENSIC_DIR / "python_decoder_spk.safetensors"))["spk"]
swift_spk = load_file(str(FORENSIC_DIR / "swift_decoder_spk.safetensors"))["spk"]

print("\n2. SPK (Speaker embedding projection):")
print(f"   Python: shape={python_spk.shape}, mean={python_spk.mean().item():.8f}")
print(f"   Swift:  shape={swift_spk.shape}, mean={swift_spk.mean().item():.8f}")
print(f"   Python first 5: {python_spk[0, :5].tolist()}")
print(f"   Swift first 5:  {swift_spk[0, :5].tolist()}")

spk_diff = (python_spk - swift_spk).abs().max().item()
print(f"   Max diff: {spk_diff:.10f}")

# 3. Compare noise
python_noise = load_file(str(FORENSIC_DIR / "python_decoder_noise.safetensors"))["noise"]
swift_noise = load_file(str(FORENSIC_DIR / "swift_decoder_noise.safetensors"))["noise"]

print("\n3. NOISE (Initial ODE noise):")
print(f"   Python: shape={python_noise.shape}, mean={python_noise.mean().item():.8f}")
print(f"   Swift:  shape={swift_noise.shape}, mean={swift_noise.mean().item():.8f}")
print(f"   Python first 5: {python_noise[0, 0, :5].tolist()}")
print(f"   Swift first 5:  {swift_noise[0, 0, :5].tolist()}")

noise_corr = np.corrcoef(python_noise.flatten().numpy(), swift_noise.flatten().numpy())[0, 1]
noise_diff = (python_noise - swift_noise).abs().max().item()
print(f"   Correlation: {noise_corr:.10f}")
print(f"   Max diff: {noise_diff:.10f}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if mu_corr > 0.9999:
    print(f"✅ MU: Perfect match (corr={mu_corr:.10f})")
elif mu_corr > 0.999:
    print(f"⚠️  MU: Very close (corr={mu_corr:.10f})")
else:
    print(f"❌ MU: Differs (corr={mu_corr:.10f})")

if spk_diff < 1e-6:
    print(f"✅ SPK: Perfect match (max_diff={spk_diff:.10e})")
elif spk_diff < 1e-3:
    print(f"⚠️  SPK: Very close (max_diff={spk_diff:.10e})")
else:
    print(f"❌ SPK: Differs (max_diff={spk_diff:.10e})")

if noise_corr > 0.9999 and noise_diff < 1e-6:
    print(f"✅ NOISE: Perfect match (corr={noise_corr:.10f}, max_diff={noise_diff:.10e})")
elif noise_corr > 0.5:
    print(f"⚠️  NOISE: Partial match (corr={noise_corr:.10f}, max_diff={noise_diff:.10e})")
else:
    print(f"❌ NOISE: Different RNG (corr={noise_corr:.10f}, max_diff={noise_diff:.10e})")

print("\n" + "=" * 80)
