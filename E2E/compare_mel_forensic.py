"""
Forensic comparison of Python vs Swift decoder mel output.
Direct numerical comparison for mathematical precision.
"""
import torch
from safetensors.torch import load_file
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"

print("=" * 80)
print("FORENSIC MEL COMPARISON - MATHEMATICAL PRECISION")
print("=" * 80)

# Load Python mel
python_path = FORENSIC_DIR / "python_mel_raw.safetensors"
if not python_path.exists():
    print(f"\n❌ Python mel not found: {python_path}")
    print(f"   Run: cd E2E && python save_python_mel_raw.py")
    exit(1)

python_data = load_file(str(python_path))
python_mel_gen = python_data["mel_gen"]

# Load Swift mel
swift_path = FORENSIC_DIR / "swift_mel_raw.safetensors"
if not swift_path.exists():
    print(f"\n❌ Swift mel not found: {swift_path}")
    print(f"   Run: cd swift && swift run SaveSwiftMelForensic")
    exit(1)

swift_data = load_file(str(swift_path))
swift_mel_full = swift_data["mel_gen"]

# Swift saved the full output (potentially multiple batches + prompt + generated)
# Python saved only generated region [1, 80, 196]
# Swift format: [B, 80, T] where T = prompt_frames + generated_frames
# We need to extract: batch 1 (python tokens), frames 500:696 (generated region)

print(f"\n✅ Loaded both mel spectrograms")
print(f"   Python generated: {python_mel_gen.shape}")
print(f"   Swift full: {swift_mel_full.shape}")

# Extract generated region from Swift
# Swift ran decoder twice: batch 0 = swift tokens, batch 1 = python tokens
# We want batch 1 (python tokens) for fair comparison
if swift_mel_full.shape[0] > 1:
    batch_idx = 1  # Python tokens (second run)
    print(f"\n   Using Swift batch {batch_idx} (python tokens)")
else:
    batch_idx = 0
    print(f"\n   Using Swift batch {batch_idx}")

# Extract generated region: frames 500:696 (196 frames)
# Prompt is 250 tokens * 2 = 500 frames, generated is 98 tokens * 2 = 196 frames
prompt_frames = 500
swift_mel_gen = swift_mel_full[batch_idx:batch_idx+1, :, prompt_frames:]

print(f"   Swift generated (extracted): {swift_mel_gen.shape}")

# === GENERATED REGION COMPARISON (Critical - this is what we hear) ===
print("\n" + "=" * 80)
print("GENERATED REGION COMPARISON (This is what causes mumbling)")
print("=" * 80)

print(f"\n📊 Python (generated):")
print(f"   Range: [{python_mel_gen.min().item():.8f}, {python_mel_gen.max().item():.8f}]")
print(f"   Mean: {python_mel_gen.mean().item():.8f}")
print(f"   Std: {python_mel_gen.std().item():.8f}")

print(f"\n📊 Swift (generated):")
print(f"   Range: [{swift_mel_gen.min().item():.8f}, {swift_mel_gen.max().item():.8f}]")
print(f"   Mean: {swift_mel_gen.mean().item():.8f}")
print(f"   Std: {swift_mel_gen.std().item():.8f}")

# Calculate differences for generated region
diff_gen = python_mel_gen - swift_mel_gen
abs_diff_gen = diff_gen.abs()

print(f"\n📊 Difference (Python - Swift):")
print(f"   Mean difference: {diff_gen.mean().item():.8f} dB")
print(f"   Abs mean diff: {abs_diff_gen.mean().item():.8f} dB")
print(f"   Max difference: {diff_gen.max().item():.8f} dB")
print(f"   Min difference: {diff_gen.min().item():.8f} dB")
print(f"   Std of diff: {diff_gen.std().item():.8f} dB")

# === PER-CHANNEL ANALYSIS ===
print("\n" + "=" * 80)
print("PER-CHANNEL ANALYSIS (Generated Region)")
print("=" * 80)

# Compute per-channel mean difference (averaged over time)
per_channel_diff = diff_gen.mean(dim=(0, 2))  # [80] channels

print(f"\n📊 Per-channel mean difference (Python - Swift):")
print(f"   Overall mean: {per_channel_diff.mean().item():.8f} dB")
print(f"   Channel range: [{per_channel_diff.min().item():.8f}, {per_channel_diff.max().item():.8f}] dB")
print(f"   Std across channels: {per_channel_diff.std().item():.8f} dB")

# Show some specific channels
print(f"\n   Sample channels:")
for i in [0, 20, 40, 60, 79]:
    ch_diff = per_channel_diff[i].item()
    py_ch_mean = python_mel_gen[0, i, :].mean().item()
    sw_ch_mean = swift_mel_gen[0, i, :].mean().item()
    print(f"     Ch{i:2d}: Python={py_ch_mean:8.5f}, Swift={sw_ch_mean:8.5f}, Diff={ch_diff:+8.5f} dB")

# === STATISTICAL TESTS ===
print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS")
print("=" * 80)

# Is the difference constant (bias) or variable (scaling)?
print(f"\n🔍 Checking if difference is constant bias or scaling issue:")
print(f"   Diff coefficient of variation: {(diff_gen.std() / diff_gen.mean().abs()).item():.6f}")
print(f"   (Low value ~0.1 = constant bias, High value >1.0 = scaling issue)")

# Correlation between magnitude and difference
python_flat = python_mel_gen.flatten()
diff_flat = diff_gen.flatten()
correlation = torch.corrcoef(torch.stack([python_flat, diff_flat]))[0, 1].item()
print(f"\n🔍 Correlation between Python mel value and difference:")
print(f"   Correlation: {correlation:.6f}")
print(f"   (Close to 0 = additive bias, Close to 1 = multiplicative scaling)")

# === RECOMMENDATIONS ===
print("\n" + "=" * 80)
print("RECOMMENDED CORRECTION")
print("=" * 80)

mean_correction = diff_gen.mean().item()
print(f"\n💡 PRECISE additive correction needed:")
print(f"   Swift mel should be increased by: {mean_correction:+.8f} dB")
print(f"\n📝 Implementation in S3Gen.swift (line ~1895):")
print(f"   After: h = finalProj(h)")
print(f"   Add: h = h + {mean_correction:.8f}")

# Check if clamping is needed
swift_corrected_max = swift_mel_gen.max().item() + mean_correction
if swift_corrected_max > 0:
    print(f"\n⚠️  After correction, max will be: {swift_corrected_max:.8f}")
    print(f"   This exceeds 0.0 - add clamping:")
    print(f"   h = minimum(h, 0.0)")
else:
    print(f"\n✅ After correction, max will be: {swift_corrected_max:.8f}")
    print(f"   Still negative - clamping may not be needed")

# Precision check
print(f"\n✅ Verification:")
if abs(mean_correction) < 0.01:
    print(f"   ✅ Difference is < 0.01 dB - EXCELLENT precision!")
elif abs(mean_correction) < 0.1:
    print(f"   ✅ Difference is < 0.1 dB - Good precision")
elif abs(mean_correction) < 0.5:
    print(f"   ⚠️  Difference is {abs(mean_correction):.3f} dB - Noticeable")
else:
    print(f"   ❌ Difference is {abs(mean_correction):.3f} dB - LARGE difference")

print("\n" + "=" * 80)
