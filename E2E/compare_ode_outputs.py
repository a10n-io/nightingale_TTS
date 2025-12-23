"""
Compare ODE outputs (before finalProj) between Python and Swift.
"""
import torch
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"

print("=" * 80)
print("COMPARE ODE OUTPUTS (Before finalProj)")
print("=" * 80)

# Load Python ODE output
python_path = FORENSIC_DIR / "python_ode_output.safetensors"
if not python_path.exists():
    print(f"\n❌ Python ODE output not found: {python_path}")
    exit(1)

python_data = load_file(str(python_path))
python_ode = python_data["ode_output"]

# Load Swift ODE output
swift_path = FORENSIC_DIR / "swift_ode_output.safetensors"
if not swift_path.exists():
    print(f"\n❌ Swift ODE output not found: {swift_path}")
    exit(1)

swift_data = load_file(str(swift_path))
swift_ode = swift_data["ode_output"]

print(f"\n✅ Loaded both ODE outputs")
print(f"   Python: {python_ode.shape}")
print(f"   Swift: {swift_ode.shape}")

# Check if transpose is needed
# Python: [B, C, T] = [2, 256, 696]
# Swift: [B, T, C] = [2, 696, 256]
if python_ode.shape != swift_ode.shape:
    if python_ode.shape[0] == swift_ode.shape[0] and python_ode.shape[1] == swift_ode.shape[2] and python_ode.shape[2] == swift_ode.shape[1]:
        print(f"\n📊 Transposing Swift ODE from [B, T, C] to [B, C, T]...")
        swift_ode = swift_ode.transpose(1, 2)
        print(f"   Swift transposed: {swift_ode.shape}")
    else:
        print(f"\n❌ Shape mismatch cannot be resolved!")
        exit(1)

# Extract the generated region (batch 1, frames 500:696)
# For Python tokens, we want batch index 1
batch_idx = 1
prompt_frames = 500

python_ode_gen = python_ode[batch_idx:batch_idx+1, :, prompt_frames:]
swift_ode_gen = swift_ode[batch_idx:batch_idx+1, :, prompt_frames:]

print(f"\n📊 Extracted generated region (python tokens, frames 500:696):")
print(f"   Python shape: {python_ode_gen.shape}")
print(f"   Swift shape: {swift_ode_gen.shape}")

# Statistics
print(f"\n📊 Python ODE (generated region):")
print(f"   Range: [{python_ode_gen.min().item():.8f}, {python_ode_gen.max().item():.8f}]")
print(f"   Mean: {python_ode_gen.mean().item():.8f}")
print(f"   Std: {python_ode_gen.std().item():.8f}")

print(f"\n📊 Swift ODE (generated region):")
print(f"   Range: [{swift_ode_gen.min().item():.8f}, {swift_ode_gen.max().item():.8f}]")
print(f"   Mean: {swift_ode_gen.mean().item():.8f}")
print(f"   Std: {swift_ode_gen.std().item():.8f}")

# Difference
diff = python_ode_gen - swift_ode_gen
abs_diff = diff.abs()

print(f"\n📊 Difference (Python - Swift):")
print(f"   Mean difference: {diff.mean().item():.8f}")
print(f"   Abs mean diff: {abs_diff.mean().item():.8f}")
print(f"   Max abs diff: {abs_diff.max().item():.8f}")
print(f"   Std of diff: {diff.std().item():.8f}")

# Check if ODE outputs match
if abs_diff.mean().item() < 1e-4:
    print(f"\n✅ ODE outputs MATCH (diff < 1e-4)")
    print(f"\n   → The 0.91 dB mel difference is NOT from ODE solver!")
    print(f"   → Issue must be in finalProj application or post-processing")
    print(f"\n   But wait - finalProj weights match perfectly...")
    print(f"   This suggests a transpose or dimension ordering issue!")
elif abs_diff.mean().item() < 1e-2:
    print(f"\n⚠️  Small ODE difference (1e-4 < diff < 1e-2)")
    print(f"\n   → ODE solver has minor numerical differences")
elif abs_diff.mean().item() < 0.1:
    print(f"\n⚠️  Moderate ODE difference (diff ~ {abs_diff.mean().item():.6f})")
    print(f"\n   → ODE solver producing different outputs")
else:
    print(f"\n❌ LARGE ODE DIFFERENCE (diff = {abs_diff.mean().item():.6f})")
    print(f"\n   → ODE solver has significant divergence")

# Correlation check
python_flat = python_ode_gen.flatten()
swift_flat = swift_ode_gen.flatten()
correlation = torch.corrcoef(torch.stack([python_flat, swift_flat]))[0, 1].item()

print(f"\n📊 Correlation between Python and Swift ODE outputs:")
print(f"   Correlation: {correlation:.8f}")
if correlation > 0.99:
    print(f"   ✅ Extremely high correlation - outputs are structurally very similar")
elif correlation > 0.95:
    print(f"   ✅ High correlation - outputs are similar")
else:
    print(f"   ⚠️  Low correlation - outputs differ structurally")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

# Now apply finalProj and check if the difference emerges
# We know finalProj weights match, so let's see what happens
from safetensors.torch import load_file as load_weights

weights_path = PROJECT_ROOT / "test_audio" / "forensic" / "python_finalproj_weights.safetensors"
if weights_path.exists():
    weights_data = load_weights(str(weights_path))
    weight = weights_data["final_proj_weight"]
    bias = weights_data["final_proj_bias"]

    print(f"\n🔬 Applying finalProj to ODE outputs...")

    # finalProj is Conv1d [80, 256, 1]
    # Input: [B, C, T] = [1, 256, 196]
    # Output: [B, 80, T] = [1, 80, 196]

    import torch.nn.functional as F

    # Apply conv1d
    python_mel = F.conv1d(python_ode_gen, weight, bias, padding=0)
    swift_mel = F.conv1d(swift_ode_gen, weight, bias, padding=0)

    print(f"\n📊 Python mel (after finalProj):")
    print(f"   Range: [{python_mel.min().item():.8f}, {python_mel.max().item():.8f}]")
    print(f"   Mean: {python_mel.mean().item():.8f}")

    print(f"\n📊 Swift mel (after finalProj):")
    print(f"   Range: [{swift_mel.min().item():.8f}, {swift_mel.max().item():.8f}]")
    print(f"   Mean: {swift_mel.mean().item():.8f}")

    mel_diff = python_mel - swift_mel
    print(f"\n📊 Mel difference (after applying SAME finalProj):")
    print(f"   Mean diff: {mel_diff.mean().item():.8f} dB")

    print(f"\n🔍 Key Finding:")
    if abs(mel_diff.mean().item()) < 0.01:
        print(f"   ✅ ODE difference carries through to mel")
        print(f"   → The 0.91 dB mel difference originates in the ODE solver")
    elif abs(mel_diff.mean().item() - 0.91) < 0.1:
        print(f"   ✅ THIS REPRODUCES THE 0.91 dB DIFFERENCE!")
        print(f"   → ODE outputs differ by ~{diff.mean().item():.6f}")
        print(f"   → This causes ~{mel_diff.mean().item():.3f} dB mel difference")
    else:
        print(f"   ⚠️  Difference doesn't match expected 0.91 dB")

print("\n" + "=" * 80)
