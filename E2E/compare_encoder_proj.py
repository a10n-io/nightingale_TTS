"""
Compare encoder_proj outputs between Python and Swift
Check for transpose issues by examining actual values
"""
import torch
import numpy as np
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"

# Load Swift encoder output
swift_data = load_file(str(FORENSIC_DIR / "swift_encoder_fixed.safetensors"))
swift_encoder_out = swift_data["encoder_output_before_proj"]  # [1, 696, 512]
swift_after_proj = swift_data["encoder_output_after_proj"]  # [1, 696, 80]

# Load Python data
python_encoder_out = load_file(str(FORENSIC_DIR / "python_encoder_before_proj.safetensors"))["encoder_before_proj"]
python_after_proj = load_file(str(FORENSIC_DIR / "python_encoder_output.safetensors"))["encoder_output"]

def compute_correlation(a, b):
    a_flat = a.flatten().numpy()
    b_flat = b.flatten().numpy()
    return np.corrcoef(a_flat, b_flat)[0, 1]

print("=" * 80)
print("ENCODER_PROJ OUTPUT COMPARISON")
print("=" * 80)

print("\nINPUT to encoder_proj (encoder output before proj):")
corr_input = compute_correlation(python_encoder_out, swift_encoder_out)
print(f"  Correlation: {corr_input:.6f}")
print(f"  Python: mean={python_encoder_out.mean().item():.6f}, std={python_encoder_out.std().item():.6f}")
print(f"  Swift:  mean={swift_encoder_out.mean().item():.6f}, std={swift_encoder_out.std().item():.6f}")

print("\nOUTPUT from encoder_proj:")
print(f"  Python shape: {python_after_proj.shape}")
print(f"  Swift shape:  {swift_after_proj.shape}")
print(f"  Python: mean={python_after_proj.mean().item():.6f}, std={python_after_proj.std().item():.6f}")
print(f"  Swift:  mean={swift_after_proj.mean().item():.6f}, std={swift_after_proj.std().item():.6f}")
print(f"  Python range: [{python_after_proj.min().item():.6f}, {python_after_proj.max().item():.6f}]")
print(f"  Swift range:  [{swift_after_proj.min().item():.6f}, {swift_after_proj.max().item():.6f}]")

corr = compute_correlation(python_after_proj, swift_after_proj)
print(f"\n📊 CORRELATION: {corr:.6f}")

if corr > 0.99:
    print("✅ PERFECT! encoder_proj outputs match.")
elif corr > 0.9:
    print("✅ Good correlation")
else:
    print("❌ Poor correlation - checking for transpose issue...")
    
    # Check first few VALUES
    print("\n" + "=" * 80)
    print("VALUE CHECK (first token, first 5 features):")
    print("=" * 80)
    print(f"  Python [0, 0, :5]: {python_after_proj[0, 0, :5].tolist()}")
    print(f"  Swift  [0, 0, :5]: {swift_after_proj[0, 0, :5].tolist()}")
    
    print("\nVALUE CHECK (first 5 tokens, first feature):")
    print(f"  Python [0, :5, 0]: {python_after_proj[0, :5, 0].tolist()}")
    print(f"  Swift  [0, :5, 0]: {swift_after_proj[0, :5, 0].tolist()}")
    
    # Check if dimensions are swapped
    if python_after_proj.shape != swift_after_proj.shape:
        print(f"\n⚠️  Shape mismatch: Python {python_after_proj.shape} vs Swift {swift_after_proj.shape}")
    
    # Check transpose
    python_transposed = python_after_proj.transpose(1, 2)  # [1, 80, 696]
    corr_transposed = compute_correlation(python_transposed, swift_after_proj)
    print(f"\n🔍 Correlation with Python TRANSPOSED [1,80,696]: {corr_transposed:.6f}")
    
    if corr_transposed > corr:
        print("⚠️  WARNING: Output dimensions may be swapped!")
