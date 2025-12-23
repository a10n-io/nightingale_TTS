"""
Compare ODE solver output (before finalProj) between Python and Swift.
This is the intermediate state 'xt' before the final projection layer.
"""
import torch
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"

print("=" * 80)
print("COMPARE ODE OUTPUT (before finalProj)")
print("=" * 80)

# We need to instrument both Python and Swift to save ODE output
# For now, let's check if the files exist

python_ode_path = FORENSIC_DIR / "python_ode_output.safetensors"
swift_ode_path = FORENSIC_DIR / "swift_ode_output.safetensors"

if not python_ode_path.exists():
    print(f"\n❌ Python ODE output not found")
    print(f"\n   We need to instrument Python decoder to save ODE output.")
    print(f"   The ODE output is the state 'xt' before finalProj is applied.")
    print(f"\n   In Python code, this is likely:")
    print(f"   - In decoder.solve_euler() or decoder.basic_euler()")
    print(f"   - The output of the ODE solver before estimator.final_proj()")
else:
    print(f"✅ Python ODE output found")

if not swift_ode_path.exists():
    print(f"\n❌ Swift ODE output not found")
    print(f"\n   We need to add forensic output in S3Gen.swift")
    print(f"   In generateMelFromTokens(), before finalProj")
    print(f"   Save the 'h' tensor right before: h = finalProj(h)")
else:
    print(f"✅ Swift ODE output found")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("\nTo find where the 0.91 dB divergence originates:")
print("\n1. Instrument Python decoder to save ODE output (xt) before finalProj")
print("2. Swift already saves this in debug mode - check if it's being saved")
print("3. Compare the ODE outputs:")
print("   - If they match → issue is in finalProj application (but weights match!)")
print("   - If they differ → issue is in ODE solver or conditioning")
print("\n4. If ODE outputs differ, trace backwards:")
print("   - Compare encoder outputs")
print("   - Compare conditioning (cond_enc output)")
print("   - Compare ODE velocity field at each step")
print("   - Check CFG application")
print("\n" + "=" * 80)
