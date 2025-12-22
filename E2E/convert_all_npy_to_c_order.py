#!/usr/bin/env python3
"""Convert all .npy files to C-order for MLX-swift compatibility."""

import numpy as np
from pathlib import Path
import shutil

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
ref_dir = PROJECT_ROOT / "E2E" / "reference_outputs" / "samantha" / "expressive_surprise_en"

# Find all .npy files
npy_files = list(ref_dir.glob("*.npy"))

print(f"Found {len(npy_files)} .npy files")
print("Converting to C-order...")

converted = 0
already_c = 0
errors = 0

for npy_file in sorted(npy_files):
    try:
        # Load the data
        data = np.load(npy_file)

        # Check if it's already C-order
        is_c_order = data.flags['C_CONTIGUOUS']

        if is_c_order:
            print(f"  ✓ {npy_file.name} already C-order")
            already_c += 1
        else:
            print(f"  → {npy_file.name} converting from F-order...")
            # Convert to C-order
            data_c = np.ascontiguousarray(data)

            # Backup original
            backup_path = npy_file.with_suffix('.npy.f_order_backup')
            if not backup_path.exists():
                shutil.copy2(npy_file, backup_path)
                print(f"    Backed up to {backup_path.name}")

            # Save as C-order
            np.save(npy_file, data_c)

            # Verify
            reloaded = np.load(npy_file)
            if np.allclose(data, reloaded) and reloaded.flags['C_CONTIGUOUS']:
                print(f"    ✓ Converted and verified")
                converted += 1
            else:
                print(f"    ✗ Verification failed!")
                errors += 1
    except Exception as e:
        print(f"  ✗ {npy_file.name} error: {e}")
        errors += 1

print(f"\nSummary:")
print(f"  Already C-order: {already_c}")
print(f"  Converted: {converted}")
print(f"  Errors: {errors}")
print(f"\n✅ All files now C-order compatible with MLX-swift!")
