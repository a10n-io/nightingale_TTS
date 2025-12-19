#!/usr/bin/env swift

import Foundation
import MLX
import MLXNN
import MLXRandom

// SURGICAL REWRITE: Clean room implementation to find the (1, 80, 64) bug
// Focus: Speaker embedding projection and decoder input preparation

print("=== SURGICAL REWRITE: Speaker Embedding Test ===\n")

// 1. Load the actual weight from the model
print("Step 1: Loading spk_embed_affine_layer weight from model...")
let modelPath = "models/chatterbox/s3gen.pt"

// We'll manually load using Python to inspect
let checkWeightScript = """
import torch
import sys

state = torch.load('\(modelPath)', map_location='cpu')
weight = state['flow.spk_embed_affine_layer.weight']
bias = state['flow.spk_embed_affine_layer.bias']

print(f"weight.shape: {list(weight.shape)}")
print(f"bias.shape: {list(bias.shape)}")
print(f"PyTorch stores as [out_features, in_features] = [{weight.shape[0]}, {weight.shape[1]}]")
print(f"So this is Linear(in={weight.shape[1]}, out={weight.shape[0]})")
print(f"Expected: Linear(in=192, out=80)")

# Save as numpy for Swift to load
import numpy as np
weight_np = weight.cpu().numpy()
bias_np = bias.cpu().numpy()

print(f"\\nSaving to test_spk_weight.npy and test_spk_bias.npy...")
np.save('/tmp/test_spk_weight.npy', weight_np)
np.save('/tmp/test_spk_bias.npy', bias_np)
print(f"Saved. Weight shape: {weight_np.shape}, Bias shape: {bias_np.shape}")
"""

// Run Python to extract weights
let pythonPath = "python/venv/bin/python"
let process = Process()
process.executableURL = URL(fileURLWithPath: pythonPath)
process.arguments = ["-c", checkWeightScript]
try! process.run()
process.waitUntilExit()

print("\nStep 2: Loading extracted weights into Swift...")
// Load the weight
import NPYLoader  // Assuming we have this
// Actually, let me just use MLX to create test data

// 2. Create test inputs with KNOWN shapes
print("\nStep 3: Creating test inputs with explicit shapes...")
let testSpeakerEmb192 = MLXArray(repeating: 0.5, [1, 192])  // Raw 192-dim embedding
print("  testSpeakerEmb192.shape: \(testSpeakerEmb192.shape)")

// 3. Create the Linear layer
print("\nStep 4: Creating Linear layer...")
print("  Initializing: Linear(192, 80)")
let spkAffine = Linear(192, 80, bias: true)
print("  spkAffine.weight.shape: \(spkAffine.weight.shape)")
print("  spkAffine.bias.shape: \(spkAffine.bias.shape)")

// 4. MANUAL FORWARD PASS - No hidden operations
print("\nStep 5: Manual forward pass (no function call)...")
print("  Input: testSpeakerEmb192 = \(testSpeakerEmb192.shape)")
print("  Weight: \(spkAffine.weight.shape)")
print("  Bias: \(spkAffine.bias.shape)")

// Normalize first (like the real code does)
let norm = sqrt(sum(testSpeakerEmb192 * testSpeakerEmb192, axis: 1, keepDims: true)) + 1e-8
let normalized = testSpeakerEmb192 / norm
print("  After normalization: \(normalized.shape)")

// Manual matmul
print("\n  Doing manual matmul: normalized @ weight.T + bias")
let weightT = spkAffine.weight.T  // Transpose weight
print("  weightT.shape: \(weightT.shape)")
let matmulResult = matmul(normalized, weightT)
print("  matmul result shape: \(matmulResult.shape)")
let withBias = matmulResult + spkAffine.bias
print("  After adding bias: \(withBias.shape)")

// Compare with Linear layer's forward pass
print("\n  Now calling spkAffine(normalized) to compare...")
let linearResult = spkAffine(normalized)
print("  Linear forward result: \(linearResult.shape)")

// 5. Check if shapes match expectations
print("\n=== SHAPE VERIFICATION ===")
if withBias.shape == [1, 80] {
    print("✅ Manual matmul produces correct shape [1, 80]")
} else {
    print("❌ Manual matmul produces WRONG shape \(withBias.shape), expected [1, 80]")
}

if linearResult.shape == [1, 80] {
    print("✅ Linear forward produces correct shape [1, 80]")
} else {
    print("❌ Linear forward produces WRONG shape \(linearResult.shape), expected [1, 80]")
}

// 6. THE SMOKING GUN TEST: Check if weight has wrong shape
print("\n=== SMOKING GUN TEST ===")
print("Checking if weight accidentally has dimensions swapped...")
if spkAffine.weight.shape == [192, 80] {
    print("✅ Weight has correct shape [192, 80] (MLX Linear format)")
} else if spkAffine.weight.shape == [80, 192] {
    print("🚨 BUG FOUND: Weight has PyTorch format [80, 192]!")
    print("   MLX Linear expects [in_features, out_features] = [192, 80]")
    print("   But got [80, 192] which is PyTorch format")
} else {
    print("❓ Weight has unexpected shape: \(spkAffine.weight.shape)")
}

// 7. Test what happens if we accidentally use the WRONG dimension
print("\n=== REPRODUCE THE BUG ===")
print("Testing: What if we accidentally pass melChannels (80) as input dim?")
let wrongLinear = Linear(80, 64, bias: false)  // WRONG: using 80 instead of 192
print("  wrongLinear.weight.shape: \(wrongLinear.weight.shape)")

let testInput80 = MLXArray(repeating: 0.5, [1, 80])
print("  testInput80.shape: \(testInput80.shape)")

let wrongResult = wrongLinear(testInput80)
print("  wrongLinear(testInput80).shape: \(wrongResult.shape)")

if wrongResult.shape == [1, 64] {
    print("  Result is [1, 64] - this could create problems if reshaped!")
}

// 8. Test decoder input preparation
print("\n=== DECODER INPUT SIMULATION ===")
let L = 564  // Sequence length
print("Simulating decoder input with L=\(L)...")

let spkCond = MLXArray(repeating: 0.5, [1, 80])  // Speaker condition [1, 80]
print("  spkCond.shape: \(spkCond.shape)")

// Expand to [1, 80, 1] then tile to [1, 80, L]
let spkExpanded = spkCond.expandedDimensions(axis: 2)
print("  After expandDims(axis=2): \(spkExpanded.shape)")

let spkTiled = tiled(spkExpanded, repetitions: [1, 1, L])
print("  After tiling to [1, 1, \(L)]: \(spkTiled.shape)")

if spkTiled.shape == [1, 80, L] {
    print("✅ Speaker embedding correctly tiled to [1, 80, \(L)]")
} else {
    print("❌ Speaker embedding has WRONG shape: \(spkTiled.shape)")
}

print("\n=== TEST COMPLETE ===")
print("If you see a broadcast error, the bug is in the code above.")
print("Check which operation created a (1, 80, 64) tensor.")
