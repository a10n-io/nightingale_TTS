import Foundation
import MLX
import MLXRandom
import MLXNN
import Nightingale

let PROJECT_ROOT = "/Users/a10n/Projects/nightingale_TTS"
let modelDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/models/chatterbox")
let refDir = "\(PROJECT_ROOT)/E2E/reference_outputs/samantha/expressive_surprise_en"

print("=" + String(repeating: "=", count: 79))
print("BLOCK1 DETAILED VERIFICATION")
print("=" + String(repeating: "=", count: 79))

// Helper to load .npy files
func loadNpy(_ path: String) throws -> MLXArray {
    return try NPYLoader.load(contentsOf: URL(fileURLWithPath: path))
}

// Load Python reference intermediate values
print("\n📥 Loading Python reference values...")
let pyHConcatload = try loadNpy("\(refDir)/detailed_step2_h_concat.npy")         // [1, 320, 696]
let pyHMasked = try loadNpy("\(refDir)/detailed_step5a_h_masked.npy")            // [1, 320, 696]
let pyAfterConv1 = try loadNpy("\(refDir)/detailed_step5b_after_conv1.npy")      // [1, 256, 696]
let pyAfterNorm1 = try loadNpy("\(refDir)/detailed_step5c_after_norm1.npy")      // [1, 256, 696]
let pyBlock1Out = try loadNpy("\(refDir)/detailed_step5d_block1_output.npy")     // [1, 256, 696]
let mask = try loadNpy("\(refDir)/step7_cond_T.npy")                              // [1, 1, 696]

eval(pyHConcatload, pyHMasked, pyAfterConv1, pyAfterNorm1, pyBlock1Out, mask)
print("✅ Loaded Python reference values")

// Load decoder weights
print("\n📦 Loading decoder weights...")
let decoderURL = modelDir.appendingPathComponent("decoder_weights.safetensors")
let weights = try MLX.loadArrays(url: decoderURL)
print("  Loaded \(weights.count) tensors")

// Create CausalBlock1D manually
print("\n🔧 Creating CausalBlock1D...")
let block1 = CausalBlock1D(dim: 320, dimOut: 256)

// Load weights manually
let convWRaw = weights["s3gen.flow.decoder.estimator.down_blocks.0.0.block1.block.0.weight"]!
let convB = weights["s3gen.flow.decoder.estimator.down_blocks.0.0.block1.block.0.bias"]!
let normW = weights["s3gen.flow.decoder.estimator.down_blocks.0.0.block1.block.2.weight"]!
let normB = weights["s3gen.flow.decoder.estimator.down_blocks.0.0.block1.block.2.bias"]!

// CRITICAL: Transpose Conv1d weights from PyTorch [out, in, kernel] to MLX [out, kernel, in]
let convW = convWRaw.swappedAxes(1, 2)

print("  Conv weight (before transpose): \(convWRaw.shape)")
print("  Conv weight (after transpose): \(convW.shape)")
print("  Conv bias: \(convB.shape)")
print("  Norm weight: \(normW.shape)")
print("  Norm bias: \(normB.shape)")

// Update block1 weights
// CausalConv1d has nested conv, so weights need "conv." prefix
block1.conv.update(parameters: ModuleParameters.unflattened(["conv.weight": convW, "conv.bias": convB]))
block1.norm.update(parameters: ModuleParameters.unflattened(["weight": normW, "bias": normB]))
print("✅ Block1 weights loaded")

// Test 1: Check h_masked
print("\n" + String(repeating: "=", count: 80))
print("TEST 1: H * MASK")
print(String(repeating: "=", count: 80))

let swiftHMasked = pyHConcatload * mask
eval(swiftHMasked)
print("Python h*mask: [\(pyHMasked.min().item(Float.self)), \(pyHMasked.max().item(Float.self))]")
print("Swift  h*mask: [\(swiftHMasked.min().item(Float.self)), \(swiftHMasked.max().item(Float.self))]")
let diffMasked = swiftHMasked - pyHMasked
eval(diffMasked)
print("Difference:    max abs = \(diffMasked.abs().max().item(Float.self))")

// Test 2: Check conv output
print("\n" + String(repeating: "=", count: 80))
print("TEST 2: AFTER CONV")
print(String(repeating: "=", count: 80))

let swiftAfterConv = block1.conv(swiftHMasked)
eval(swiftAfterConv)
print("Python after conv: [\(pyAfterConv1.min().item(Float.self)), \(pyAfterConv1.max().item(Float.self))]")
print("Swift  after conv: [\(swiftAfterConv.min().item(Float.self)), \(swiftAfterConv.max().item(Float.self))]")
let diffConv = swiftAfterConv - pyAfterConv1
eval(diffConv)
print("Difference:        max abs = \(diffConv.abs().max().item(Float.self))")

// Test 3: Check norm output
print("\n" + String(repeating: "=", count: 80))
print("TEST 3: AFTER NORM")
print(String(repeating: "=", count: 80))

// Transpose for norm
var swiftAfterNorm = swiftAfterConv.transposed(0, 2, 1)  // [B, T, C]
swiftAfterNorm = block1.norm(swiftAfterNorm)
swiftAfterNorm = swiftAfterNorm.transposed(0, 2, 1)  // [B, C, T]
eval(swiftAfterNorm)
print("Python after norm: [\(pyAfterNorm1.min().item(Float.self)), \(pyAfterNorm1.max().item(Float.self))]")
print("Swift  after norm: [\(swiftAfterNorm.min().item(Float.self)), \(swiftAfterNorm.max().item(Float.self))]")
let diffNorm = swiftAfterNorm - pyAfterNorm1
eval(diffNorm)
print("Difference:        max abs = \(diffNorm.abs().max().item(Float.self))")

// Test 4: Check mish output
print("\n" + String(repeating: "=", count: 80))
print("TEST 4: AFTER MISH")
print(String(repeating: "=", count: 80))

let swiftAfterMish = mish(swiftAfterNorm)
eval(swiftAfterMish)
print("Swift after mish: [\(swiftAfterMish.min().item(Float.self)), \(swiftAfterMish.max().item(Float.self))]")

// Test 5: Check final output (after mask)
print("\n" + String(repeating: "=", count: 80))
print("TEST 5: FINAL OUTPUT (AFTER MASK)")
print(String(repeating: "=", count: 80))

let swiftBlock1Out = swiftAfterMish * mask
eval(swiftBlock1Out)
print("Python block1 out: [\(pyBlock1Out.min().item(Float.self)), \(pyBlock1Out.max().item(Float.self))]")
print("Swift  block1 out: [\(swiftBlock1Out.min().item(Float.self)), \(swiftBlock1Out.max().item(Float.self))]")
let diffFinal = swiftBlock1Out - pyBlock1Out
eval(diffFinal)
print("Difference:        max abs = \(diffFinal.abs().max().item(Float.self))")
let rmse = sqrt((diffFinal * diffFinal).mean()).item(Float.self)
print("RMSE:              \(rmse)")

print("\n" + String(repeating: "=", count: 80))
if rmse < 0.01 {
    print("✅ PERFECT MATCH!")
} else if rmse < 0.1 {
    print("✅ GOOD MATCH (RMSE < 0.1)")
} else {
    print("❌ MISMATCH FOUND!")
}
print(String(repeating: "=", count: 80))
