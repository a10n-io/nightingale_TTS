import Foundation
import MLX
import MLXNN
import Nightingale

let PROJECT_ROOT = "/Users/a10n/Projects/nightingale_TTS"
let modelDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/models/chatterbox")
let refDir = "\(PROJECT_ROOT)/E2E/reference_outputs/samantha/expressive_surprise_en"

print("=" + String(repeating: "=", count: 79))
print("MANUAL LAYERNORM COMPUTATION TEST")
print("=" + String(repeating: "=", count: 79))

// Helper to load .npy files
func loadNpy(_ path: String) throws -> MLXArray {
    return try NPYLoader.load(contentsOf: URL(fileURLWithPath: path))
}

// Load Python reference
let pyInput = try loadNpy("\(refDir)/tfmr_trace_input.npy")  // [1, 696, 256]
let pyAfterNorm1 = try loadNpy("\(refDir)/tfmr_trace_after_norm1.npy")
let pyMeanRef = try loadNpy("\(refDir)/tfmr_ln_mean.npy")
let pyVarRef = try loadNpy("\(refDir)/tfmr_ln_var.npy")

// Load weights
let decoderURL = modelDir.appendingPathComponent("decoder_weights.safetensors")
let weightsRaw = try MLX.loadArrays(url: decoderURL)
let norm1Weight = weightsRaw["s3gen.flow.decoder.estimator.down_blocks.0.1.0.norm1.weight"]!
let norm1Bias = weightsRaw["s3gen.flow.decoder.estimator.down_blocks.0.1.0.norm1.bias"]!

eval(pyInput, pyMeanRef, pyVarRef, norm1Weight, norm1Bias)
print("\n📥 Loaded data:")
print("  Input: \(pyInput.shape), range=[\(pyInput.min().item(Float.self)), \(pyInput.max().item(Float.self))]")
print("  Python mean (ref): \(pyMeanRef.shape), range=[\(pyMeanRef.min().item(Float.self)), \(pyMeanRef.max().item(Float.self))]")
print("  Python var (ref): \(pyVarRef.shape), range=[\(pyVarRef.min().item(Float.self)), \(pyVarRef.max().item(Float.self))]")
print("  norm1.weight: [\(norm1Weight.min().item(Float.self)), \(norm1Weight.max().item(Float.self))], mean=\(MLX.mean(norm1Weight).item(Float.self))")
print("  norm1.bias: [\(norm1Bias.min().item(Float.self)), \(norm1Bias.max().item(Float.self))], mean=\(MLX.mean(norm1Bias).item(Float.self))")

// Test 1: Manual LayerNorm computation
print("\n" + String(repeating: "=", count: 80))
print("TEST 1: MANUAL LAYERNORM COMPUTATION")
print(String(repeating: "=", count: 80))

let eps: Float = 1e-5

// Compute mean over last dimension - test different axis specifications
print("\nTesting different axis specifications:")
let mean_neg1 = MLX.mean(pyInput, axis: -1, keepDims: true)
eval(mean_neg1)
print("  Swift mean (axis=-1): \(mean_neg1.shape), range=[\(mean_neg1.min().item(Float.self)), \(mean_neg1.max().item(Float.self))]")

let mean_2 = MLX.mean(pyInput, axis: 2, keepDims: true)
eval(mean_2)
print("  Swift mean (axis=2): \(mean_2.shape), range=[\(mean_2.min().item(Float.self)), \(mean_2.max().item(Float.self))]")

print("  Python mean (ref): \(pyMeanRef.shape), range=[\(pyMeanRef.min().item(Float.self)), \(pyMeanRef.max().item(Float.self))]")

// Check which matches
let diff_neg1 = mean_neg1 - pyMeanRef
let diff_2 = mean_2 - pyMeanRef
eval(diff_neg1, diff_2)
print("  Diff with axis=-1: max abs = \(diff_neg1.abs().max().item(Float.self))")
print("  Diff with axis=2: max abs = \(diff_2.abs().max().item(Float.self))")

let mean = mean_2  // Use whichever matches better

// Compute variance over last dimension with different axis specifications
print("\nTesting variance with different axis specifications:")
let var_neg1_ddof0 = MLX.variance(pyInput, axis: -1, keepDims: true, ddof: 0)
eval(var_neg1_ddof0)
print("  Swift var (axis=-1, ddof=0): range=[\(var_neg1_ddof0.min().item(Float.self)), \(var_neg1_ddof0.max().item(Float.self))]")

let var_2_ddof0 = MLX.variance(pyInput, axis: 2, keepDims: true, ddof: 0)
eval(var_2_ddof0)
print("  Swift var (axis=2, ddof=0): range=[\(var_2_ddof0.min().item(Float.self)), \(var_2_ddof0.max().item(Float.self))]")

print("  Python var (ref): range=[\(pyVarRef.min().item(Float.self)), \(pyVarRef.max().item(Float.self))]")

// Check which matches
let var_diff_neg1 = var_neg1_ddof0 - pyVarRef
let var_diff_2 = var_2_ddof0 - pyVarRef
eval(var_diff_neg1, var_diff_2)
print("  Diff with axis=-1: max abs = \(var_diff_neg1.abs().max().item(Float.self))")
print("  Diff with axis=2: max abs = \(var_diff_2.abs().max().item(Float.self))")

let variance = var_2_ddof0  // Use whichever matches better

// Use ddof=0 for normalization (should match PyTorch unbiased=False)
let normalized = (pyInput - mean) / MLX.sqrt(variance + eps)
eval(normalized)
print("\nNormalized (before weight/bias): \(normalized.shape), range=[\(normalized.min().item(Float.self)), \(normalized.max().item(Float.self))]")

// Apply affine transform
let output = normalized * norm1Weight + norm1Bias
eval(output)
print("After affine (weight * x + bias): \(output.shape), range=[\(output.min().item(Float.self)), \(output.max().item(Float.self))]")

print("\nComparison:")
print("  Python output:  [\(pyAfterNorm1.min().item(Float.self)), \(pyAfterNorm1.max().item(Float.self))]")
print("  Manual output:  [\(output.min().item(Float.self)), \(output.max().item(Float.self))]")
let diff = output - pyAfterNorm1
eval(diff)
print("  Difference:     max abs = \(diff.abs().max().item(Float.self))")

// Test 2: MLX LayerNorm with loaded weights
print("\n" + String(repeating: "=", count: 80))
print("TEST 2: MLX LAYERNORM MODULE")
print(String(repeating: "=", count: 80))

let ln = LayerNorm(dimensions: 256, eps: 1e-5)
ln.update(parameters: ModuleParameters.unflattened(["weight": norm1Weight, "bias": norm1Bias]))

// Verify weights loaded
if let w = ln.weight, let b = ln.bias {
    eval(w, b)
    print("Loaded weights:")
    print("  weight: [\(w.min().item(Float.self)), \(w.max().item(Float.self))], mean=\(MLX.mean(w).item(Float.self))")
    print("  bias: [\(b.min().item(Float.self)), \(b.max().item(Float.self))], mean=\(MLX.mean(b).item(Float.self))")
}

let outputLN = ln(pyInput)
eval(outputLN)
print("\nMLX LayerNorm output: [\(outputLN.min().item(Float.self)), \(outputLN.max().item(Float.self))]")
print("Python output:        [\(pyAfterNorm1.min().item(Float.self)), \(pyAfterNorm1.max().item(Float.self))]")
let diffLN = outputLN - pyAfterNorm1
eval(diffLN)
print("Difference:           max abs = \(diffLN.abs().max().item(Float.self))")

print("\n" + String(repeating: "=", count: 80))
let rmse = MLX.sqrt(MLX.mean(diffLN * diffLN)).item(Float.self)
print("RMSE: \(rmse)")
if rmse < 0.001 {
    print("✅ PERFECT MATCH!")
} else if rmse < 0.01 {
    print("✅ GOOD MATCH")
} else {
    print("❌ MISMATCH!")
}
print(String(repeating: "=", count: 80))
