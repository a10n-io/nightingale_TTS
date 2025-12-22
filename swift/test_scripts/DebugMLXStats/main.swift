import Foundation
import MLX
import Nightingale

print("=" + String(repeating: "=", count: 79))
print("DEBUG MLX MEAN/VARIANCE")
print("=" + String(repeating: "=", count: 79))

// Create a simple test tensor [1, 1, 3]
let test = MLXArray([Float(1.0), Float(2.0), Float(3.0)]).reshaped([1, 1, 3])

eval(test)
print("\nTest tensor: \(test.shape)")
print("Values: \(test)")
print("Expected mean: 2.0")

// Compute mean over last dimension
let mean_axis_neg1 = MLX.mean(test, axis: -1, keepDims: true)
let mean_axis_2 = MLX.mean(test, axis: 2, keepDims: true)
eval(mean_axis_neg1, mean_axis_2)

print("\nMean over axis=-1: \(mean_axis_neg1.shape)")
print("Value: \(mean_axis_neg1.item(Float.self))")
print("Expected: 2.0")

print("\nMean over axis=2: \(mean_axis_2.shape)")
print("Value: \(mean_axis_2.item(Float.self))")
print("Expected: 2.0")

// Compute variance over last dimension
let var_axis_neg1 = MLX.variance(test, axis: -1, keepDims: true, ddof: 0)
let var_axis_2 = MLX.variance(test, axis: 2, keepDims: true, ddof: 0)
eval(var_axis_neg1, var_axis_2)

print("\nVariance over axis=-1 (ddof=0): \(var_axis_neg1.shape)")
print("Value: \(var_axis_neg1.item(Float.self))")
print("Expected: ~0.667")

print("\nVariance over axis=2 (ddof=0): \(var_axis_2.shape)")
print("Value: \(var_axis_2.item(Float.self))")
print("Expected: ~0.667")

// Now test with the actual data shape [1, 696, 256]
print("\n" + String(repeating: "=", count: 80))
print("Test with actual input data")
print(String(repeating: "=", count: 80))

// Load the actual input
let PROJECT_ROOT = "/Users/a10n/Projects/nightingale_TTS"
let refDir = "\(PROJECT_ROOT)/E2E/reference_outputs/samantha/expressive_surprise_en"

func loadNpy(_ path: String) throws -> MLXArray {
    return try NPYLoader.load(contentsOf: URL(fileURLWithPath: path))
}

let pyInput = try loadNpy("\(refDir)/tfmr_trace_input.npy")
let pyMean = try loadNpy("\(refDir)/tfmr_ln_mean.npy")

eval(pyInput, pyMean)
print("\nInput: \(pyInput.shape)")

// Take first sequence as test - try different indexing
print("\nTesting different indexing methods:")
let firstSeq1 = pyInput[0, 0]  // Integer indexing
eval(firstSeq1)
print("  pyInput[0, 0]: shape=\(firstSeq1.shape), range=[\(firstSeq1.min().item(Float.self)), \(firstSeq1.max().item(Float.self))]")

let firstSeq2 = pyInput[0...0, 0...0, 0...]  // Range slicing
let firstSeq2Squeezed = firstSeq2.squeezed()
eval(firstSeq2Squeezed)
print("  pyInput[0...0, 0...0, 0...].squeezed(): shape=\(firstSeq2Squeezed.shape), range=[\(firstSeq2Squeezed.min().item(Float.self)), \(firstSeq2Squeezed.max().item(Float.self))]")

let firstSeq3 = pyInput[0][0]  // Chained indexing
eval(firstSeq3)
print("  pyInput[0][0]: shape=\(firstSeq3.shape), range=[\(firstSeq3.min().item(Float.self)), \(firstSeq3.max().item(Float.self))]")

// Use the properly sliced version
let firstSeq = firstSeq2Squeezed

// Print first 10 values to see if they're really all the same
print("\nFirst 10 values of first sequence:")
for i in 0..<10 {
    let val = firstSeq[i].item(Float.self)
    print("  [\(i)]: \(val)")
}

// MLX mean
let mlxMean = MLX.mean(firstSeq)
eval(mlxMean)
print("\nMLX mean of first sequence: \(mlxMean.item(Float.self))")

// Python mean for this sequence
let pyFirstMean = pyMean[0, 0, 0]
eval(pyFirstMean)
print("Python mean (from reference): \(pyFirstMean.item(Float.self))")

// Expected from Python: [-0.03070468, -0.00576415, 0.00731245, -0.01389381, -0.01197754, ...]
print("\nExpected first 5 values from Python: [-0.03070468, -0.00576415, 0.00731245, -0.01389381, -0.01197754]")

print("\n" + String(repeating: "=", count: 80))
