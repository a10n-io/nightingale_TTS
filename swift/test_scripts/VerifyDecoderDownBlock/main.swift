import Foundation
import MLX
import MLXRandom
import MLXNN
import Nightingale

let PROJECT_ROOT = "/Users/a10n/Projects/nightingale_TTS"
let modelDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/models/chatterbox")
let refDir = "\(PROJECT_ROOT)/E2E/reference_outputs/samantha/expressive_surprise_en"

print("=" + String(repeating: "=", count: 79))
print("DECODER DOWN BLOCK VERIFICATION")
print("=" + String(repeating: "=", count: 79))

// Helper to load .npy files
func loadNpy(_ path: String) throws -> MLXArray {
    return try NPYLoader.load(contentsOf: URL(fileURLWithPath: path))
}

// Helper for comparison
func compare(_ name: String, swift: MLXArray, python: MLXArray) {
    eval(swift, python)
    let swiftMin = swift.min().item(Float.self)
    let swiftMax = swift.max().item(Float.self)
    let pyMin = python.min().item(Float.self)
    let pyMax = python.max().item(Float.self)
    let diff = swift - python
    eval(diff)
    let maxAbsDiff = diff.abs().max().item(Float.self)
    let rmse = MLX.sqrt(MLX.mean(diff * diff)).item(Float.self)

    print("\(name):")
    print("  Python: [\(pyMin), \(pyMax)]")
    print("  Swift:  [\(swiftMin), \(swiftMax)]")
    print("  Diff:   max abs = \(maxAbsDiff), RMSE = \(rmse)")

    if rmse < 0.001 {
        print("  ✅ PERFECT")
    } else if rmse < 0.1 {
        print("  ✅ GOOD")
    } else if rmse < 1.0 {
        print("  ⚠️  ACCEPTABLE")
    } else {
        print("  ❌ MISMATCH")
    }
}

// Load Python reference intermediate values
print("\n📥 Loading Python reference values...")
let pyH = try loadNpy("\(refDir)/dec_down_step1_h_concat.npy")
let pyAfterConv1 = try loadNpy("\(refDir)/dec_down_step2_after_conv1.npy")
let pyAfterNorm1 = try loadNpy("\(refDir)/dec_down_step2_after_norm1.npy")
let pyAfterMish = try loadNpy("\(refDir)/dec_down_step2_after_mish.npy")
let pyTimeEmb = try loadNpy("\(refDir)/dec_down_step2_t_emb_proj.npy")
let pyWithTime = try loadNpy("\(refDir)/dec_down_step2_with_time.npy")
let pyAfterBlock2 = try loadNpy("\(refDir)/dec_down_step2_after_block2.npy")
let pyMasked = try loadNpy("\(refDir)/dec_down_step2_masked.npy")
let pyResConv = try loadNpy("\(refDir)/dec_down_step2_res_conv.npy")
let pyResNetOut = try loadNpy("\(refDir)/dec_down_step2_resnet_out.npy")

print("✅ Loaded Python reference values")

// Load decoder weights
print("\n📦 Loading decoder weights...")
let decoderURL = modelDir.appendingPathComponent("decoder_weights.safetensors")
let weightsRaw = try MLX.loadArrays(url: decoderURL)
print("  Loaded \(weightsRaw.count) tensors")

// Transpose Linear weights from PyTorch [out, in] to MLX [in, out]
var weights: [String: MLXArray] = [:]
for (key, value) in weightsRaw {
    // Don't transpose conv weights, only linear weights
    let isLinear = key.hasSuffix(".weight") && value.ndim == 2 && (
        key.contains("mlp.1.") ||
        key.contains("queryProj") ||
        key.contains("keyProj") ||
        key.contains("valueProj") ||
        key.contains("outProj") ||
        key.contains(".layers.")
    )
    weights[key] = isLinear ? value.T : value
}

// Load inputs
print("\n📥 Loading inputs...")
let xBatched = try loadNpy("\(refDir)/step7_step1_x_before.npy")  // [2, 80, 696]
let muBatched = try loadNpy("\(refDir)/step7_mu_T.npy")           // [2, 80, 696]
let cond = try loadNpy("\(refDir)/step6_x_cond.npy")              // [1, 80, 696]
let spkBatched = try loadNpy("\(refDir)/step7_spk_emb.npy")       // [2, 80]
let maskBatched = try loadNpy("\(refDir)/step7_cond_T.npy")       // [2, 1, 696]

// Extract conditional pass (index 0)
let x = xBatched[0...0]      // [1, 80, 696]
let mu = muBatched[0...0]    // [1, 80, 696]
let spk = spkBatched[0...0]  // [1, 80]
let mask = maskBatched[0...0]  // [1, 1, 696]

print("  x: \(x.shape)")
print("  mu: \(mu.shape)")
print("  cond: \(cond.shape)")
print("  spk: \(spk.shape)")
print("  mask: \(mask.shape)")

// Step 1: Concatenate h
print("\n" + String(repeating: "=", count: 80))
print("STEP 1: Concatenate h")
print(String(repeating: "=", count: 80))

let spkExpanded = tiled(spk.expandedDimensions(axis: -1), repetitions: [1, 1, Int(x.shape[2])])  // [1, 80, 696]
let h = MLX.concatenated([x, mu, spkExpanded, cond], axis: 1)  // [1, 320, 696]

compare("h concat", swift: h, python: pyH)

// Step 2: ResNet block - Conv1
print("\n" + String(repeating: "=", count: 80))
print("STEP 2: Conv1")
print(String(repeating: "=", count: 80))

let prefix = "s3gen.flow.decoder.estimator.down_blocks.0.0"
let conv1W = weights["\(prefix).block1.block.0.weight"]!  // [256, 320, 3]
let conv1B = weights["\(prefix).block1.block.0.bias"]!    // [256]

// Pad for causal conv (kernel=3, pad left by 2)
let hPadded = padded(h, widths: [.init((0, 0)), .init((0, 0)), .init((2, 0))])  // [1, 320, 698]
print("  h padded: \(hPadded.shape)")

// Conv1d: Try transposing to [B, T, C] format as MLX might expect that
// hPadded is [1, 320, 698], transpose to [1, 698, 320]
let hPaddedTransposed = hPadded.transposed(0, 2, 1)  // [1, 698, 320]
// conv1W is [256, 320, 3] - MLX conv1d expects [out_c, in_c, k] which is correct
// But we might need to transpose it to [out_c, k, in_c]
let conv1WTransposed = conv1W.transposed(0, 2, 1)  // [256, 3, 320]
let hConv1Transposed = MLX.conv1d(hPaddedTransposed, conv1WTransposed, stride: 1, padding: 0) + conv1B.reshaped([1, 1, -1])  // [1, 696, 256]
let hConv1 = hConv1Transposed.transposed(0, 2, 1)  // [1, 256, 696]
compare("After conv1", swift: hConv1, python: pyAfterConv1)

// Step 3: Norm1 (GroupNorm with 8 groups)
print("\n" + String(repeating: "=", count: 80))
print("STEP 3: GroupNorm1")
print(String(repeating: "=", count: 80))

let norm1W = weights["\(prefix).block1.block.2.weight"]!  // [256]
let norm1B = weights["\(prefix).block1.block.2.bias"]!    // [256]

// GroupNorm: split channels into 8 groups
let B = Int(hConv1.shape[0])
let C = Int(hConv1.shape[1])
let T = Int(hConv1.shape[2])
let numGroups = 8
let channelsPerGroup = C / numGroups

// Reshape to [B, numGroups, channelsPerGroup, T]
let hReshaped = hConv1.reshaped([B, numGroups, channelsPerGroup, T])

// Compute mean and variance per group
let groupMean = hReshaped.mean(axes: [2, 3], keepDims: true)  // [B, numGroups, 1, 1]
let diff1 = hReshaped - groupMean
let groupVariance = (diff1 * diff1).mean(axes: [2, 3], keepDims: true)

// Normalize
let eps: Float = 1e-5
let hNormed = (hReshaped - groupMean) / MLX.sqrt(groupVariance + eps)

// Reshape back to [B, C, T]
let hGroupNorm = hNormed.reshaped([B, C, T])

// Apply affine transform
let hNorm1 = hGroupNorm * norm1W.reshaped([1, -1, 1]) + norm1B.reshaped([1, -1, 1])
compare("After norm1", swift: hNorm1, python: pyAfterNorm1)

// Step 4: Mish activation
print("\n" + String(repeating: "=", count: 80))
print("STEP 4: Mish")
print(String(repeating: "=", count: 80))

let hMish = hNorm1 * MLX.tanh(MLXNN.softplus(hNorm1))
compare("After mish", swift: hMish, python: pyAfterMish)

// Step 5: Time embedding projection
print("\n" + String(repeating: "=", count: 80))
print("STEP 5: Time embedding")
print(String(repeating: "=", count: 80))

let timeEmb = try loadNpy("\(refDir)/dec_trace_time_emb.npy")  // [1, 1024]
let mlp1W = weights["\(prefix).mlp.1.weight"]!  // [1024, 256] in PyTorch -> [256, 1024] in MLX
let mlp1B = weights["\(prefix).mlp.1.bias"]!    // [256]

// Time MLP: mish(time_emb) -> linear
let timeEmbMish = timeEmb * MLX.tanh(MLXNN.softplus(timeEmb))
let tEmb = MLX.matmul(timeEmbMish, mlp1W) + mlp1B
let tEmbExpanded = tEmb.reshaped([1, -1, 1])  // [1, 256, 1]
compare("Time emb proj", swift: tEmbExpanded, python: pyTimeEmb)

// Add time embedding
let hWithTime = hMish + tEmbExpanded
compare("After adding time", swift: hWithTime, python: pyWithTime)

// Step 6: Conv2
print("\n" + String(repeating: "=", count: 80))
print("STEP 6: Conv2 + Norm2 + Mish")
print(String(repeating: "=", count: 80))

let conv2W = weights["\(prefix).block2.block.0.weight"]!  // [256, 256, 3]
let conv2B = weights["\(prefix).block2.block.0.bias"]!    // [256]
let norm2W = weights["\(prefix).block2.block.2.weight"]!  // [256]
let norm2B = weights["\(prefix).block2.block.2.bias"]!    // [256]

// Pad and conv
let hPadded2 = padded(hWithTime, widths: [.init((0, 0)), .init((0, 0)), .init((2, 0))])
let hPadded2Transposed = hPadded2.transposed(0, 2, 1)
let conv2WTransposed = conv2W.transposed(0, 2, 1)
let hConv2Transposed = MLX.conv1d(hPadded2Transposed, conv2WTransposed, stride: 1, padding: 0) + conv2B.reshaped([1, 1, -1])
let hConv2 = hConv2Transposed.transposed(0, 2, 1)

// GroupNorm
let hReshaped2 = hConv2.reshaped([B, numGroups, channelsPerGroup, Int(hConv2.shape[2])])
let groupMean2 = hReshaped2.mean(axes: [2, 3], keepDims: true)
let diff2 = hReshaped2 - groupMean2
let groupVariance2 = (diff2 * diff2).mean(axes: [2, 3], keepDims: true)
let hNormed2 = (hReshaped2 - groupMean2) / MLX.sqrt(groupVariance2 + eps)
let hGroupNorm2 = hNormed2.reshaped([B, Int(hConv2.shape[1]), Int(hConv2.shape[2])])
let hNorm2 = hGroupNorm2 * norm2W.reshaped([1, -1, 1]) + norm2B.reshaped([1, -1, 1])

// Mish
let hMish2 = hNorm2 * MLX.tanh(MLXNN.softplus(hNorm2))
compare("After block2", swift: hMish2, python: pyAfterBlock2)

// Step 7: Apply mask
print("\n" + String(repeating: "=", count: 80))
print("STEP 7: Apply mask")
print(String(repeating: "=", count: 80))

let hMasked = hMish2 * mask
compare("After masking", swift: hMasked, python: pyMasked)

// Step 8: Residual connection
print("\n" + String(repeating: "=", count: 80))
print("STEP 8: Residual connection")
print(String(repeating: "=", count: 80))

let resConvW = weights["\(prefix).res_conv.weight"]!  // [256, 320, 1]
let resConvB = weights["\(prefix).res_conv.bias"]!    // [256]

let hTransposedRes = h.transposed(0, 2, 1)
let resConvWTransposed = resConvW.transposed(0, 2, 1)
let hResTransposed = MLX.conv1d(hTransposedRes, resConvWTransposed, stride: 1, padding: 0) + resConvB.reshaped([1, 1, -1])
let hRes = hResTransposed.transposed(0, 2, 1)
compare("Residual conv", swift: hRes, python: pyResConv)

let hOut = hMasked + hRes
compare("After residual", swift: hOut, python: pyResNetOut)

// Final summary
print("\n" + String(repeating: "=", count: 80))
print("SUMMARY")
print(String(repeating: "=", count: 80))

let finalDiff = hOut - pyResNetOut
eval(finalDiff)
let finalDiffSq = finalDiff * finalDiff
let finalRMSE = MLX.sqrt(MLX.mean(finalDiffSq)).item(Float.self)
print("Final RMSE: \(finalRMSE)")

if finalRMSE < 0.001 {
    print("✅ PERFECT MATCH!")
} else if finalRMSE < 0.1 {
    print("✅ EXCELLENT MATCH")
} else if finalRMSE < 1.0 {
    print("⚠️  ACCEPTABLE - Minor differences")
} else {
    print("❌ SIGNIFICANT MISMATCH")
}
print(String(repeating: "=", count: 80))
