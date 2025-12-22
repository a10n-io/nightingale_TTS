import Foundation
import MLX
import MLXRandom
import MLXNN
import Nightingale

let PROJECT_ROOT = "/Users/a10n/Projects/nightingale_TTS"
let modelDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/models/chatterbox")
let refDir = "\(PROJECT_ROOT)/E2E/reference_outputs/samantha/expressive_surprise_en"

print("=" + String(repeating: "=", count: 79))
print("FIRST TRANSFORMER BLOCK VERIFICATION")
print("=" + String(repeating: "=", count: 79))

// Helper to load .npy files
func loadNpy(_ path: String) throws -> MLXArray {
    return try NPYLoader.load(contentsOf: URL(fileURLWithPath: path))
}

// Load Python reference intermediate values
print("\n📥 Loading Python reference values...")
let pyInput = try loadNpy("\(refDir)/tfmr_trace_input.npy")
let pyAfterNorm1 = try loadNpy("\(refDir)/tfmr_trace_after_norm1.npy")
let pyQuery = try loadNpy("\(refDir)/tfmr_trace_query.npy")
let pyKey = try loadNpy("\(refDir)/tfmr_trace_key.npy")
let pyValue = try loadNpy("\(refDir)/tfmr_trace_value.npy")
let pyAttnScores = try loadNpy("\(refDir)/tfmr_trace_attn_scores.npy")
let pyAttnWeights = try loadNpy("\(refDir)/tfmr_trace_attn_weights.npy")
let pyAttnOutBeforeProj = try loadNpy("\(refDir)/tfmr_trace_attn_out_before_proj.npy")
let pyAttnOutput = try loadNpy("\(refDir)/tfmr_trace_attn_output.npy")
let pyAfterRes1 = try loadNpy("\(refDir)/tfmr_trace_after_res1.npy")
let pyAfterNorm3 = try loadNpy("\(refDir)/tfmr_trace_after_norm3.npy")
let pyFFOutput = try loadNpy("\(refDir)/tfmr_trace_ff_output.npy")
let pyFinalOutput = try loadNpy("\(refDir)/tfmr_trace_final_output.npy")

// Also need the mask - transform to additive bias mask
let maskFull = try loadNpy("\(refDir)/step7_cond_T.npy")
let mask = maskFull[0...0]  // [1, 1, 696]
let mask2D = mask.squeezed(axis: 1)  // [1, 696]

// Create additive bias mask: [1, 1, 1, 696]
// Where mask == 0, bias = -1e9; where mask == 1, bias = 0
let mask3D = mask2D.expandedDimensions(axis: 1).expandedDimensions(axis: 1)  // [1, 1, 1, 696]
let biasMask = MLX.where(mask3D .== 0, MLXArray(-1e9), MLXArray.zeros(mask3D.shape))

eval(pyInput, biasMask)
print("✅ Loaded Python reference values")

// Load decoder weights
print("\n📦 Loading decoder weights...")
let decoderURL = modelDir.appendingPathComponent("decoder_weights.safetensors")
let weightsRaw = try MLX.loadArrays(url: decoderURL)
print("  Loaded \(weightsRaw.count) tensors")

// Transpose Linear weights from PyTorch [out, in] to MLX [in, out]
var weights: [String: MLXArray] = [:]
for (key, value) in weightsRaw {
    let isLinear = key.hasSuffix(".weight") && value.ndim == 2 && (
        key.contains("queryProj") ||
        key.contains("keyProj") ||
        key.contains("valueProj") ||
        key.contains("outProj") ||
        key.contains(".layers.")
    )
    weights[key] = isLinear ? value.T : value
    if isLinear {
        print("  Transposed \(key): \(value.shape) -> \(value.T.shape)")
    }
}

// Create transformer block
print("\n🔧 Creating FlowTransformerBlock...")
let tfmr = FlowTransformerBlock(
    dim: 256,
    numHeads: 8,
    headDim: 64
)

// Load transformer weights for down_blocks.0.transformers.0
let prefix = "s3gen.flow.decoder.estimator.down_blocks.0.1.0"

// norm1
let norm1W = weights["\(prefix).norm1.weight"]!
let norm1B = weights["\(prefix).norm1.bias"]!
tfmr.norm1.update(parameters: ModuleParameters.unflattened(["weight": norm1W, "bias": norm1B]))

// Verify weights were loaded
if let w = tfmr.norm1.weight, let b = tfmr.norm1.bias {
    eval(w, b)
    print("  norm1.weight: [\(w.min().item(Float.self)), \(w.max().item(Float.self))], mean=\(MLX.mean(w).item(Float.self))")
    print("  norm1.bias: [\(b.min().item(Float.self)), \(b.max().item(Float.self))], mean=\(MLX.mean(b).item(Float.self))")
} else {
    print("  WARNING: norm1 weights/bias not loaded!")
}

// Attention projections (need transpose for FixedLinear)
let qW_raw = weights["\(prefix).attn1.to_q.weight"]!
let qW = qW_raw.T  // [512, 256] -> [256, 512]
tfmr.attention.queryProj.weight = qW

let kW_raw = weights["\(prefix).attn1.to_k.weight"]!
let kW = kW_raw.T  // [512, 256] -> [256, 512]
tfmr.attention.keyProj.weight = kW

let vW_raw = weights["\(prefix).attn1.to_v.weight"]!
let vW = vW_raw.T  // [512, 256] -> [256, 512]
tfmr.attention.valueProj.weight = vW

let outW_raw = weights["\(prefix).attn1.to_out.0.weight"]!
let outW = outW_raw.T  // [256, 512] -> [512, 256]
let outB = weights["\(prefix).attn1.to_out.0.bias"]!
tfmr.attention.outProj.weight = outW
tfmr.attention.outProj.bias = outB

// norm2 (called norm3 in Python)
let norm2W = weights["\(prefix).norm3.weight"]!
let norm2B = weights["\(prefix).norm3.bias"]!
tfmr.norm2.update(parameters: ModuleParameters.unflattened(["weight": norm2W, "bias": norm2B]))

// Feedforward layers (need transpose for FixedLinear)
let ff1W_raw = weights["\(prefix).ff.net.0.proj.weight"]!
let ff1W = ff1W_raw.T  // [1024, 256] -> [256, 1024]
let ff1B = weights["\(prefix).ff.net.0.proj.bias"]!
tfmr.ff.layers[0].weight = ff1W
tfmr.ff.layers[0].bias = ff1B

let ff2W_raw = weights["\(prefix).ff.net.2.weight"]!
let ff2W = ff2W_raw.T  // [256, 1024] -> [1024, 256]
let ff2B = weights["\(prefix).ff.net.2.bias"]!
tfmr.ff.layers[1].weight = ff2W
tfmr.ff.layers[1].bias = ff2B

print("✅ Transformer weights loaded")

// Test 1: Input comparison
print("\n" + String(repeating: "=", count: 80))
print("TEST 1: INPUT")
print(String(repeating: "=", count: 80))

let swiftInput = pyInput  // Use Python input directly
eval(swiftInput)
print("Python input: [\(pyInput.min().item(Float.self)), \(pyInput.max().item(Float.self))]")
print("Swift  input: [\(swiftInput.min().item(Float.self)), \(swiftInput.max().item(Float.self))]")
let diffInput = swiftInput - pyInput
eval(diffInput)
print("Difference:   max abs = \(diffInput.abs().max().item(Float.self))")

// Test 2: After norm1
print("\n" + String(repeating: "=", count: 80))
print("TEST 2: AFTER NORM1")
print(String(repeating: "=", count: 80))

let swiftNorm1 = tfmr.norm1(swiftInput)
eval(swiftNorm1)
print("Python after norm1: [\(pyAfterNorm1.min().item(Float.self)), \(pyAfterNorm1.max().item(Float.self))]")
print("Swift  after norm1: [\(swiftNorm1.min().item(Float.self)), \(swiftNorm1.max().item(Float.self))]")
let diffNorm1 = swiftNorm1 - pyAfterNorm1
eval(diffNorm1)
print("Difference:         max abs = \(diffNorm1.abs().max().item(Float.self))")

// Test 3: Query projection
print("\n" + String(repeating: "=", count: 80))
print("TEST 3: QUERY PROJECTION")
print(String(repeating: "=", count: 80))

let swiftQuery = tfmr.attention.queryProj(swiftNorm1)
eval(swiftQuery)
print("Python query: [\(pyQuery.min().item(Float.self)), \(pyQuery.max().item(Float.self))]")
print("Swift  query: [\(swiftQuery.min().item(Float.self)), \(swiftQuery.max().item(Float.self))]")
let diffQuery = swiftQuery - pyQuery
eval(diffQuery)
print("Difference:   max abs = \(diffQuery.abs().max().item(Float.self))")

// Test 4: Key projection
print("\n" + String(repeating: "=", count: 80))
print("TEST 4: KEY PROJECTION")
print(String(repeating: "=", count: 80))

let swiftKey = tfmr.attention.keyProj(swiftNorm1)
eval(swiftKey)
print("Python key: [\(pyKey.min().item(Float.self)), \(pyKey.max().item(Float.self))]")
print("Swift  key: [\(swiftKey.min().item(Float.self)), \(swiftKey.max().item(Float.self))]")
let diffKey = swiftKey - pyKey
eval(diffKey)
print("Difference: max abs = \(diffKey.abs().max().item(Float.self))")

// Test 5: Value projection
print("\n" + String(repeating: "=", count: 80))
print("TEST 5: VALUE PROJECTION")
print(String(repeating: "=", count: 80))

let swiftValue = tfmr.attention.valueProj(swiftNorm1)
eval(swiftValue)
print("Python value: [\(pyValue.min().item(Float.self)), \(pyValue.max().item(Float.self))]")
print("Swift  value: [\(swiftValue.min().item(Float.self)), \(swiftValue.max().item(Float.self))]")
let diffValue = swiftValue - pyValue
eval(diffValue)
print("Difference:   max abs = \(diffValue.abs().max().item(Float.self))")

// Test 6: Full attention (with manual Q/K/V/mask for debugging)
print("\n" + String(repeating: "=", count: 80))
print("TEST 6: FULL ATTENTION OUTPUT")
print(String(repeating: "=", count: 80))

// Use the attention module with properly transformed bias mask
let swiftAttnOutput = tfmr.attention(swiftNorm1, mask: biasMask)
eval(swiftAttnOutput)
print("Python attention output: [\(pyAttnOutput.min().item(Float.self)), \(pyAttnOutput.max().item(Float.self))]")
print("Swift  attention output: [\(swiftAttnOutput.min().item(Float.self)), \(swiftAttnOutput.max().item(Float.self))]")
let diffAttn = swiftAttnOutput - pyAttnOutput
eval(diffAttn)
print("Difference:              max abs = \(diffAttn.abs().max().item(Float.self))")

// Test 7: After residual 1
print("\n" + String(repeating: "=", count: 80))
print("TEST 7: AFTER RESIDUAL 1")
print(String(repeating: "=", count: 80))

let swiftAfterRes1 = swiftInput + swiftAttnOutput
eval(swiftAfterRes1)
print("Python after res1: [\(pyAfterRes1.min().item(Float.self)), \(pyAfterRes1.max().item(Float.self))]")
print("Swift  after res1: [\(swiftAfterRes1.min().item(Float.self)), \(swiftAfterRes1.max().item(Float.self))]")
let diffRes1 = swiftAfterRes1 - pyAfterRes1
eval(diffRes1)
print("Difference:        max abs = \(diffRes1.abs().max().item(Float.self))")

// Test 8: After norm2 (norm3 in Python)
print("\n" + String(repeating: "=", count: 80))
print("TEST 8: AFTER NORM2")
print(String(repeating: "=", count: 80))

let swiftNorm2 = tfmr.norm2(swiftAfterRes1)
eval(swiftNorm2)
print("Python after norm2: [\(pyAfterNorm3.min().item(Float.self)), \(pyAfterNorm3.max().item(Float.self))]")
print("Swift  after norm2: [\(swiftNorm2.min().item(Float.self)), \(swiftNorm2.max().item(Float.self))]")
let diffNorm2 = swiftNorm2 - pyAfterNorm3
eval(diffNorm2)
print("Difference:         max abs = \(diffNorm2.abs().max().item(Float.self))")

// Test 9: Feedforward output
print("\n" + String(repeating: "=", count: 80))
print("TEST 9: FEEDFORWARD OUTPUT")
print(String(repeating: "=", count: 80))

let swiftFF = tfmr.ff(swiftNorm2)
eval(swiftFF)
print("Python FF output: [\(pyFFOutput.min().item(Float.self)), \(pyFFOutput.max().item(Float.self))]")
print("Swift  FF output: [\(swiftFF.min().item(Float.self)), \(swiftFF.max().item(Float.self))]")
let diffFF = swiftFF - pyFFOutput
eval(diffFF)
print("Difference:       max abs = \(diffFF.abs().max().item(Float.self))")

// Test 10: Final output (after residual 2)
print("\n" + String(repeating: "=", count: 80))
print("TEST 10: FINAL OUTPUT")
print(String(repeating: "=", count: 80))

let swiftFinal = swiftAfterRes1 + swiftFF
eval(swiftFinal)
print("Python final output: [\(pyFinalOutput.min().item(Float.self)), \(pyFinalOutput.max().item(Float.self))]")
print("Swift  final output: [\(swiftFinal.min().item(Float.self)), \(swiftFinal.max().item(Float.self))]")
let diffFinal = swiftFinal - pyFinalOutput
eval(diffFinal)
print("Difference:          max abs = \(diffFinal.abs().max().item(Float.self))")
let rmse = MLX.sqrt(MLX.mean(diffFinal * diffFinal)).item(Float.self)
print("RMSE:                \(rmse)")

print("\n" + String(repeating: "=", count: 80))
if rmse < 0.001 {
    print("✅ PERFECT MATCH!")
} else if rmse < 0.01 {
    print("✅ EXCELLENT MATCH (RMSE < 0.01)")
} else if rmse < 0.1 {
    print("✅ GOOD MATCH (RMSE < 0.1)")
} else {
    print("❌ MISMATCH FOUND!")
}
print(String(repeating: "=", count: 80))
