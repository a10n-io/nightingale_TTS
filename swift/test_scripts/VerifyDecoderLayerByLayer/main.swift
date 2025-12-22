import Foundation
import MLX
import MLXRandom
import MLXNN
import Nightingale

let PROJECT_ROOT = "/Users/a10n/Projects/nightingale_TTS"
let modelDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/models/chatterbox")
let refDir = "\(PROJECT_ROOT)/E2E/reference_outputs/samantha/expressive_surprise_en"

print("=" + String(repeating: "=", count: 79))
print("DECODER LAYER-BY-LAYER VERIFICATION")
print("=" + String(repeating: "=", count: 79))

// Helper to load .npy files
func loadNpy(_ path: String) throws -> MLXArray {
    return try NPYLoader.load(contentsOf: URL(fileURLWithPath: path))
}

// Load Python reference inputs for step 7 (first ODE step)
print("\n📥 Loading Python reference data...")

// CRITICAL: step6_x_cond is the conditioning features [1, 80, T]
// step7_cond_T is the mask [2, 1, T] - NOT the decoder input!
let refXCond = try loadNpy("\(refDir)/step6_x_cond.npy")                 // [1, 80, 696]
let refXBatched = try loadNpy("\(refDir)/step7_step1_x_before.npy")      // [2, 80, 696]
let refMuBatched = try loadNpy("\(refDir)/step7_mu_T.npy")               // [2, 80, 696]
let refSpkBatched = try loadNpy("\(refDir)/step7_spk_emb.npy")           // [2, 80]
let refT = try loadNpy("\(refDir)/step7_step1_t.npy")                    // []

// Extract individual samples
let refX = refXBatched[0].expandedDimensions(axis: 0)       // [1, 80, 696]
let refMu = refMuBatched[0].expandedDimensions(axis: 0)     // [1, 80, 696]
let refCond = refXCond                                       // [1, 80, 696] - use x_cond!
let refSpk = refSpkBatched[0].expandedDimensions(axis: 0)   // [1, 80]

// Expected outputs
let refVCond = try loadNpy("\(refDir)/step7_step1_dxdt_cond.npy")    // [1, 80, 696]
let refVUncond = try loadNpy("\(refDir)/step7_step1_dxdt_uncond.npy") // [1, 80, 696]

print("✅ Loaded reference data:")
print("  x:      \(refX.shape)")
print("  mu:     \(refMu.shape)")
print("  cond:   \(refCond.shape)")
print("  spk:    \(refSpk.shape)")
print("  t:      \(refT.shape)")
print("  vCond:  \(refVCond.shape)")
print("  vUncond:\(refVUncond.shape)")

// Verify Python vUncond is truly zero
eval(refVUncond)
print("\n🔍 Python vUncond verification:")
print("  Range: [\(refVUncond.min().item(Float.self)), \(refVUncond.max().item(Float.self))]")
print("  Sum: \(refVUncond.sum().item(Float.self))")

print("\n📊 Python vCond reference:")
eval(refVCond)
print("  Range: [\(refVCond.min().item(Float.self)), \(refVCond.max().item(Float.self))]")
print("  Mean: \(refVCond.mean().item(Float.self))")

// Load decoder weights
print("\n📦 Loading decoder weights...")
let decoderURL = modelDir.appendingPathComponent("decoder_weights.safetensors")
let flowWeights = try MLX.loadArrays(url: decoderURL)
print("  Loaded \(flowWeights.count) tensors")

// Create decoder config matching Python
print("\n🔧 Creating decoder...")
let config = S3GenConfig()
let decoder = FlowMatchingDecoder(config: config)

// Load decoder weights (simplified - load all s3gen.flow.decoder.* keys)
print("Loading decoder weights...")
var decoderWeights: [String: MLXArray] = [:]
for (key, value) in flowWeights {
    if key.hasPrefix("s3gen.flow.decoder.") {
        var newKey = key.replacingOccurrences(of: "s3gen.flow.decoder.", with: "")
        // Python has "estimator.time_mlp" but Swift has "timeMLP" directly
        newKey = newKey.replacingOccurrences(of: "estimator.", with: "")
        // Python uses "linear_1" but Swift uses "linear1"
        newKey = newKey.replacingOccurrences(of: "time_mlp", with: "timeMLP")
        newKey = newKey.replacingOccurrences(of: "linear_1", with: "linear1")
        newKey = newKey.replacingOccurrences(of: "linear_2", with: "linear2")

        // CRITICAL: PyTorch Linear weights are [Out, In], but MLX FixedLinear expects [In, Out]
        // Transpose Linear layer weights (FixedLinear layers: TimeMLP, FeedForward, Attention)
        // Do NOT transpose Conv1d weights - decoder convs are already in correct format
        let isLinearWeight = newKey.hasSuffix(".weight") && value.ndim == 2 &&
                            !newKey.contains("conv") && !newKey.contains("norm") && !newKey.contains("embedding")

        if isLinearWeight {
            decoderWeights[newKey] = value.T
            print("  [TRANSPOSE] \(newKey): \(value.shape) -> \(value.T.shape)")
        } else {
            decoderWeights[newKey] = value
        }
    }
}
print("  Found \(decoderWeights.count) decoder weights")

// Update decoder with weights
try decoder.update(parameters: ModuleParameters.unflattened(decoderWeights))
print("✅ Decoder loaded")

// CRITICAL: Python uses cond_T as the mask, where:
//   cond_T[0] (conditional) has mask=0 for prompt, mask=1 for generation
//   cond_T[1] (unconditional) is ALL ZEROS
let refCondMaskBatched = try loadNpy("\(refDir)/step7_cond_T.npy")  // [2, 1, 696]
let maskCondPython = refCondMaskBatched[0].expandedDimensions(axis: 0)  // [1, 1, 696]
let maskUncondPython = refCondMaskBatched[1].expandedDimensions(axis: 0)  // [1, 1, 696]

print("\n🔍 Loaded Python masks:")
print("  maskCond sum: \(maskCondPython.sum().item(Float.self))/\(maskCondPython.size) ones")
print("  maskUncond sum: \(maskUncondPython.sum().item(Float.self))/\(maskUncondPython.size) ones")

print("\n" + String(repeating: "=", count: 80))
print("TEST 1: CONDITIONAL PASS (mask=Python)")
print(String(repeating: "=", count: 80))

// Enable debugging for TimeMLP
TimeMLP.debugEnabled = true
FlowMatchingDecoder.debugStep = 1
let swiftVCond = decoder(x: refX, mu: refMu, t: refT, speakerEmb: refSpk, cond: refCond, mask: maskCondPython)
eval(swiftVCond)

print("\n📊 Conditional output comparison:")
print("Python vCond:  [\(refVCond.min().item(Float.self)), \(refVCond.max().item(Float.self))], mean=\(refVCond.mean().item(Float.self))")
print("Swift vCond:   [\(swiftVCond.min().item(Float.self)), \(swiftVCond.max().item(Float.self))], mean=\(swiftVCond.mean().item(Float.self))")

let diffCond = swiftVCond - refVCond
eval(diffCond)
print("Difference:    [\(diffCond.min().item(Float.self)), \(diffCond.max().item(Float.self))], mean=\(diffCond.mean().item(Float.self))")
print("Max abs diff:  \(diffCond.abs().max().item(Float.self))")
print("RMSE:          \(sqrt((diffCond * diffCond).mean()).item(Float.self))")

print("\n" + String(repeating: "=", count: 80))
print("TEST 2: UNCONDITIONAL PASS (mask=Python zeros)")
print(String(repeating: "=", count: 80))

// CRITICAL: Python's unconditional pass CFG setup (flow_matching.py lines 127-132):
//   x_in[B:] = x         ← SAME as conditional
//   mask_in[B:] = mask   ← SAME as conditional
//   mu_in[B:] = NOT SET  ← Remains ZEROS from initialization!
//   spks_in[B:] = NOT SET ← Remains ZEROS from initialization!
//   cond_in[B:] = NOT SET ← Remains ZEROS from initialization!
let zeroCond = MLXArray.zeros(like: refCond)
let zeroMu = MLXArray.zeros(like: refMu)
let zeroSpk = MLXArray.zeros(like: refSpk)

FlowMatchingDecoder.debugStep = 1
let swiftVUncond = decoder(x: refX, mu: zeroMu, t: refT, speakerEmb: zeroSpk, cond: zeroCond, mask: maskUncondPython)
eval(swiftVUncond)

print("\n📊 Unconditional output comparison:")
print("Python vUncond: [\(refVUncond.min().item(Float.self)), \(refVUncond.max().item(Float.self))], mean=\(refVUncond.mean().item(Float.self))")
print("Swift vUncond:  [\(swiftVUncond.min().item(Float.self)), \(swiftVUncond.max().item(Float.self))], mean=\(swiftVUncond.mean().item(Float.self))")

let diffUncond = swiftVUncond - refVUncond
eval(diffUncond)
print("Difference:     [\(diffUncond.min().item(Float.self)), \(diffUncond.max().item(Float.self))], mean=\(diffUncond.mean().item(Float.self))")
print("Max abs diff:   \(diffUncond.abs().max().item(Float.self))")

print("\n" + String(repeating: "=", count: 80))
print("DIAGNOSIS")
print(String(repeating: "=", count: 80))

if diffCond.abs().max().item(Float.self) < 0.01 {
    print("✅ Conditional pass MATCHES Python!")
} else {
    print("❌ Conditional pass DIFFERS from Python")
    print("   RMSE: \(sqrt((diffCond * diffCond).mean()).item(Float.self))")
}

if swiftVUncond.abs().max().item(Float.self) < 0.001 {
    print("✅ Unconditional pass outputs zeros!")
} else {
    print("❌ Unconditional pass should output zeros but doesn't")
    print("   This is the ROOT CAUSE of wrong velocities")
    print("   Max abs value: \(swiftVUncond.abs().max().item(Float.self))")
}

print("\n" + String(repeating: "=", count: 80))
