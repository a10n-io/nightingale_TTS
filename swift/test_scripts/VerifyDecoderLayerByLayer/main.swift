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

// Helper function to remap Python decoder keys to Swift decoder keys
func remapDecoderKey(_ key: String) -> String {
    var k = key

    // Remove prefix
    k = k.replacingOccurrences(of: "s3gen.flow.decoder.", with: "")
    k = k.replacingOccurrences(of: "estimator.", with: "")

    // Block names with underscore conversion
    k = k.replacingOccurrences(of: "down_blocks.", with: "downBlocks.")
    k = k.replacingOccurrences(of: "mid_blocks.", with: "midBlocks.")
    k = k.replacingOccurrences(of: "up_blocks.", with: "upBlocks.")

    // CRITICAL: Python UNet structure vs Swift UNet structure
    // Python: down_blocks[0][0] = CausalResnetBlock1D, [0][1-4] = transformers, [0][5] = downsample
    // Swift: downBlocks[0].resnet, downBlocks[0].transformers[0-3], downBlocks[0].downLayer
    // Map .0.0. -> .0.resnet. (first inner element is the resnet)
    k = k.replacingOccurrences(of: "downBlocks.0.0.", with: "downBlocks.0.resnet.")
    k = k.replacingOccurrences(of: "midBlocks.0.0.", with: "midBlocks.0.resnet.")
    k = k.replacingOccurrences(of: "midBlocks.1.0.", with: "midBlocks.1.resnet.")
    k = k.replacingOccurrences(of: "midBlocks.2.0.", with: "midBlocks.2.resnet.")
    k = k.replacingOccurrences(of: "midBlocks.3.0.", with: "midBlocks.3.resnet.")
    k = k.replacingOccurrences(of: "midBlocks.4.0.", with: "midBlocks.4.resnet.")
    k = k.replacingOccurrences(of: "midBlocks.5.0.", with: "midBlocks.5.resnet.")
    k = k.replacingOccurrences(of: "midBlocks.6.0.", with: "midBlocks.6.resnet.")
    k = k.replacingOccurrences(of: "midBlocks.7.0.", with: "midBlocks.7.resnet.")
    k = k.replacingOccurrences(of: "midBlocks.8.0.", with: "midBlocks.8.resnet.")
    k = k.replacingOccurrences(of: "midBlocks.9.0.", with: "midBlocks.9.resnet.")
    k = k.replacingOccurrences(of: "midBlocks.10.0.", with: "midBlocks.10.resnet.")
    k = k.replacingOccurrences(of: "midBlocks.11.0.", with: "midBlocks.11.resnet.")
    k = k.replacingOccurrences(of: "upBlocks.0.0.", with: "upBlocks.0.resnet.")

    // Map transformer indices: Python uses .0.1.X. where X is transformer index in a nested list
    // Python: down_blocks[0][1][0] = first transformer, down_blocks[0][1][1] = second, etc.
    // Swift: downBlocks[0].transformers[0], downBlocks[0].transformers[1], etc.
    k = k.replacingOccurrences(of: "downBlocks.0.1.0.", with: "downBlocks.0.transformers.0.")
    k = k.replacingOccurrences(of: "downBlocks.0.1.1.", with: "downBlocks.0.transformers.1.")
    k = k.replacingOccurrences(of: "downBlocks.0.1.2.", with: "downBlocks.0.transformers.2.")
    k = k.replacingOccurrences(of: "downBlocks.0.1.3.", with: "downBlocks.0.transformers.3.")
    // Mid blocks have 4 transformers each in index 1 list
    for i in 0...11 {
        k = k.replacingOccurrences(of: "midBlocks.\(i).1.0.", with: "midBlocks.\(i).transformers.0.")
        k = k.replacingOccurrences(of: "midBlocks.\(i).1.1.", with: "midBlocks.\(i).transformers.1.")
        k = k.replacingOccurrences(of: "midBlocks.\(i).1.2.", with: "midBlocks.\(i).transformers.2.")
        k = k.replacingOccurrences(of: "midBlocks.\(i).1.3.", with: "midBlocks.\(i).transformers.3.")
    }
    k = k.replacingOccurrences(of: "upBlocks.0.1.0.", with: "upBlocks.0.transformers.0.")
    k = k.replacingOccurrences(of: "upBlocks.0.1.1.", with: "upBlocks.0.transformers.1.")
    k = k.replacingOccurrences(of: "upBlocks.0.1.2.", with: "upBlocks.0.transformers.2.")
    k = k.replacingOccurrences(of: "upBlocks.0.1.3.", with: "upBlocks.0.transformers.3.")

    // Downsample/Upsample - Python uses index 2 for down/up convs
    // Note: downLayer is CausalConv1d which contains Conv1d as .conv property
    k = k.replacingOccurrences(of: "downBlocks.0.2.", with: "downBlocks.0.downLayer.conv.")
    k = k.replacingOccurrences(of: "upBlocks.0.2.", with: "upBlocks.0.upLayer.conv.")

    // CRITICAL: CausalBlock1D structure mapping
    // Python: block = Sequential(CausalConv1d[0], Transpose[1], LayerNorm[2], Transpose[3], Mish[4])
    // Swift: conv: CausalConv1d, norm: LayerNorm
    // Map .block.0. -> .conv.conv. (CausalConv1d contains Conv1d)
    // Map .block.2. -> .norm. (LayerNorm)
    k = k.replacingOccurrences(of: ".block.0.", with: ".conv.conv.")
    k = k.replacingOccurrences(of: ".block.2.", with: ".norm.")

    // ResNet components
    // Python uses mlp.1 for the linear layer, Swift uses mlpLinear
    k = k.replacingOccurrences(of: ".mlp.1.", with: ".mlpLinear.")
    k = k.replacingOccurrences(of: "mlp_linear", with: "mlpLinear")
    k = k.replacingOccurrences(of: "res_conv", with: "resConv")

    // Transformer components
    k = k.replacingOccurrences(of: ".attn1.", with: ".attention.")
    k = k.replacingOccurrences(of: "to_q.", with: "queryProj.")
    k = k.replacingOccurrences(of: "to_k.", with: "keyProj.")
    k = k.replacingOccurrences(of: "to_v.", with: "valueProj.")
    k = k.replacingOccurrences(of: "to_out.0.", with: "outProj.")
    k = k.replacingOccurrences(of: ".norm3.", with: ".norm2.")
    k = k.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.layers.0.")
    k = k.replacingOccurrences(of: ".ff.net.2.", with: ".ff.layers.1.")

    // TimeMLP
    k = k.replacingOccurrences(of: "time_mlp", with: "timeMLP")
    k = k.replacingOccurrences(of: "linear_1", with: "linear1")
    k = k.replacingOccurrences(of: "linear_2", with: "linear2")

    // Final components
    k = k.replacingOccurrences(of: "final_block", with: "finalBlock")
    k = k.replacingOccurrences(of: "final_proj", with: "finalProj")

    return k
}

// Load decoder weights with proper key remapping
print("Loading decoder weights...")
var decoderWeights: [String: MLXArray] = [:]
for (key, value) in flowWeights {
    if key.hasPrefix("s3gen.flow.decoder.") {
        let newKey = remapDecoderKey(key)

        // CRITICAL: PyTorch Linear weights are [Out, In], but MLX Linear expects [In, Out]
        let isLinearWeight = newKey.hasSuffix(".weight") && value.ndim == 2 &&
                            !newKey.contains("conv") && !newKey.contains("norm") && !newKey.contains("embedding")

        // CRITICAL: PyTorch Conv1d weights are [Out, In, K], but MLX Conv1d expects [Out, K, In]
        let isConv1dWeight = newKey.hasSuffix(".weight") && value.ndim == 3 &&
                            !newKey.contains("norm") && !newKey.contains("embedding")

        if isLinearWeight {
            decoderWeights[newKey] = value.T
        } else if isConv1dWeight {
            decoderWeights[newKey] = value.transposed(0, 2, 1)
        } else {
            decoderWeights[newKey] = value
        }
    }
}
print("  Found \(decoderWeights.count) decoder weights")

// Debug: Print decoder expected keys
print("\n🔍 Decoder expects these parameter keys (sample):")
let decoderParams = decoder.parameters().flattened()
for (key, _) in decoderParams.prefix(20) {
    print("  \(key)")
}

// Debug: Print provided keys
print("\n🔍 Provided weight keys (sample):")
for (key, _) in decoderWeights.prefix(20) {
    print("  \(key)")
}

// Update decoder with weights - use try? to avoid crash
print("\n🔧 Attempting to load weights...")
do {
    try decoder.update(parameters: ModuleParameters.unflattened(decoderWeights))
    print("✅ Decoder loaded successfully")
} catch {
    print("❌ Failed to load decoder: \(error)")
    exit(1)
}

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
