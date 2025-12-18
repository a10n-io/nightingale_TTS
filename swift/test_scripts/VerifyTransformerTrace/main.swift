import Foundation
import MLX
import MLXNN
import Nightingale

print(String(repeating: "=", count: 80))
print("TRANSFORMER INTERNAL TRACE - Finding the Divergence Point")
print(String(repeating: "=", count: 80))
print()

// Paths
let modelPath = "/Users/a10n/Projects/nightingale/models"
let debugDir = "/Users/a10n/Projects/nightingale/verification_outputs/step7_debug"

// Helper function to compare and report
func compare(_ name: String, _ swift: MLXArray, _ pyFile: String, threshold: Float = 0.01) -> Bool {
    let pyRef = try! NPYLoader.load(contentsOf: URL(fileURLWithPath: "\(debugDir)/\(pyFile)")).asType(.float32)
    let diff = MLX.abs(swift - pyRef)
    eval(diff)

    let maxDiff = diff.max().item(Float.self)
    let meanDiff = diff.mean().item(Float.self)

    print("\(name):")
    print("  Swift[0,0,:5]: \(swift[0, 0, 0..<5].asArray(Float.self))")
    print("  Python[0,0,:5]: \(pyRef[0, 0, 0..<5].asArray(Float.self))")
    print("  Max diff: \(maxDiff)")
    print("  Mean diff: \(meanDiff)")

    if maxDiff < threshold {
        print("  ✅ MATCH")
        print()
        return true
    } else {
        print("  ❌ DIVERGENCE FOUND!")
        print()
        return false
    }
}

// Load S3Gen model
print("Loading S3Gen model...")
let modelsURL = URL(fileURLWithPath: modelPath)
let hfWeightsURL = modelsURL.appendingPathComponent("chatterbox_hf.safetensors")
let vocoderWeightsURL = modelsURL.appendingPathComponent("vocoder_weights.safetensors")

let allWeights = try! MLX.loadArrays(url: hfWeightsURL)
let vocoderWeights = try! MLX.loadArrays(url: vocoderWeightsURL)
let s3gen = S3Gen(flowWeights: allWeights, vocoderWeights: vocoderWeights)

// Load decoder weights (same remapping as E2E test)
print("Loading decoder weights...")
var decoderWeights: [String: MLXArray] = [:]
for (key, value) in allWeights {
    if key.contains("decoder.estimator") {
        var remappedKey = key
        if remappedKey.hasPrefix("s3gen.flow.decoder.estimator.") {
            remappedKey = remappedKey.replacingOccurrences(of: "s3gen.flow.decoder.estimator.", with: "")
        } else if remappedKey.hasPrefix("flow.decoder.estimator.") {
            remappedKey = remappedKey.replacingOccurrences(of: "flow.decoder.estimator.", with: "")
        }

        remappedKey = remappedKey.replacingOccurrences(of: "time_mlp", with: "timeMLP")
        remappedKey = remappedKey.replacingOccurrences(of: ".linear_1.", with: ".linear1.")
        remappedKey = remappedKey.replacingOccurrences(of: ".linear_2.", with: ".linear2.")
        remappedKey = remappedKey.replacingOccurrences(of: "down_blocks_", with: "downBlocks.")
        remappedKey = remappedKey.replacingOccurrences(of: "mid_blocks_", with: "midBlocks.")
        remappedKey = remappedKey.replacingOccurrences(of: "up_blocks_", with: "upBlocks.")
        remappedKey = remappedKey.replacingOccurrences(of: ".transformer_", with: ".transformers.")
        remappedKey = remappedKey.replacingOccurrences(of: "mlp_linear", with: "mlpLinear")
        remappedKey = remappedKey.replacingOccurrences(of: "res_conv", with: "resConv")
        remappedKey = remappedKey.replacingOccurrences(of: "final_block", with: "finalBlock")
        remappedKey = remappedKey.replacingOccurrences(of: "final_proj", with: "finalProj")
        remappedKey = remappedKey.replacingOccurrences(of: "downsample", with: "downLayer")
        remappedKey = remappedKey.replacingOccurrences(of: "upsample", with: "upLayer")
        remappedKey = remappedKey.replacingOccurrences(of: ".attn.", with: ".attention.")
        remappedKey = remappedKey.replacingOccurrences(of: "query_proj", with: "queryProj")
        remappedKey = remappedKey.replacingOccurrences(of: "key_proj", with: "keyProj")
        remappedKey = remappedKey.replacingOccurrences(of: "value_proj", with: "valueProj")
        remappedKey = remappedKey.replacingOccurrences(of: "out_proj", with: "outProj")
        remappedKey = remappedKey.replacingOccurrences(of: "ff.net.0.", with: "ff.layers.0.")
        remappedKey = remappedKey.replacingOccurrences(of: "ff.net.2.", with: "ff.layers.1.")
        // Python calls it norm3, Swift calls it norm2
        remappedKey = remappedKey.replacingOccurrences(of: ".norm3.", with: ".norm2.")

        decoderWeights[remappedKey] = value
    }
}

let decoderParams = ModuleParameters.unflattened(decoderWeights)
s3gen.decoder.update(parameters: decoderParams)
print("✅ Decoder weights loaded")
print()

// CRITICAL: Load out_proj.bias (not in safetensors, exported from Python)
print("Loading out_proj.bias from Python export...")
let outProjBias = try! NPYLoader.load(contentsOf: URL(fileURLWithPath: "\(debugDir)/decoder_down_0_tfmr_0_outproj_bias.npy")).asType(.float32)
let biasParams = ModuleParameters.unflattened(["downBlocks.0.transformers.0.attention.outProj.bias": outProjBias])
s3gen.decoder.update(parameters: biasParams)
print("  ✅ Loaded outProj.bias for down_blocks[0].transformers[0]")
print()

// Load the PERFECT ResNet output (verified to match Python)
print("Loading ResNet output (verified perfect match)...")
let resnetOut = try! NPYLoader.load(contentsOf: URL(fileURLWithPath: "\(debugDir)/debug_resnet_out.npy")).asType(.float32)
let h = resnetOut.transposed(0, 2, 1)  // [1, 256, 908] -> [1, 908, 256]
eval(h)

print("  h shape: \(h.shape)")
print("  h[0,0,:5]: \(h[0, 0, 0..<5].asArray(Float.self))")
print()

// Get the transformer block
let tfmrBlock = s3gen.decoder.downBlocks[0].transformers[0]

// Create attention mask
let attnMask = MLXArray.zeros([1, 1, 908, 908]).asType(.float32)

print(String(repeating: "=", count: 80))
print("TRANSFORMER INTERNAL AUTOPSY - Step-by-Step Comparison")
print(String(repeating: "=", count: 80))
print()

// STEP 1: LayerNorm 1
print("STEP 1: LayerNorm 1 (Pre-Attention)")
print(String(repeating: "-", count: 80))
let norm1Out = tfmrBlock.norm1(h)
eval(norm1Out)

if !compare("LayerNorm 1", norm1Out, "debug_tf_norm1.npy") {
    print("🎯 DIVERGENCE FOUND: LayerNorm 1")
    print("   Hypothesis: Epsilon mismatch (1e-5 vs 1e-6) or axis ordering")
    print()
    exit(1)
}

// STEP 2: Multi-Head Attention
print("STEP 2: Multi-Head Attention")
print(String(repeating: "-", count: 80))
let attnOut = tfmrBlock.attention(norm1Out, mask: attnMask)
eval(attnOut)

if !compare("Attention", attnOut, "debug_tf_attn.npy") {
    print("🎯 DIVERGENCE FOUND: Multi-Head Attention")
    print("   Hypothesis:")
    print("   - QK scaling factor (1/sqrt(d_head))")
    print("   - Attention mask application")
    print("   - Softmax axis or numerical stability")
    print()
    exit(1)
}

// STEP 3: Residual Connection 1
print("STEP 3: Residual Connection 1")
print(String(repeating: "-", count: 80))
let resid1Out = h + attnOut
eval(resid1Out)

if !compare("Residual 1", resid1Out, "debug_tf_resid1.npy") {
    print("🎯 DIVERGENCE FOUND: First Residual Connection")
    print("   (This should never fail if Attention passed)")
    print()
    exit(1)
}

// STEP 4: LayerNorm 2 (Pre-FFN) - Python calls this norm3
print("STEP 4: LayerNorm 2 (Pre-FFN)")
print(String(repeating: "-", count: 80))
let norm2Out = tfmrBlock.norm2(resid1Out)
eval(norm2Out)

if !compare("LayerNorm 2", norm2Out, "debug_tf_norm3.npy") {
    print("🎯 DIVERGENCE FOUND: LayerNorm 2 (Pre-FFN)")
    print("   Hypothesis: Epsilon mismatch or axis ordering")
    print()
    exit(1)
}

// STEP 5: Feed-Forward Network
print("STEP 5: Feed-Forward Network")
print(String(repeating: "-", count: 80))
let ffOut = tfmrBlock.ff(norm2Out)
eval(ffOut)

if !compare("FeedForward", ffOut, "debug_tf_ff.npy") {
    print("🎯 DIVERGENCE FOUND: Feed-Forward Network")
    print("   Hypothesis:")
    print("   - Activation function (GELU vs SiLU)")
    print("   - GLU variant (Gated Linear Unit)")
    print("   - Weight matrix ordering")
    print()
    exit(1)
}

// STEP 6: Residual Connection 2
print("STEP 6: Final Residual Connection")
print(String(repeating: "-", count: 80))
let finalOut = resid1Out + ffOut
eval(finalOut)

if !compare("Final Output", finalOut, "debug_tf_final.npy") {
    print("🎯 DIVERGENCE FOUND: Second Residual Connection")
    print("   (This should never fail if FFN passed)")
    print()
    exit(1)
}

// STEP 7: Full Forward Pass (Sanity Check)
print("STEP 7: Full Forward Pass (Sanity Check)")
print(String(repeating: "-", count: 80))
let fullOut = tfmrBlock(h, mask: attnMask)
eval(fullOut)

if !compare("Full Forward", fullOut, "debug_tf_final.npy") {
    print("❌ Full forward pass differs from manual trace")
    print("   (This indicates a structural issue in the Swift implementation)")
    print()
    exit(1)
}

print(String(repeating: "=", count: 80))
print("✅ ALL TRANSFORMER STAGES MATCH PYTHON!")
print("If you see this message, the transformer is actually working correctly.")
print("The divergence must be happening elsewhere (e.g., mask usage in full decoder).")
print(String(repeating: "=", count: 80))
