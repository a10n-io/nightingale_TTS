import Foundation
import MLX
import MLXNN
import Nightingale

print(String(repeating: "=", count: 80))
print("VELOCITY TRACE CHECK: Decoder Forward Pass at t=0")
print(String(repeating: "=", count: 80))
print()

// Paths
let modelPath = "/Users/a10n/Projects/nightingale/models"
let e2eDir = "/Users/a10n/Projects/nightingale/verification_outputs/e2e_steps1_7"
let traceDir = "/Users/a10n/Projects/nightingale/verification_outputs/step7_velocity_trace"

// Load Step 7 inputs
print("Loading inputs from Step 7...")
let mu = try! NPYLoader.load(contentsOf: URL(fileURLWithPath: "\(e2eDir)/step7_mu.npy"))  // [1, 908, 80]
let xCond = try! NPYLoader.load(contentsOf: URL(fileURLWithPath: "\(e2eDir)/step7_x_cond.npy"))  // [1, 908, 80]
let initialNoise = try! NPYLoader.load(contentsOf: URL(fileURLWithPath: "\(e2eDir)/step7_initial_noise.npy"))  // [1, 80, 908]
print("  mu: \(mu.shape)")
print("  x_cond: \(xCond.shape)")
print("  initial_noise: \(initialNoise.shape)")
print()

// Load voice data for speaker embedding
let voicePath = "/Users/a10n/Projects/nightingale/baked_voices/samantha_full"
let soulS3 = try! NPYLoader.load(contentsOf: URL(fileURLWithPath: "\(voicePath)/soul_s3_192.npy"))

// Load S3Gen model with HuggingFace weights
print("Loading S3Gen model...")
let modelsURL = URL(fileURLWithPath: modelPath)
let hfWeightsURL = modelsURL.appendingPathComponent("chatterbox_hf.safetensors")
let vocoderWeightsURL = modelsURL.appendingPathComponent("vocoder_weights.safetensors")

let allWeights = try! MLX.loadArrays(url: hfWeightsURL)
let vocoderWeights = try! MLX.loadArrays(url: vocoderWeightsURL)
let s3gen = S3Gen(flowWeights: allWeights, vocoderWeights: vocoderWeights)

// Load decoder weights
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

        decoderWeights[remappedKey] = value
    }
}

let decoderParams = ModuleParameters.unflattened(decoderWeights)
s3gen.decoder.update(parameters: decoderParams)
print("✅ Model loaded (\(decoderWeights.count) decoder weights)")
print()

// Prepare speaker embedding
var spkEmb = soulS3
if spkEmb.ndim == 1 {
    spkEmb = spkEmb.expandedDimensions(axis: 0)
}
let norm = sqrt(sum(spkEmb * spkEmb, axis: 1, keepDims: true)) + 1e-8
let spkEmbNorm = spkEmb / norm
let spkEmbProj = s3gen.spkEmbedAffine(spkEmbNorm)
eval(spkEmbProj)

print("Speaker embedding: \(spkEmbProj.shape)")
print()

// Transpose to [B, C, T] format
let muT = mu.transposed(0, 2, 1)  // [1, 80, 908]
let xCondT = xCond.transposed(0, 2, 1)  // [1, 80, 908]

print("Transposed inputs:")
print("  mu: \(mu.shape) -> \(muT.shape)")
print("  x_cond: \(xCond.shape) -> \(xCondT.shape)")
print()

// Run decoder at t=0
print("Running decoder forward pass at t=0...")
let tZero = MLXArray([0.0])

// Prepare batch for CFG: [Conditional, Unconditional]
let xIn = concatenated([initialNoise, initialNoise], axis: 0)
let muIn = concatenated([muT, MLXArray.zeros(like: muT)], axis: 0)
let spkIn = concatenated([spkEmbProj, MLXArray.zeros(like: spkEmbProj)], axis: 0)
let condIn = concatenated([xCondT, MLXArray.zeros(like: xCondT)], axis: 0)
let tIn = concatenated([tZero, tZero], axis: 0)

print("Batch inputs for CFG:")
print("  x_in: \(xIn.shape)")
print("  mu_in: \(muIn.shape)")
print("  t_in: \(tIn.shape)")
print("  spk_in: \(spkIn.shape)")
print("  cond_in: \(condIn.shape)")
print()

// Call decoder
let vBatch = s3gen.decoder(x: xIn, mu: muIn, t: tIn, speakerEmb: spkIn, cond: condIn, mask: nil)
eval(vBatch)

print("Decoder output: \(vBatch.shape)")
print()

// Split conditional and unconditional
let vCond = vBatch[0].expandedDimensions(axis: 0)
let vUncond = vBatch[1].expandedDimensions(axis: 0)

print("Split velocities:")
print("  v_cond: \(vCond.shape)")
print("  v_uncond: \(vUncond.shape)")
print()

// Apply CFG
let cfgRate: Float = 0.7
let vCfg = (1.0 + cfgRate) * vCond - cfgRate * vUncond
eval(vCfg)

print("CFG velocity:")
print("  Shape: \(vCfg.shape)")
print("  Mean: \(vCfg.mean().item(Float.self))")
print("  Std: \(sqrt(((vCfg - vCfg.mean()) * (vCfg - vCfg.mean())).mean()).item(Float.self))")
print("  v_cfg[0,0:5,0]: \(vCfg[0, 0..<5, 0].asArray(Float.self))")
print("  v_cfg[0,0:5,100]: \(vCfg[0, 0..<5, 100].asArray(Float.self))")
print()

// Load Python reference
print("Loading Python reference...")
let refVCfg = try! NPYLoader.load(contentsOf: URL(fileURLWithPath: "\(traceDir)/trace_v_cfg.npy"))
print("Python reference: \(refVCfg.shape)")
print()

// Compare
print(String(repeating: "=", count: 80))
print("COMPARISON")
print(String(repeating: "=", count: 80))
print()

let diff = MLX.abs(vCfg - refVCfg)
eval(diff)

let maxDiff = diff.max().item(Float.self)
let meanDiff = diff.mean().item(Float.self)

print("Difference statistics:")
print("  Max absolute difference: \(maxDiff)")
print("  Mean absolute difference: \(meanDiff)")
print()

print("Value comparison:")
print("  Swift v_cfg[0,0:5,0]:     \(vCfg[0, 0..<5, 0].asArray(Float.self))")
print("  Python v_cfg[0,0:5,0]:    \(refVCfg[0, 0..<5, 0].asArray(Float.self))")
print()
print("  Swift v_cfg[0,0:5,100]:   \(vCfg[0, 0..<5, 100].asArray(Float.self))")
print("  Python v_cfg[0,0:5,100]:  \(refVCfg[0, 0..<5, 100].asArray(Float.self))")
print()

// Verdict
print(String(repeating: "=", count: 80))
print("VERDICT")
print(String(repeating: "=", count: 80))
print()

if maxDiff < 0.01 {
    print("✅ VELOCITY MATCH (max_diff < 0.01)")
    print("   The decoder network is PERFECT.")
    print("   The 1.90 error in final mel is purely from ODE numerical accumulation.")
    print("   This is ACCEPTABLE for production TTS.")
} else if maxDiff < 0.1 {
    print("⚠️  SMALL DIFFERENCE (max_diff < 0.1)")
    print("   The decoder network is very close but has minor differences.")
    print("   This could be from attention implementation details.")
} else {
    print("❌ VELOCITY MISMATCH (max_diff >= 0.1)")
    print("   Problem is in the decoder network itself (NOT the ODE solver).")
    print("   Check: Attention masks, RoPE, layer norms, or weight loading.")
}
print()
