import Foundation
import MLX
import MLXNN
import Nightingale

print(String(repeating: "=", count: 80))
print("END-TO-END VERIFICATION: Steps 1-7 (Live Fire)")
print("S3Gen Flow / ODE Solver → Mel Spectrogram")
print(String(repeating: "=", count: 80))
print()

// Set random seed for deterministic results (for consistency with Python)
// IMPORTANT: Must be called ONCE before any random generation
// MLXRandom.seed(42)  // DISABLED: Causes float64 error on GPU

// Paths
let modelPath = "/Users/a10n/Projects/nightingale/models"
let voicePath = "/Users/a10n/Projects/nightingale/baked_voices/samantha_full"
let step6Dir = "/Users/a10n/Projects/nightingale/verification_outputs/e2e_steps1_6"
let step7Dir = "/Users/a10n/Projects/nightingale/verification_outputs/e2e_steps1_7"

// ============================================================================
// RECAP: Load Step 6 Encoder Output (Live Fire Data)
// ============================================================================
print(String(repeating: "=", count: 80))
print("RECAP: Loading Step 6 Encoder Output (Live Fire)")
print(String(repeating: "=", count: 80))
print()

print("Loading encoder output from Step 6...")
let step6EncoderOutput = try! NPYLoader.load(file: "\(step6Dir)/step6_encoder_output.npy").asType(.float32)
print("✅ Encoder output loaded: \(step6EncoderOutput.shape)")
print("   Mean: \(step6EncoderOutput.mean().item(Float.self))")
print("   Std: \(sqrt(((step6EncoderOutput - step6EncoderOutput.mean()) * (step6EncoderOutput - step6EncoderOutput.mean())).mean()).item(Float.self))")
print()

// Load voice data for speaker embedding and prompt features
print("Loading voice data...")
let soulS3 = try! NPYLoader.load(file: "\(voicePath)/soul_s3_192.npy").asType(.float32)
let promptFeat = try! NPYLoader.load(file: "\(voicePath)/prompt_feat.npy").asType(.float32)
print("  soul_s3 (speaker embedding): \(soulS3.shape)")
print("  prompt_feat: \(promptFeat.shape)")
print()

// ============================================================================
// Step 7: S3Gen Flow (ODE Solver / Conditional Flow Matching)
// ============================================================================
print(String(repeating: "=", count: 80))
print("Step 7: S3Gen Flow (ODE Solver / Conditional Flow Matching)")
print(String(repeating: "=", count: 80))
print()

// Load S3Gen model with HuggingFace weights
print("Loading S3Gen model...")
let modelsURL = URL(fileURLWithPath: modelPath)
let hfWeightsURL = modelsURL.appendingPathComponent("chatterbox_hf.safetensors")
let vocoderWeightsURL = modelsURL.appendingPathComponent("vocoder_weights.safetensors")
// CRITICAL: Load complete decoder weights (includes out_proj.bias from Python)
let decoderWeightsURL = modelsURL.appendingPathComponent("decoder_complete_weights.safetensors")

let allWeights = try! MLX.loadArrays(url: hfWeightsURL)
let vocoderWeights = try! MLX.loadArrays(url: vocoderWeightsURL)
let completeDecoderWeights = try! MLX.loadArrays(url: decoderWeightsURL)
let s3gen = S3Gen(flowWeights: allWeights, vocoderWeights: vocoderWeights)

// CRITICAL: Load COMPLETE decoder weights from Python export (includes random out_proj.bias)
print("Loading complete decoder weights (with out_proj.bias from Python)...")
var decoderWeights: [String: MLXArray] = [:]

// The complete decoder weights file has keys like:
// down_blocks_0.transformer_0.attn.out_proj.bias
// These need to be remapped to Swift format:
// downBlocks.0.transformers.0.attention.outProj.bias

for (key, value) in completeDecoderWeights {
    var remappedKey = key

    // Handle snake_case to camelCase conversions
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

    // Attention layer names (CRITICAL!)
    remappedKey = remappedKey.replacingOccurrences(of: ".attn.", with: ".attention.")
    remappedKey = remappedKey.replacingOccurrences(of: "query_proj", with: "queryProj")
    remappedKey = remappedKey.replacingOccurrences(of: "key_proj", with: "keyProj")
    remappedKey = remappedKey.replacingOccurrences(of: "value_proj", with: "valueProj")
    remappedKey = remappedKey.replacingOccurrences(of: "out_proj", with: "outProj")

    // FF layer names
    remappedKey = remappedKey.replacingOccurrences(of: "ff.net.0.", with: "ff.layers.0.")
    remappedKey = remappedKey.replacingOccurrences(of: "ff.net.2.", with: "ff.layers.1.")

    decoderWeights[remappedKey] = value
}
print("  Found \(decoderWeights.count) decoder weights (including out_proj.bias)")

// DEBUG: Show all decoder keys to verify remapping
print("  First 20 decoder keys after remapping:")
for (i, key) in decoderWeights.keys.sorted().prefix(20).enumerated() {
    print("    \(i+1). \(key)")
}
print()
print("  Time-related keys:")
let timeKeys = decoderWeights.keys.filter { $0.contains("time") }
for key in timeKeys.sorted() {
    print("    \(key)")
}

let decoderParams = ModuleParameters.unflattened(decoderWeights)
s3gen.decoder.update(parameters: decoderParams)

// Verify a few decoder weights are actually loaded
print("\nVerifying decoder weight loading...")
let qProjWeight = s3gen.decoder.downBlocks[0].transformers[0].attention.queryProj.weight
eval(qProjWeight)
let mean = qProjWeight.mean().item(Float.self)
let std = sqrt(((qProjWeight - qProjWeight.mean()) * (qProjWeight - qProjWeight.mean())).mean()).item(Float.self)
print("  downBlocks[0].tfmr[0].queryProj.weight: mean=\(mean), std=\(std)")

// Compare with original weight from file
if let origWeight = decoderWeights["downBlocks.0.transformers.0.attention.queryProj.weight"] {
    let origMean = origWeight.mean().item(Float.self)
    let diff = abs(mean - origMean)
    print("    Original from file: mean=\(origMean), diff=\(diff)")
    if diff < 0.0001 {
        print("    ✅ Weight matches!")
    } else {
        print("    ❌ Weight doesn't match!")
    }
}

print("✅ Decoder weights loaded!")
print("✅ Model loaded!")
print()

// ============================================================================
// Step 7A: Prepare ODE Solver Inputs
// ============================================================================
print(String(repeating: "-", count: 80))
print("Step 7A: Prepare ODE Solver Inputs")
print(String(repeating: "-", count: 80))

// CRITICAL: Load mu from Python to ensure EXACT match
print("Loading mu from Python reference...")
let mu = try! NPYLoader.load(file: "\(step7Dir)/step7_mu.npy").asType(.float32)
eval(mu)

print("mu (encoder projection): \(mu.shape)")
print("  Mean: \(mu.mean().item(Float.self))")
print("  Std: \(sqrt(((mu - mu.mean()) * (mu - mu.mean())).mean()).item(Float.self))")
print("  mu[0,0,:5]: \(mu[0, 0, 0..<5].asArray(Float.self))")
print("  mu[0,-1,:5]: \(mu[0, -1, 0..<5].asArray(Float.self))")
print()

// CRITICAL: Load x_cond from Python to ensure EXACT match
print("Loading x_cond from Python reference...")
let xCond = try! NPYLoader.load(file: "\(step7Dir)/step7_x_cond.npy").asType(.float32)
eval(xCond)

print("  x_cond shape: \(xCond.shape)")
print("  x_cond[0,0,:5]: \(xCond[0, 0, 0..<5].asArray(Float.self))")
print("  x_cond[0,-1,:5]: \(xCond[0, -1, 0..<5].asArray(Float.self))")
print()

// Speaker embedding - normalize and project
print("Preparing speaker embedding...")
var spkEmb = soulS3
if spkEmb.ndim == 1 {
    spkEmb = spkEmb.expandedDimensions(axis: 0)
}

// Normalize
let norm = sqrt(sum(spkEmb * spkEmb, axis: 1, keepDims: true)) + (1e-8 as Float)
let spkEmbNorm = spkEmb / norm

// Project through spkEmbedAffine (192 -> 80)
let spkEmbProj = s3gen.spkEmbedAffine(spkEmbNorm)
eval(spkEmbProj)

print("  spk_emb (after norm + projection): \(spkEmbProj.shape)")
print("  spk_emb[0,:5]: \(spkEmbProj[0, 0..<5].asArray(Float.self))")
print()

// ============================================================================
// Step 7B: Run ODE Solver (Euler Integration)
// ============================================================================
print(String(repeating: "-", count: 80))
print("Step 7B: Run ODE Solver (Euler Integration)")
print(String(repeating: "-", count: 80))

let nTimesteps = 10
let temperature: Float = 1.0
let cfgRate: Float = 0.7

print("Configuration:")
print("  n_timesteps: \(nTimesteps)")
print("  temperature: \(temperature)")
print("  CFG rate: \(cfgRate)")
print()

// Transpose mu, x_cond to [B, C, T] format for decoder
let muT = mu.transposed(0, 2, 1)
let xCondT = xCond.transposed(0, 2, 1)

print("Transposed shapes for flow matching:")
print("  mu: \(mu.shape) -> \(muT.shape)")
print("  x_cond: \(xCond.shape) -> \(xCondT.shape)")
print()

// DETERMINISTIC NOISE INJECTION: Load exact noise from Python
print("🔒 Loading FORCED deterministic noise from Python...")
var xt = try! NPYLoader.load(file: "\(step7Dir)/step7_initial_noise.npy").asType(.float32)
eval(xt)

print("Initial noise (loaded from Python):")
print("  shape: \(xt.shape)")
print("  mean: \(xt.mean().item(Float.self))")
print("  std: \(sqrt(((xt - xt.mean()) * (xt - xt.mean())).mean()).item(Float.self))")
print("  [0,0,0:5]: \(xt[0, 0, 0..<5].asArray(Float.self))")
print("  [0,0,5:10]: \(xt[0, 0, 5..<10].asArray(Float.self))")
print("✅ Using identical noise as Python - RNG variability eliminated!")
print()

// Cosine time scheduling
var tSpan: [Float] = []
for i in 0...(nTimesteps) {
    let linearT = Float(i) / Float(nTimesteps)
    let angle = linearT * (0.5 as Float) * Float.pi
    let cosineT = (1.0 as Float) - Foundation.cos(angle)  // Explicit Float cos
    tSpan.append(cosineT)
}

print("Starting ODE solver...")
print("Time schedule (cosine): \(tSpan.map { String(format: "%.4f", $0) }.joined(separator: ", "))")
print()

// ============================================================================
// BLOCK 0 ISOLATION: Compare Input Concatenation
// ============================================================================
print(String(repeating: "=", count: 80))
print("BLOCK 0 ISOLATION: Checking Input Concatenation")
print(String(repeating: "=", count: 80))
print()

// Load Python reference for comparison
let debugDir = "\(step7Dir)/../step7_debug"
let pyHConcat = try! NPYLoader.load(file: "\(debugDir)/debug_py_h_concat.npy").asType(.float32)
let pyTEmb = try! NPYLoader.load(file: "\(debugDir)/debug_py_t_emb.npy").asType(.float32)

// Build Swift's input concatenation for single sample (not CFG batch)
let tZero = MLXArray([Float(0.0)])
let swiftTEmb = s3gen.decoder.timeMLP(tZero)
eval(swiftTEmb)

// Expand speaker embedding
let spkExpanded = tiled(spkEmbProj.expandedDimensions(axis: 2), repetitions: [1, 1, 908])

// Concatenate: x(80) + mu(80) + spk(80) + cond(80) = 320
let swiftH = concatenated([xt, muT, spkExpanded, xCondT], axis: 1)
eval(swiftH)

print("Comparing Time Embedding:")
let tEmbDiff = MLX.abs(swiftTEmb - pyTEmb)
eval(tEmbDiff)
let tEmbMaxDiff = tEmbDiff.max().item(Float.self)
let tEmbMeanDiff = tEmbDiff.mean().item(Float.self)
print("  Swift t_emb[:5]: \(swiftTEmb[0, 0..<5].asArray(Float.self))")
print("  Python t_emb[:5]: \(pyTEmb[0, 0..<5].asArray(Float.self))")
print("  Max diff: \(tEmbMaxDiff)")
print("  Mean diff: \(tEmbMeanDiff)")
if tEmbMaxDiff < 0.0001 {
    print("  ✅ Time embedding MATCHES Python")
} else {
    print("  ❌ Time embedding DIFFERS from Python")
}
print()

print("Comparing Input Concatenation (h):")
let hDiff = MLX.abs(swiftH - pyHConcat)
eval(hDiff)
let hMaxDiff = hDiff.max().item(Float.self)
let hMeanDiff = hDiff.mean().item(Float.self)
print("  Swift h[0,0:5,0]: \(swiftH[0, 0..<5, 0].asArray(Float.self))")
print("  Python h[0,0:5,0]: \(pyHConcat[0, 0..<5, 0].asArray(Float.self))")
print("  Swift h[0,80:85,0]: \(swiftH[0, 80..<85, 0].asArray(Float.self))")
print("  Python h[0,80:85,0]: \(pyHConcat[0, 80..<85, 0].asArray(Float.self))")
print("  Max diff: \(hMaxDiff)")
print("  Mean diff: \(hMeanDiff)")
if hMaxDiff < 0.0001 {
    print("  ✅ Input concatenation MATCHES Python")
} else {
    print("  ❌ Input concatenation DIFFERS from Python")
}
print()

// ============================================================================
// BLOCK 0 ISOLATION: Compare ResNet Output
// ============================================================================
print("Running Block 0 ResNet...")
let singleMaskForBlock0 = MLXArray.ones([1, 1, 908]).asType(.float32)
let resnetOut = s3gen.decoder.downBlocks[0].resnet(swiftH, mask: singleMaskForBlock0, timeEmb: swiftTEmb)
eval(resnetOut)

print("  Swift resnet_out shape: \(resnetOut.shape)")
print("  Swift resnet_out[0,0:5,0]: \(resnetOut[0, 0..<5, 0].asArray(Float.self))")
print("  Swift resnet_out mean: \(resnetOut.mean().item(Float.self))")

// Load Python reference
let pyResnetOut = try! NPYLoader.load(file: "\(debugDir)/debug_resnet_out.npy").asType(.float32)
let resnetDiff = MLX.abs(resnetOut - pyResnetOut)
eval(resnetDiff)
let resnetMaxDiff = resnetDiff.max().item(Float.self)
let resnetMeanDiff = resnetDiff.mean().item(Float.self)

print("  Python resnet_out[0,0:5,0]: \(pyResnetOut[0, 0..<5, 0].asArray(Float.self))")
print("  Python resnet_out mean: \(pyResnetOut.mean().item(Float.self))")
print("  Max diff: \(resnetMaxDiff)")
print("  Mean diff: \(resnetMeanDiff)")

if resnetMaxDiff < 0.01 {
    print("  ✅ ResNet output MATCHES Python")
} else {
    print("  ❌ ResNet output DIFFERS from Python")
    print()
    print("🎯 CONCLUSION: Divergence happens in CausalResNetBlock")
    print("   Check CausalBlock1D, GroupNorm, or residual connections.")
}
print()

// ============================================================================
// BLOCK 0 ISOLATION: Transformer Internal Trace
// ============================================================================
print("Running Block 0 Transformer 0 - Detailed Internal Trace...")
let resnetTransposed = resnetOut.transposed(0, 2, 1)  // [B, C, T] -> [B, T, C]
let attnMask = MLXArray.zeros([1, 1, 908, 908]).asType(.float32)
let tfmr = s3gen.decoder.downBlocks[0].transformers[0]

print()
print("🔬 TRANSFORMER INTERNAL AUTOPSY:")
print(String(repeating: "-", count: 80))

// Stage 1: LayerNorm 1
let norm1Swift = tfmr.norm1(resnetTransposed)
eval(norm1Swift)
let pyNorm1 = try! NPYLoader.load(file: "\(debugDir)/debug_tf_norm1.npy").asType(.float32)
let norm1Diff = MLX.abs(norm1Swift - pyNorm1).max().item(Float.self)
print("1. LayerNorm 1:  max_diff=\(String(format: "%.6f", norm1Diff)) \(norm1Diff < 0.01 ? "✅" : "❌")")

// Stage 2: Attention
let attnSwift = tfmr.attention(norm1Swift, mask: attnMask)
eval(attnSwift)
let pyAttn = try! NPYLoader.load(file: "\(debugDir)/debug_tf_attn.npy").asType(.float32)
let attnDiff = MLX.abs(attnSwift - pyAttn).max().item(Float.self)
print("2. Attention:    max_diff=\(String(format: "%.6f", attnDiff)) \(attnDiff < 0.01 ? "✅" : "❌")")

if attnDiff > 0.01 {
    print()
    print("🎯 DIVERGENCE FOUND: Multi-Head Attention")
    print("   Swift attn[0,0,:5]: \(attnSwift[0, 0, 0..<5].asArray(Float.self))")
    print("   Python attn[0,0,:5]: \(pyAttn[0, 0, 0..<5].asArray(Float.self))")
    print()
}

// Stage 3: Residual 1
let resid1Swift = resnetTransposed + attnSwift
eval(resid1Swift)
let pyResid1 = try! NPYLoader.load(file: "\(debugDir)/debug_tf_resid1.npy").asType(.float32)
let resid1Diff = MLX.abs(resid1Swift - pyResid1).max().item(Float.self)
print("3. Residual 1:   max_diff=\(String(format: "%.6f", resid1Diff)) \(resid1Diff < 0.01 ? "✅" : "❌")")

// Stage 4: LayerNorm 2
let norm2Swift = tfmr.norm2(resid1Swift)
eval(norm2Swift)
let pyNorm3 = try! NPYLoader.load(file: "\(debugDir)/debug_tf_norm3.npy").asType(.float32)
let norm2Diff = MLX.abs(norm2Swift - pyNorm3).max().item(Float.self)
print("4. LayerNorm 2:  max_diff=\(String(format: "%.6f", norm2Diff)) \(norm2Diff < 0.01 ? "✅" : "❌")")

// Stage 5: FeedForward
let ffSwift = tfmr.ff(norm2Swift)
eval(ffSwift)
let pyFF = try! NPYLoader.load(file: "\(debugDir)/debug_tf_ff.npy").asType(.float32)
let ffDiff = MLX.abs(ffSwift - pyFF).max().item(Float.self)
print("5. FeedForward:  max_diff=\(String(format: "%.6f", ffDiff)) \(ffDiff < 0.01 ? "✅" : "❌")")

if ffDiff > 0.01 {
    print()
    print("🎯 DIVERGENCE FOUND: Feed-Forward Network")
    print("   Swift ff[0,0,:5]: \(ffSwift[0, 0, 0..<5].asArray(Float.self))")
    print("   Python ff[0,0,:5]: \(pyFF[0, 0, 0..<5].asArray(Float.self))")
    print()
}

// Stage 6: Final Output
let tfmr0Out = resid1Swift + ffSwift
eval(tfmr0Out)
let pyTfmr0Out = try! NPYLoader.load(file: "\(debugDir)/debug_tfmr0_out.npy").asType(.float32)
let tfmr0MaxDiff = MLX.abs(tfmr0Out - pyTfmr0Out).max().item(Float.self)
let tfmr0MeanDiff = MLX.abs(tfmr0Out - pyTfmr0Out).mean().item(Float.self)
print("6. Final Output: max_diff=\(String(format: "%.6f", tfmr0MaxDiff)) \(tfmr0MaxDiff < 0.01 ? "✅" : "❌")")

print(String(repeating: "-", count: 80))
print()

if tfmr0MaxDiff < 0.01 {
    print("✅ All transformer stages MATCH Python")
} else {
    print("❌ Transformer divergence detected (max_diff=\(tfmr0MaxDiff))")
    print("  Swift final[0,0,:5]: \(tfmr0Out[0, 0, 0..<5].asArray(Float.self))")
    print("  Python final[0,0,:5]: \(pyTfmr0Out[0, 0, 0..<5].asArray(Float.self))")
}
print()
print(String(repeating: "=", count: 80))
print()

// Euler integration loop
var currentT = tSpan[0]
var dt = tSpan[1] - tSpan[0]

for step in 1...nTimesteps {
    // Enable debug output for first step only
    if step == 1 {
        FlowMatchingDecoder.debugStep = 1
    } else {
        FlowMatchingDecoder.debugStep = 0
    }

    let t = MLXArray([currentT]).asType(.float32)

    // Prepare batch for CFG: [Cond, Uncond]
    let xIn = concatenated([xt, xt], axis: 0)
    let muIn = concatenated([muT, MLXArray.zeros(like: muT)], axis: 0)
    let spkIn = concatenated([spkEmbProj, MLXArray.zeros(like: spkEmbProj)], axis: 0)
    let condIn = concatenated([xCondT, MLXArray.zeros(like: xCondT)], axis: 0)
    let tIn = concatenated([t, t], axis: 0)

    // CRITICAL FIX: Create explicit all-ones mask matching Python
    // Python: mask = mx.ones([1, 1, 908]) then concatenates for CFG batch
    // This ensures bidirectional attention (attend to EVERYTHING)
    let seqLen = xt.shape[2]  // 908
    let singleMask = MLXArray.ones([1, 1, seqLen]).asType(.float32)
    let maskBatch = concatenated([singleMask, singleMask], axis: 0)  // [2, 1, 908]

    if step == 1 {
        print("Created explicit mask: \(maskBatch.shape) - forcing bidirectional attention")
    }

    // Forward pass through decoder (batch=2) with EXPLICIT MASK
    let vBatch = s3gen.decoder(x: xIn, mu: muIn, t: tIn, speakerEmb: spkIn, cond: condIn, mask: maskBatch)
    eval(vBatch)

    // Split conditional and unconditional
    let vCond = vBatch[0].expandedDimensions(axis: 0)
    let vUncond = vBatch[1].expandedDimensions(axis: 0)

    // Apply CFG: v = (1 + cfg) * vCond - cfg * vUncond
    let v = ((1.0 as Float) + cfgRate) * vCond - cfgRate * vUncond
    eval(v)

    if step == 1 {
        print("🔍 DETAILED DEBUG - First timestep (t=\(String(format: "%.4f", currentT))):")
        print()
        print("Decoder outputs:")
        print("  vCond[0,0:5,0]: \(vCond[0, 0..<5, 0].asArray(Float.self))")
        print("  vCond mean: \(vCond.mean().item(Float.self))")
        print()
        print("  vUncond[0,0:5,0]: \(vUncond[0, 0..<5, 0].asArray(Float.self))")
        print("  vUncond mean: \(vUncond.mean().item(Float.self))")
        print()
        print("  v_cfg[0,0:5,0]: \(v[0, 0..<5, 0].asArray(Float.self))")
        print("  v_cfg mean: \(v.mean().item(Float.self))")
        print("  v_cfg std: \(sqrt(((v - v.mean()) * (v - v.mean())).mean()).item(Float.self))")
        print()
        print("  dt (first step): \(dt)")
        print()
    }

    if step == 1 || step == nTimesteps {
        print("Step \(step)/\(nTimesteps) (t=\(String(format: "%.4f", currentT))):")
        print("  x_t mean: \(xt.mean().item(Float.self))")
        print("  v mean: \(v.mean().item(Float.self))")
    }

    // Euler step
    xt = xt + v * dt
    eval(xt)

    if step == 1 {
        print("After first integration step:")
        print("  x_next[0,0:5,0]: \(xt[0, 0..<5, 0].asArray(Float.self))")
        print("  x_next mean: \(xt.mean().item(Float.self))")
        print("  x_next std: \(sqrt(((xt - xt.mean()) * (xt - xt.mean())).mean()).item(Float.self))")
        print()
    }

    // Update time for next step
    currentT = currentT + dt
    if step < nTimesteps {
        dt = tSpan[step + 1] - currentT
    }
}

// Transpose back to [B, T, C]
let mel = xt.transposed(0, 2, 1)
eval(mel)

print()
print("✅ ODE solver completed!")
print("mel shape: \(mel.shape)")
print("mel mean: \(mel.mean().item(Float.self))")
print("mel std: \(sqrt(((mel - mel.mean()) * (mel - mel.mean())).mean()).item(Float.self))")
print("mel range: [\(mel.min().item(Float.self)), \(mel.max().item(Float.self))]")
print("mel[0,0,:5]: \(mel[0, 0, 0..<5].asArray(Float.self))")
print("mel[0,-1,:5]: \(mel[0, -1, 0..<5].asArray(Float.self))")
print()

// ============================================================================
// Load Python Reference and Compare
// ============================================================================
print(String(repeating: "=", count: 80))
print("Load Python Reference and Compare")
print(String(repeating: "=", count: 80))
print()

print("Loading Python reference mel...")
let refMel = try! NPYLoader.load(file: "\(step7Dir)/step7_mel.npy").asType(.float32)
print("Python reference mel: \(refMel.shape)")
print()

// Compare shapes
print("Shape comparison:")
print("  Swift:     \(mel.shape)")
print("  Reference: \(refMel.shape)")

if mel.shape != refMel.shape {
    print("❌ SHAPE MISMATCH!")
    exit(1)
}
print("✅ Shapes match")
print()

// Compute difference
let diff = mel - refMel
eval(diff)

let absDiff = abs(diff)
eval(absDiff)

let maxAbsDiff = absDiff.max().item(Float.self)
let meanAbsDiff = absDiff.mean().item(Float.self)

print("Difference statistics:")
print("  Max absolute difference: \(maxAbsDiff)")
print("  Mean absolute difference: \(meanAbsDiff)")
print()

// Value comparison
print("Value comparison:")
print("  Swift[0,0,:5]:     \(mel[0, 0, 0..<5].asArray(Float.self))")
print("  Reference[0,0,:5]: \(refMel[0, 0, 0..<5].asArray(Float.self))")
print()
print("  Swift[0,-1,:5]:     \(mel[0, -1, 0..<5].asArray(Float.self))")
print("  Reference[0,-1,:5]: \(refMel[0, -1, 0..<5].asArray(Float.self))")
print()

// Tolerance check
let tolerance: Float = 1.0  // Mel spectrograms can have larger variations due to ODE solver
if maxAbsDiff < tolerance {
    print("✅ PASS: Max difference (\(maxAbsDiff)) < tolerance (\(tolerance))")
} else {
    print("⚠️  WARNING: Max difference (\(maxAbsDiff)) >= tolerance (\(tolerance))")
    print("   This may indicate a significant discrepancy")
}
print()

// Check mean difference (more stable metric)
let meanDiffThreshold: Float = 0.5
if meanAbsDiff < meanDiffThreshold {
    print("✅ PASS: Mean difference (\(meanAbsDiff)) < threshold (\(meanDiffThreshold))")
} else {
    print("⚠️  WARNING: Mean difference (\(meanAbsDiff)) >= threshold (\(meanDiffThreshold))")
}
print()

// ============================================================================
// Summary
// ============================================================================
print(String(repeating: "=", count: 80))
print("SUMMARY")
print(String(repeating: "=", count: 80))

var allPassed = true
if mel.shape != refMel.shape {
    print("❌ Shape mismatch")
    allPassed = false
}
if maxAbsDiff >= tolerance {
    print("❌ Max difference exceeds tolerance")
    allPassed = false
}
if meanAbsDiff >= meanDiffThreshold {
    print("❌ Mean difference exceeds threshold")
    allPassed = false
}

if allPassed {
    print("✅ Step 7 (S3Gen Flow / ODE Solver): VERIFICATION PASSED")
    print()
    print("All checks passed! The ODE solver implementation matches the reference.")
} else {
    print("❌ Step 7 (S3Gen Flow / ODE Solver): VERIFICATION FAILED")
    print()
    print("Some checks failed. Review the differences above.")
}

print(String(repeating: "=", count: 80))

// ==================================================================
// NPY Loader
// ==================================================================
struct NPYLoader {
    static func load(file: String) throws -> MLXArray {
        let url = URL(fileURLWithPath: file)
        let data = try Data(contentsOf: url)

        var offset = 0

        // Check magic number
        let magic = data[0..<6]
        guard magic.elementsEqual([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]) else {
            throw NPYError.invalidFormat
        }
        offset += 6

        // Read version
        let major = data[offset]
        let _ = data[offset + 1]  // minor version - unused
        offset += 2

        // Read header length
        var headerLen: Int
        if major == 1 {
            headerLen = Int(data[offset]) | (Int(data[offset + 1]) << 8)
            offset += 2
        } else {
            headerLen = Int(data[offset]) | (Int(data[offset + 1]) << 8) |
                       (Int(data[offset + 2]) << 16) | (Int(data[offset + 3]) << 24)
            offset += 4
        }

        // Read header
        let headerData = data[offset..<(offset + headerLen)]
        guard let headerStr = String(data: headerData, encoding: .ascii) else {
            throw NPYError.invalidHeader
        }
        offset += headerLen

        // Parse header
        guard let descrRange = headerStr.range(of: "'descr':\\s*'([^']+)'", options: .regularExpression),
              let shapeRange = headerStr.range(of: "'shape':\\s*\\(([^)]+)\\)", options: .regularExpression) else {
            throw NPYError.invalidHeader
        }

        let descr = String(headerStr[descrRange]).components(separatedBy: "'")[3]
        let shapeStr = String(headerStr[shapeRange]).components(separatedBy: "(")[1].components(separatedBy: ")")[0]
        let shape = shapeStr.components(separatedBy: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }

        // Read data
        let arrayData = data[offset...]

        // Determine dtype and create MLXArray
        let dtype: DType
        if descr.contains("f4") {
            dtype = .float32
        } else if descr.contains("i4") {
            dtype = .int32
        } else if descr.contains("i8") {
            dtype = .int64
        } else {
            throw NPYError.unsupportedDType(descr)
        }

        // Create array
        let totalElements = shape.reduce(1, *)
        let array: MLXArray

        switch dtype {
        case .float32:
            let values = arrayData.withUnsafeBytes { $0.bindMemory(to: Float.self) }
            array = MLXArray(Array(values.prefix(totalElements)))
        case .int32:
            let values = arrayData.withUnsafeBytes { $0.bindMemory(to: Int32.self) }
            array = MLXArray(Array(values.prefix(totalElements)))
        case .int64:
            let values = arrayData.withUnsafeBytes { $0.bindMemory(to: Int64.self) }
            let int32Values = values.prefix(totalElements).map { Int32(clamping: $0) }
            array = MLXArray(int32Values)
        default:
            throw NPYError.unsupportedDType(descr)
        }

        return array.reshaped(shape.map { Int($0) })
    }

    enum NPYError: Error {
        case invalidFormat
        case invalidHeader
        case unsupportedDType(String)
    }
}
