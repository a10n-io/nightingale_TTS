import Foundation
import MLX
import MLXNN
import Nightingale

print(String(repeating: "=", count: 80))
print("LIVE-FIRE E2E TEST - Swift")
print("Comparing against Python canonical output")
print(String(repeating: "=", count: 80))
print()

// Paths
let modelPath = "/Users/a10n/Projects/nightingale/models"
let liveFireDir = "/Users/a10n/Projects/nightingale/verification_outputs/live_fire"

// Configuration - MUST MATCH Python
let nTimesteps = 10
let cfgRate: Float = 0.7

print("Configuration:")
print("  ODE timesteps: \(nTimesteps)")
print("  CFG rate: \(cfgRate)")
print()

// ============================================================================
// Load Model with COMPLETE Decoder Weights from Python
// ============================================================================
print(String(repeating: "=", count: 80))
print("Loading Model with Python's Decoder Weights")
print(String(repeating: "=", count: 80))
print()

let modelsURL = URL(fileURLWithPath: modelPath)
let hfWeightsURL = modelsURL.appendingPathComponent("chatterbox_hf.safetensors")
let vocoderWeightsURL = modelsURL.appendingPathComponent("vocoder_weights.safetensors")
let decoderWeightsURL = modelsURL.appendingPathComponent("decoder_complete_weights.safetensors")

let allWeights = try! MLX.loadArrays(url: hfWeightsURL)
let vocoderWeights = try! MLX.loadArrays(url: vocoderWeightsURL)
let completeDecoderWeights = try! MLX.loadArrays(url: decoderWeightsURL)

let s3gen = S3Gen(flowWeights: allWeights, vocoderWeights: vocoderWeights)

// Remap decoder weights from Python format to Swift format
var decoderWeights: [String: MLXArray] = [:]
for (key, value) in completeDecoderWeights {
    var remappedKey = key

    // Snake_case to camelCase conversions
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

    // Attention layer names
    remappedKey = remappedKey.replacingOccurrences(of: ".attn.", with: ".attention.")
    remappedKey = remappedKey.replacingOccurrences(of: "query_proj", with: "queryProj")
    remappedKey = remappedKey.replacingOccurrences(of: "key_proj", with: "keyProj")
    remappedKey = remappedKey.replacingOccurrences(of: "value_proj", with: "valueProj")
    remappedKey = remappedKey.replacingOccurrences(of: "out_proj", with: "outProj")

    // FF layer names
    remappedKey = remappedKey.replacingOccurrences(of: "ff.net.0.", with: "ff.layers.0.")
    remappedKey = remappedKey.replacingOccurrences(of: "ff.net.2.", with: "ff.layers.1.")

    // CRITICAL: Python's pre-FFN LayerNorm is called "norm3", Swift calls it "norm2"
    remappedKey = remappedKey.replacingOccurrences(of: ".norm3.", with: ".norm2.")

    decoderWeights[remappedKey] = value
}

let decoderParams = ModuleParameters.unflattened(decoderWeights)
s3gen.decoder.update(parameters: decoderParams)
print("Loaded \(decoderWeights.count) decoder weights (including out_proj.bias)")
print()

// ============================================================================
// Load Inputs from Python (EXACT same values)
// ============================================================================
print(String(repeating: "=", count: 80))
print("Loading Inputs from Python Canonical Test")
print(String(repeating: "=", count: 80))
print()

// Python saved in [B, T, D] format
let muBTD = try! NPYLoader.load(file: "\(liveFireDir)/mu.npy").asType(.float32)
let xCondBTD = try! NPYLoader.load(file: "\(liveFireDir)/x_cond.npy").asType(.float32)
let spkEmbProj = try! NPYLoader.load(file: "\(liveFireDir)/spk_emb_proj.npy").asType(.float32)

print("Loaded from Python:")
print("  mu: \(muBTD.shape)")
print("  x_cond: \(xCondBTD.shape)")
print("  spk_emb_proj: \(spkEmbProj.shape)")
print()

// Transpose to [B, D, T] for decoder
let mu = muBTD.transposed(0, 2, 1)
let xCond = xCondBTD.transposed(0, 2, 1)
eval(mu)
eval(xCond)

print("Transposed for decoder:")
print("  mu: \(mu.shape)")
print("  x_cond: \(xCond.shape)")
print()

// ============================================================================
// Run ODE Solver (Euler Integration)
// ============================================================================
print(String(repeating: "=", count: 80))
print("Running ODE Solver")
print(String(repeating: "=", count: 80))
print()

let seqLen = mu.shape[2]

// Load the EXACT same initial noise that Python used (deterministic fixed noise)
let initialNoise = try! NPYLoader.load(file: "\(liveFireDir)/initial_noise.npy").asType(.float32)
var xt = initialNoise
eval(xt)

print("Initial noise: \(xt.shape)")
print("  Mean: \(xt.mean().item(Float.self))")
print("  First 5: \(xt[0, 0, 0..<5].asArray(Float.self))")
print()

// Cosine time scheduling
var tSpan: [Float] = []
for i in 0...nTimesteps {
    let linearT = Float(i) / Float(nTimesteps)
    let angle = linearT * 0.5 * Float.pi
    let cosineT = 1.0 - Foundation.cos(angle)
    tSpan.append(cosineT)
}

print("Time schedule: \(tSpan.map { String(format: "%.4f", $0) }.joined(separator: ", "))")
print()

// Euler integration
var currentT = tSpan[0]
var dt = tSpan[1] - tSpan[0]

// Load Python debug references for comparison
let pyTEmb = try! NPYLoader.load(file: "\(liveFireDir)/debug_t_emb.npy").asType(.float32)
let pyHConcat = try! NPYLoader.load(file: "\(liveFireDir)/debug_h_concat.npy").asType(.float32)
let pyHResnet = try! NPYLoader.load(file: "\(liveFireDir)/debug_h_resnet.npy").asType(.float32)
let pyVelocity = try! NPYLoader.load(file: "\(liveFireDir)/debug_velocity.npy").asType(.float32)
let pyXNext = try! NPYLoader.load(file: "\(liveFireDir)/debug_x_next.npy").asType(.float32)

for step in 1...nTimesteps {
    let t = MLXArray([currentT]).asType(.float32)

    // Prepare batch for CFG: [Cond, Uncond]
    let xIn = concatenated([xt, xt], axis: 0)
    let muIn = concatenated([mu, MLXArray.zeros(like: mu)], axis: 0)
    let spkIn = concatenated([spkEmbProj, MLXArray.zeros(like: spkEmbProj)], axis: 0)
    let condIn = concatenated([xCond, MLXArray.zeros(like: xCond)], axis: 0)
    let tIn = concatenated([t, t], axis: 0)

    // Mask: [B, 1, T] - all ones for bidirectional attention
    let singleMask = MLXArray.ones([1, 1, seqLen]).asType(.float32)
    let maskBatch = concatenated([singleMask, singleMask], axis: 0)

    // Debug: Compare first step intermediate values
    if step == 1 {
        print()
        print(String(repeating: "=", count: 80))
        print("STEP 1 DEBUG - Comparing with Python")
        print(String(repeating: "=", count: 80))
        print()

        // Enable decoder debug mode
        FlowMatchingDecoder.debugStep = 1

        // Time embedding comparison
        let swiftTEmb = s3gen.decoder.timeMLP(tIn)
        eval(swiftTEmb)
        print("Time Embedding:")
        print("  Swift  [0,:5]: \(swiftTEmb[0, 0..<5].asArray(Float.self))")
        print("  Python [0,:5]: \(pyTEmb[0, 0..<5].asArray(Float.self))")
        let tEmbDiff = MLX.abs(swiftTEmb - pyTEmb).max().item(Float.self)
        print("  Max diff: \(tEmbDiff) \(tEmbDiff < 0.001 ? "✅" : "❌")")
        print()

        // Input concatenation
        let spkExpanded = tiled(spkIn.expandedDimensions(axis: 2), repetitions: [1, 1, seqLen])
        let swiftH = concatenated([xIn, muIn, spkExpanded, condIn], axis: 1)
        eval(swiftH)
        print("Input Concatenation (h):")
        print("  Swift  [0,:5,0]: \(swiftH[0, 0..<5, 0].asArray(Float.self))")
        print("  Python [0,:5,0]: \(pyHConcat[0, 0..<5, 0].asArray(Float.self))")
        let hDiff = MLX.abs(swiftH - pyHConcat).max().item(Float.self)
        print("  Max diff: \(hDiff) \(hDiff < 0.001 ? "✅" : "❌")")
        print()

        // ResNet output
        let swiftResnet = s3gen.decoder.downBlocks[0].resnet(swiftH, mask: maskBatch, timeEmb: swiftTEmb)
        eval(swiftResnet)
        print("ResNet Output:")
        print("  Swift  [0,:5,0]: \(swiftResnet[0, 0..<5, 0].asArray(Float.self))")
        print("  Python [0,:5,0]: \(pyHResnet[0, 0..<5, 0].asArray(Float.self))")
        print("  Swift  mean: \(swiftResnet.mean().item(Float.self))")
        print("  Python mean: \(pyHResnet.mean().item(Float.self))")
        let resnetDiff = MLX.abs(swiftResnet - pyHResnet).max().item(Float.self)
        print("  Max diff: \(resnetDiff) \(resnetDiff < 0.01 ? "✅" : "❌")")
        print()

        // Transformer Block Tracing
        print("First Transformer Block:")

        // Load Python references
        let pyNorm1Out = try! NPYLoader.load(file: "\(liveFireDir)/debug_norm1_out.npy").asType(.float32)
        let pyAttnOut = try! NPYLoader.load(file: "\(liveFireDir)/debug_attn_out.npy").asType(.float32)
        let pyTfmrOut = try! NPYLoader.load(file: "\(liveFireDir)/debug_tfmr_out.npy").asType(.float32)

        // Swift: [B, C, T] -> [B, T, C]
        let resnetT = swiftResnet.transposed(0, 2, 1)
        let tfmr = s3gen.decoder.downBlocks[0].transformers[0]

        // Norm1
        let swiftNorm1 = tfmr.norm1(resnetT)
        eval(swiftNorm1)
        let norm1Diff = MLX.abs(swiftNorm1 - pyNorm1Out).max().item(Float.self)
        print("  Norm1:")
        print("    Swift  [0,0,:5]: \(swiftNorm1[0, 0, 0..<5].asArray(Float.self))")
        print("    Python [0,0,:5]: \(pyNorm1Out[0, 0, 0..<5].asArray(Float.self))")
        print("    Max diff: \(norm1Diff) \(norm1Diff < 0.001 ? "✅" : "❌")")

        // Attention - create full attention mask (all zeros = no masking)
        let attnMask = MLXArray.zeros([2, 1, seqLen, seqLen]).asType(.float32)
        let swiftAttn = tfmr.attention(swiftNorm1, mask: attnMask)
        eval(swiftAttn)
        let attnDiff = MLX.abs(swiftAttn - pyAttnOut).max().item(Float.self)
        print("  Attention:")
        print("    Swift  [0,0,:5]: \(swiftAttn[0, 0, 0..<5].asArray(Float.self))")
        print("    Python [0,0,:5]: \(pyAttnOut[0, 0, 0..<5].asArray(Float.self))")
        print("    Swift  mean: \(swiftAttn.mean().item(Float.self))")
        print("    Python mean: \(pyAttnOut.mean().item(Float.self))")
        print("    Max diff: \(attnDiff) \(attnDiff < 0.01 ? "✅" : "❌")")

        // Full transformer output (attention + FF)
        let swiftTfmrFull = tfmr(resnetT, mask: attnMask)
        eval(swiftTfmrFull)
        let tfmrDiff = MLX.abs(swiftTfmrFull - pyTfmrOut).max().item(Float.self)
        print("  Full Transformer:")
        print("    Swift  [0,0,:5]: \(swiftTfmrFull[0, 0, 0..<5].asArray(Float.self))")
        print("    Python [0,0,:5]: \(pyTfmrOut[0, 0, 0..<5].asArray(Float.self))")
        print("    Swift  mean: \(swiftTfmrFull.mean().item(Float.self))")
        print("    Python mean: \(pyTfmrOut.mean().item(Float.self))")
        print("    Max diff: \(tfmrDiff) \(tfmrDiff < 0.01 ? "✅" : "❌")")

        // Check FF weights from loaded decoder weights dictionary
        let ffKey = "downBlocks.0.transformers.0.ff.layers.0.weight"
        if let ffW0 = decoderWeights[ffKey] {
            eval(ffW0)
            print("  FF Layer 0 Weight (from loaded dict):")
            print("    Key: \(ffKey)")
            print("    Shape: \(ffW0.shape)")
            print("    Mean: \(ffW0.mean().item(Float.self))")
            print("    [0,:5]: \(ffW0[0, 0..<5].asArray(Float.self))")
            print("    Expected: [-0.077164, 0.080627, -0.095919, 0.035134, -0.016093]")
        } else {
            print("  FF Layer 0 Weight: KEY NOT FOUND! '\(ffKey)'")
            // Show what keys we DO have that match ff
            let ffKeys = decoderWeights.keys.filter { $0.contains("ff") && $0.contains("transformer") }.sorted().prefix(5)
            print("  Available FF keys: \(ffKeys)")
        }

        // Also check norm2 weights (called norm3 in Python export)
        let norm2Key = "downBlocks.0.transformers.0.norm2.weight"
        if let norm2W = decoderWeights[norm2Key] {
            eval(norm2W)
            print("  Norm2 Weight:")
            print("    Shape: \(norm2W.shape)")
            print("    Mean: \(norm2W.mean().item(Float.self))")
            print("    Expected mean: 0.4093")
        } else {
            print("  Norm2 Weight: KEY NOT FOUND! '\(norm2Key)'")
            let normKeys = decoderWeights.keys.filter { $0.contains("norm") && $0.contains("transformer_0") || $0.contains("transformers.0") }.sorted().prefix(10)
            print("  Available norm keys: \(Array(normKeys))")
        }
        print()

        FlowMatchingDecoder.debugStep = 0
    }

    // Forward pass
    let vBatch = s3gen.decoder(x: xIn, mu: muIn, t: tIn, speakerEmb: spkIn, cond: condIn, mask: maskBatch)
    eval(vBatch)

    // Split and apply CFG
    let vCond = vBatch[0].expandedDimensions(axis: 0)
    let vUncond = vBatch[1].expandedDimensions(axis: 0)
    let v = (1.0 + cfgRate) * vCond - cfgRate * vUncond
    eval(v)

    if step == 1 {
        print("Velocity (CFG):")
        print("  Swift  [0,:5,0]: \(v[0, 0..<5, 0].asArray(Float.self))")
        print("  Python [0,:5,0]: \(pyVelocity[0, 0..<5, 0].asArray(Float.self))")
        print("  Swift  mean: \(v.mean().item(Float.self))")
        print("  Python mean: \(pyVelocity.mean().item(Float.self))")
        let velDiff = MLX.abs(v - pyVelocity).max().item(Float.self)
        print("  Max diff: \(velDiff) \(velDiff < 0.1 ? "✅" : "❌")")
        print()
    }

    // Euler step
    xt = xt + v * dt
    eval(xt)

    if step == 1 {
        print("After Euler Step (x_next):")
        print("  Swift  [0,:5,0]: \(xt[0, 0..<5, 0].asArray(Float.self))")
        print("  Python [0,:5,0]: \(pyXNext[0, 0..<5, 0].asArray(Float.self))")
        print("  Swift  mean: \(xt.mean().item(Float.self))")
        print("  Python mean: \(pyXNext.mean().item(Float.self))")
        let xNextDiff = MLX.abs(xt - pyXNext).max().item(Float.self)
        print("  Max diff: \(xNextDiff) \(xNextDiff < 0.1 ? "✅" : "❌")")
        print()
        print(String(repeating: "=", count: 80))
        print()
    }

    if step == 1 || step == nTimesteps {
        print("Step \(step)/\(nTimesteps) (t=\(String(format: "%.4f", currentT))): xt mean=\(String(format: "%.4f", xt.mean().item(Float.self)))")
    }

    // Update time
    currentT = currentT + dt
    if step < nTimesteps {
        dt = tSpan[step + 1] - currentT
    }
}

// Final mel is xt in [B, D, T] format
let swiftMel = xt
eval(swiftMel)

print()
print("Swift mel spectrogram:")
print("  Shape: \(swiftMel.shape)")
print("  Mean: \(swiftMel.mean().item(Float.self))")
print("  Std: \(sqrt(((swiftMel - swiftMel.mean()) * (swiftMel - swiftMel.mean())).mean()).item(Float.self))")
print("  Range: [\(swiftMel.min().item(Float.self)), \(swiftMel.max().item(Float.self))]")
print()

// ============================================================================
// Compare with Python Reference
// ============================================================================
print(String(repeating: "=", count: 80))
print("Comparing with Python Reference")
print(String(repeating: "=", count: 80))
print()

let pythonMel = try! NPYLoader.load(file: "\(liveFireDir)/python_mel.npy").asType(.float32)

print("Python mel spectrogram:")
print("  Shape: \(pythonMel.shape)")
print("  Mean: \(pythonMel.mean().item(Float.self))")
print("  Std: \(sqrt(((pythonMel - pythonMel.mean()) * (pythonMel - pythonMel.mean())).mean()).item(Float.self))")
print("  Range: [\(pythonMel.min().item(Float.self)), \(pythonMel.max().item(Float.self))]")
print()

// Compute difference
let diff = MLX.abs(swiftMel - pythonMel)
eval(diff)

let maxDiff = diff.max().item(Float.self)
let meanDiff = diff.mean().item(Float.self)

print("Difference:")
print("  Max absolute: \(maxDiff)")
print("  Mean absolute: \(meanDiff)")
print()

// Sample values
print("Sample values [0, 0, 0:5]:")
print("  Swift:  \(swiftMel[0, 0, 0..<5].asArray(Float.self))")
print("  Python: \(pythonMel[0, 0, 0..<5].asArray(Float.self))")
print()

print("Sample values [0, 40, 0:5]:")
print("  Swift:  \(swiftMel[0, 40, 0..<5].asArray(Float.self))")
print("  Python: \(pythonMel[0, 40, 0..<5].asArray(Float.self))")
print()

// ============================================================================
// Verdict
// ============================================================================
print(String(repeating: "=", count: 80))
print("VERDICT")
print(String(repeating: "=", count: 80))
print()

let tolerance: Float = 0.01
if maxDiff < tolerance {
    print("PASS: Max difference \(maxDiff) < tolerance \(tolerance)")
    print()
    print("Swift mel output MATCHES Python canonical output!")
} else {
    print("FAIL: Max difference \(maxDiff) >= tolerance \(tolerance)")
    print()
    print("Swift and Python outputs DIVERGE")
    print("Need to investigate the decoder implementation")
}
print()
print(String(repeating: "=", count: 80))

// ==================================================================
// NPY Loader
// ==================================================================
struct NPYLoader {
    static func load(file: String) throws -> MLXArray {
        let url = URL(fileURLWithPath: file)
        let data = try Data(contentsOf: url)

        var offset = 0

        let magic = data[0..<6]
        guard magic.elementsEqual([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]) else {
            throw NPYError.invalidFormat
        }
        offset += 6

        let major = data[offset]
        offset += 2

        var headerLen: Int
        if major == 1 {
            headerLen = Int(data[offset]) | (Int(data[offset + 1]) << 8)
            offset += 2
        } else {
            headerLen = Int(data[offset]) | (Int(data[offset + 1]) << 8) |
                       (Int(data[offset + 2]) << 16) | (Int(data[offset + 3]) << 24)
            offset += 4
        }

        let headerData = data[offset..<(offset + headerLen)]
        guard let headerStr = String(data: headerData, encoding: .ascii) else {
            throw NPYError.invalidHeader
        }
        offset += headerLen

        guard let descrRange = headerStr.range(of: "'descr':\\s*'([^']+)'", options: .regularExpression),
              let shapeRange = headerStr.range(of: "'shape':\\s*\\(([^)]+)\\)", options: .regularExpression) else {
            throw NPYError.invalidHeader
        }

        let descr = String(headerStr[descrRange]).components(separatedBy: "'")[3]
        let shapeStr = String(headerStr[shapeRange]).components(separatedBy: "(")[1].components(separatedBy: ")")[0]
        let shape = shapeStr.components(separatedBy: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }

        let arrayData = data[offset...]

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
