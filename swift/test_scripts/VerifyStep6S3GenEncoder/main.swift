import Foundation
import MLX
import MLXNN
import Nightingale

print(String(repeating: "=", count: 80))
print("VERIFY STEP 6: S3Gen Encoder (REAL DATA)")
print(String(repeating: "=", count: 80))
print()

// Set random seed for deterministic results
MLXRandom.seed(42)

// Paths
let projectRoot = URL(fileURLWithPath: #file)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .path

let voicePath = "\(projectRoot)/baked_voices/samantha_full"
let modelPath = "\(projectRoot)/models"
let step4Path = "\(projectRoot)/verification_outputs/step4"
let referencePath = "\(projectRoot)/verification_outputs/step6"

// Load S3Gen model using HuggingFace weights (to match Python gold standard)
print("Loading S3Gen model from HuggingFace weights...")
let modelsURL = URL(fileURLWithPath: modelPath)
let hfWeightsURL = modelsURL.appendingPathComponent("chatterbox_hf.safetensors")
let vocoderFP16URL = modelsURL.appendingPathComponent("vocoder_weights.safetensors")

print("Loading HuggingFace model weights: \(hfWeightsURL.path)")
let allWeights = try! MLX.loadArrays(url: hfWeightsURL)
print("Loaded \(allWeights.count) total weights from HuggingFace model")

// For vocoder, use separate file (HF model doesn't include vocoder)
let vocoderWeights = try! MLX.loadArrays(url: vocoderFP16URL)

// Pass all weights - S3Gen will filter the ones it needs
let s3gen = S3Gen(flowWeights: allWeights, vocoderWeights: vocoderWeights)
print("Model loaded with HuggingFace weights!")
print()

// Load voice data
print("Loading voice data...")
let soulS3 = try! NPYLoader.load(file: "\(voicePath)/soul_s3_192.npy")
let promptTokens = try! NPYLoader.load(file: "\(voicePath)/prompt_token.npy")

print("soul_s3: \(soulS3.shape)")
print("prompt_tokens: \(promptTokens.shape)")
print()

// Load REAL speech tokens from Step 4
print("Loading real speech tokens from Step 4...")
let speechTokensPath = "\(step4Path)/generated_speech_tokens.npy"
guard FileManager.default.fileExists(atPath: speechTokensPath) else {
    print("ERROR: \(speechTokensPath) not found!")
    print("Run Step 4 first to generate speech tokens.")
    exit(1)
}

let speechTokens1D = try! NPYLoader.load(file: speechTokensPath)
let testSpeechTokens = speechTokens1D.reshaped([1, speechTokens1D.shape[0]])
print("Loaded speech tokens: \(speechTokens1D.shape)")
print("First 10: \(Array(speechTokens1D[0..<min(10, speechTokens1D.shape[0])].asArray(Int32.self)))")
print("Reshaped to: \(testSpeechTokens.shape)")
print()

// ========================================================================
// Step 6A: Prepare Inputs (Token + Speaker Embeddings)
// ========================================================================
print(String(repeating: "=", count: 80))
print("Step 6A: Token + Speaker Embeddings")
print(String(repeating: "=", count: 80))

// Concatenate tokens
let fullTokens = concatenated([promptTokens, testSpeechTokens], axis: 1)
let fullTokenLen = fullTokens.shape[1]
print("Full tokens: \(fullTokens.shape)")

// Create mask
let seqRange = MLXArray(0..<fullTokenLen).reshaped([1, fullTokenLen])
let mask = less(seqRange, MLXArray(fullTokenLen))
let maskFloat = mask.expandedDimensions(axis: -1).asType(.float32)

// Apply input embedding
let inputEmb = s3gen.inputEmbedding
let numEmbeddings = inputEmb.weight.shape[0]
let tokensClipped = clip(fullTokens, min: MLXArray(0), max: MLXArray(numEmbeddings - 1))
var tokenEmb = inputEmb(tokensClipped.asType(DType.int32))
tokenEmb = tokenEmb * maskFloat
eval(tokenEmb)

print("Token embedding: \(tokenEmb.shape)")
print("Token embedding[0,0,:5]: \(tokenEmb[0, 0, 0..<5].asArray(Float.self))")
print()

// ========================================================================
// Step 6B: Run Encoder
// ========================================================================
print(String(repeating: "=", count: 80))
print("Step 6B: Run Encoder")
print(String(repeating: "=", count: 80))

// Pass sequence length to match Python encoder call: encoder(token_emb, xs_lens)
let seqLen = MLXArray([fullTokenLen])
print("Passing seqLen: \(seqLen.asArray(Int32.self))")
let encoderOutput = s3gen.encoder(tokenEmb, seqLen: seqLen)
eval(encoderOutput)

let mean = encoderOutput.mean().item(Float.self)
let variance = ((encoderOutput - mean) * (encoderOutput - mean)).mean().item(Float.self)
let std = sqrt(variance)

print("Encoder output: \(encoderOutput.shape)")
print("  mean: \(mean)")
print("  std: \(std)")
print("  min: \(encoderOutput.min().item(Float.self))")
print("  max: \(encoderOutput.max().item(Float.self))")
print("Encoder output[0,0,:5]: \(encoderOutput[0, 0, 0..<5].asArray(Float.self))")
print("Encoder output[0,-1,:5]: \(encoderOutput[0, -1, 0..<5].asArray(Float.self))")
print()

// ========================================================================
// Step 6C: Load Reference and Compare
// ========================================================================
print(String(repeating: "=", count: 80))
print("Step 6C: Load Reference and Compare")
print(String(repeating: "=", count: 80))

print("Loading Python reference encoder output...")
let refEncoderOutput = try! NPYLoader.load(file: "\(referencePath)/encoder_output.npy")
print("Python reference encoder output: \(refEncoderOutput.shape)")
print()

// Compare shapes
print("Shape comparison:")
print("  Swift:     \(encoderOutput.shape)")
print("  Reference: \(refEncoderOutput.shape)")

if encoderOutput.shape != refEncoderOutput.shape {
    print("❌ SHAPE MISMATCH!")
    exit(1)
}
print("✅ Shapes match")
print()

// Compute difference
let diff = encoderOutput - refEncoderOutput
eval(diff)

let absDiff = abs(diff)
eval(absDiff)

let maxAbsDiff = absDiff.max().item(Float.self)
let meanAbsDiff = absDiff.mean().item(Float.self)
let diffStd = ((diff - diff.mean()) * (diff - diff.mean())).mean().item(Float.self)

print("Difference statistics:")
print("  Max absolute difference: \(maxAbsDiff)")
print("  Mean absolute difference: \(meanAbsDiff)")
print("  Difference std: \(sqrt(diffStd))")
print()

// Value comparison (first/last elements)
print("Value comparison:")
print("  Swift[0,0,:5]:     \(encoderOutput[0, 0, 0..<5].asArray(Float.self))")
print("  Reference[0,0,:5]: \(refEncoderOutput[0, 0, 0..<5].asArray(Float.self))")
print()
print("  Swift[0,-1,:5]:     \(encoderOutput[0, -1, 0..<5].asArray(Float.self))")
print("  Reference[0,-1,:5]: \(refEncoderOutput[0, -1, 0..<5].asArray(Float.self))")
print()

// Tolerance check (relaxed for neural network inference with FP16 weights)
let tolerance: Float = 0.1  // 10% of typical value range
if maxAbsDiff < tolerance {
    print("✅ PASS: Max difference (\(maxAbsDiff)) < tolerance (\(tolerance))")
} else {
    print("⚠️  WARNING: Max difference (\(maxAbsDiff)) >= tolerance (\(tolerance))")
    print("   This may indicate a significant discrepancy")
}
print()

// Check if mean difference is acceptable (more stable than max)
let meanDiffThreshold: Float = 0.05  // 5% of typical std
if meanAbsDiff < meanDiffThreshold {
    print("✅ PASS: Mean difference (\(meanAbsDiff)) < threshold (\(meanDiffThreshold))")
} else {
    print("⚠️  WARNING: Mean difference (\(meanAbsDiff)) >= threshold (\(meanDiffThreshold))")
}
print()

// ========================================================================
// Summary
// ========================================================================
print(String(repeating: "=", count: 80))
print("SUMMARY")
print(String(repeating: "=", count: 80))

var allPassed = true
if encoderOutput.shape != refEncoderOutput.shape {
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
    print("✅ Step 6 (S3Gen Encoder): VERIFICATION PASSED")
    print()
    print("All checks passed! The encoder implementation matches the reference.")
} else {
    print("❌ Step 6 (S3Gen Encoder): VERIFICATION FAILED")
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
        let minor = data[offset + 1]
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
