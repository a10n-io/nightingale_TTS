import Foundation
import MLX
import MLXNN
import Nightingale

// ==================================================================
// STEP 5: S3GEN SPEECH CODE EMBEDDING VERIFICATION (REAL DATA)
// ==================================================================

print(String(repeating: "=", count: 80))
print("STEP 5: S3GEN SPEECH CODE EMBEDDING VERIFICATION (REAL DATA)")
print(String(repeating: "=", count: 80))
print()

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
let outputPath = "\(projectRoot)/verification_outputs/step5"

// Load S3Gen model
print("Loading S3Gen model...")
let modelsURL = URL(fileURLWithPath: modelPath)
let s3genFP16URL = modelsURL.appendingPathComponent("s3gen_fp16.safetensors")
let vocoderFP16URL = modelsURL.appendingPathComponent("vocoder_weights.safetensors")

let s3genWeights = try! MLX.loadArrays(url: s3genFP16URL)
let vocoderWeights = try! MLX.loadArrays(url: vocoderFP16URL)
let s3gen = S3Gen(flowWeights: s3genWeights, vocoderWeights: vocoderWeights)
print("Model loaded!")
print()

// Load voice data
print("Loading voice data...")
let soulS3 = try! NPYLoader.load(file: "\(voicePath)/soul_s3_192.npy")
let promptTokens = try! NPYLoader.load(file: "\(voicePath)/prompt_token.npy")
let promptFeat = try! NPYLoader.load(file: "\(voicePath)/prompt_feat.npy")

print("soul_s3: \(soulS3.shape)")
print("prompt_tokens: \(promptTokens.shape)")
print("prompt_feat: \(promptFeat.shape)")
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

// =================================================================
// Step 5A: Concatenate Prompt + Speech Tokens
// =================================================================
print(String(repeating: "=", count: 80))
print("Step 5A: Concatenate Prompt + Speech Tokens")
print(String(repeating: "=", count: 80))

let fullTokens = concatenated([promptTokens, testSpeechTokens], axis: 1)

print("Prompt tokens: \(promptTokens.shape)")
print("Speech tokens: \(testSpeechTokens.shape)")
print("Full tokens: \(fullTokens.shape)")

let promptTokenLen = promptTokens.shape[1]
let speechTokenLen = testSpeechTokens.shape[1]
let fullTokenLen = promptTokenLen + speechTokenLen

print("Prompt len: \(promptTokenLen)")
print("Speech len: \(speechTokenLen)")
print("Full len: \(fullTokenLen)")
print()

// =================================================================
// Step 5B: Create Mask
// =================================================================
print(String(repeating: "=", count: 80))
print("Step 5B: Create Mask")
print(String(repeating: "=", count: 80))

let batchSize = fullTokens.shape[0]
let maxLen = fullTokenLen

// Create sequence mask
let seqRange = MLXArray(0..<maxLen).reshaped([1, maxLen])
let mask = less(seqRange, MLXArray(fullTokenLen))
let maskFloat = mask.expandedDimensions(axis: -1).asType(.float32)

print("Mask shape: \(maskFloat.shape)")
print("Mask sum: \(maskFloat.sum().item(Float.self))")
print()

// =================================================================
// Step 5C: Apply Input Embedding
// =================================================================
print(String(repeating: "=", count: 80))
print("Step 5C: Apply Input Embedding")
print(String(repeating: "=", count: 80))

let inputEmb = s3gen.inputEmbedding
let vocabSize = inputEmb.weight.shape[0]
let embedDim = inputEmb.weight.shape[1]
print("Input embedding: vocab_size=\(vocabSize), embed_dim=\(embedDim)")
print()

// Debug: Check embedding weight values for first token
print("DEBUG: Checking embedding weights...")
print("Embedding weight[1568, :5]: \(inputEmb.weight[1568, 0..<5].asArray(Float.self))")
print("Embedding weight[3708, :5]: \(inputEmb.weight[3708, 0..<5].asArray(Float.self))")
print()

// Clip tokens to valid range
let numEmbeddings = inputEmb.weight.shape[0]
let tokensClipped = clip(fullTokens, min: MLXArray(0), max: MLXArray(numEmbeddings - 1))

// Apply embedding
var tokenEmb = inputEmb(tokensClipped.asType(DType.int32))
eval(tokenEmb)

print("Token embedding shape: \(tokenEmb.shape)")
print("Token embedding[0,0,:5]: \(tokenEmb[0, 0, 0..<5].asArray(Float.self))")
print("Token embedding[0,-1,:5]: \(tokenEmb[0, -1, 0..<5].asArray(Float.self))")
print()

// Apply mask
tokenEmb = tokenEmb * maskFloat
eval(tokenEmb)

print("After mask - shape: \(tokenEmb.shape)")
print("After mask[0,0,:5]: \(tokenEmb[0, 0, 0..<5].asArray(Float.self))")
print("After mask[0,-1,:5]: \(tokenEmb[0, -1, 0..<5].asArray(Float.self))")
print()

// =================================================================
// Step 5D: Speaker Embedding (Normalize + Affine)
// =================================================================
print(String(repeating: "=", count: 80))
print("Step 5D: Speaker Embedding (Normalize + Affine)")
print(String(repeating: "=", count: 80))

var soulS3Batch = soulS3
if soulS3.ndim == 1 {
    soulS3Batch = soulS3.expandedDimensions(axis: 0)
}

print("S3 soul shape: \(soulS3Batch.shape)")
print("S3 soul[:5]: \(soulS3Batch[0, 0..<5].asArray(Float.self))")
print()

// Normalize (same as flow.py)
let norm = sqrt(sum(soulS3Batch * soulS3Batch, axis: 1, keepDims: true))
let embeddingNorm = soulS3Batch / (norm + MLXArray(1e-8))
print("S3 norm: \(norm.item(Float.self))")
print("Normalized[:5]: \(embeddingNorm[0, 0..<5].asArray(Float.self))")
print()

// Apply affine projection
let spkEmb = s3gen.spkEmbedAffine(embeddingNorm)
eval(spkEmb)

print("Speaker embedding shape: \(spkEmb.shape)")
print("Speaker embedding[:5]: \(spkEmb[0, 0..<5].asArray(Float.self))")
print()

// =================================================================
// Load Python Reference
// =================================================================
print(String(repeating: "=", count: 80))
print("LOADING PYTHON REFERENCE")
print(String(repeating: "=", count: 80))

let refFullTokens = try! NPYLoader.load(file: "\(outputPath)/full_tokens.npy")
let refMask = try! NPYLoader.load(file: "\(outputPath)/mask.npy")
let refTokenEmb = try! NPYLoader.load(file: "\(outputPath)/token_emb.npy")
let refSpkEmb = try! NPYLoader.load(file: "\(outputPath)/spk_emb.npy")

print("Loaded reference outputs:")
print("  full_tokens: \(refFullTokens.shape)")
print("  mask: \(refMask.shape)")
print("  token_emb: \(refTokenEmb.shape)")
print("  spk_emb: \(refSpkEmb.shape)")
print()

// =================================================================
// VERIFICATION
// =================================================================
print(String(repeating: "=", count: 80))
print("VERIFICATION")
print(String(repeating: "=", count: 80))

func maxDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    let diff = abs(a - b)
    return diff.max().item(Float.self)
}

// Compare full_tokens
let fullTokensDiff = maxDiff(fullTokens.asType(.float32), refFullTokens.asType(.float32))
print("full_tokens max_diff: \(String(format: "%.2e", fullTokensDiff))")

// Compare mask
let maskDiff = maxDiff(maskFloat, refMask.asType(.float32))
print("mask max_diff: \(String(format: "%.2e", maskDiff))")

// Compare token_emb
let tokenEmbDiff = maxDiff(tokenEmb, refTokenEmb)
print("token_emb max_diff: \(String(format: "%.2e", tokenEmbDiff))")

// Compare spk_emb
let spkEmbDiff = maxDiff(spkEmb, refSpkEmb)
print("spk_emb max_diff: \(String(format: "%.2e", spkEmbDiff))")
print()

// =================================================================
// PASS/FAIL
// =================================================================
let threshold: Float = 0.001

let fullTokensPass = fullTokensDiff < threshold
let maskPass = maskDiff < threshold
let tokenEmbPass = tokenEmbDiff < threshold
let spkEmbPass = spkEmbDiff < threshold

print(String(repeating: "=", count: 80))
print("RESULTS")
print(String(repeating: "=", count: 80))
print("Full tokens: \(fullTokensPass ? "✅ PASSED" : "❌ FAILED") (max_diff: \(String(format: "%.2e", fullTokensDiff)))")
print("Mask: \(maskPass ? "✅ PASSED" : "❌ FAILED") (max_diff: \(String(format: "%.2e", maskDiff)))")
print("Token embedding: \(tokenEmbPass ? "✅ PASSED" : "❌ FAILED") (max_diff: \(String(format: "%.2e", tokenEmbDiff)))")
print("Speaker embedding: \(spkEmbPass ? "✅ PASSED" : "❌ FAILED") (max_diff: \(String(format: "%.2e", spkEmbDiff)))")
print(String(repeating: "=", count: 80))

let allPass = fullTokensPass && maskPass && tokenEmbPass && spkEmbPass
if allPass {
    print("✅ ALL TESTS PASSED")
} else {
    print("❌ SOME TESTS FAILED")
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
