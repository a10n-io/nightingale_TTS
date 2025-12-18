import Foundation
import MLX
import MLXNN
import Nightingale

// MARK: - NPY Loader

struct NPYLoader {
    static func load(contentsOf url: URL) throws -> MLXArray {
        let data = try Data(contentsOf: url)
        var offset = 0

        guard data.count >= 6 else { fatalError("Invalid NPY") }
        offset = 8

        let major = data[6]
        var headerLen: Int
        if major == 1 {
            headerLen = Int(data[offset]) | (Int(data[offset + 1]) << 8)
            offset += 2
        } else {
            headerLen = Int(data[offset]) | (Int(data[offset + 1]) << 8) |
                       (Int(data[offset + 2]) << 16) | (Int(data[offset + 3]) << 24)
            offset += 4
        }

        guard let headerStr = String(data: data[offset..<(offset + headerLen)], encoding: .utf8) else {
            fatalError("Invalid header")
        }
        offset += headerLen

        guard let shapeMatch = headerStr.range(of: "'shape':\\s*\\(([^)]*)\\)", options: .regularExpression) else {
            fatalError("No shape")
        }
        let shapeStr = String(headerStr[shapeMatch]).replacingOccurrences(of: "'shape': (", with: "").replacingOccurrences(of: ")", with: "")
        let shape = shapeStr.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }

        guard let dtypeMatch = headerStr.range(of: "'descr':\\s*'([^']*)'", options: .regularExpression) else {
            fatalError("No dtype")
        }
        let dtypeStr = String(headerStr[dtypeMatch]).replacingOccurrences(of: "'descr': '", with: "").replacingOccurrences(of: "'", with: "")

        let arrayData = data[offset...]
        let array: MLXArray

        if dtypeStr.hasSuffix("f4") {
            let count = arrayData.count / MemoryLayout<Float>.size
            let floats = arrayData.withUnsafeBytes { $0.bindMemory(to: Float.self) }
            array = MLXArray(Array(floats.prefix(count)))
        } else if dtypeStr.hasSuffix("i4") {
            let count = arrayData.count / MemoryLayout<Int32>.size
            let ints = arrayData.withUnsafeBytes { $0.bindMemory(to: Int32.self) }
            array = MLXArray(Array(ints.prefix(count)))
        } else {
            fatalError("Unsupported dtype: \(dtypeStr)")
        }

        return shape.isEmpty ? array : array.reshaped(shape)
    }
}

// MARK: - Key Remapping

func remapT3Keys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var remapped: [String: MLXArray] = [:]
    for (key, value) in weights {
        if let newKey = remapT3Key(key) {
            remapped[newKey] = value
        }
    }
    return remapped
}

func remapT3Key(_ key: String) -> String? {
    var k = key
    if k.hasPrefix("t3.") { k = String(k.dropFirst("t3.".count)) }
    if k.hasPrefix("s3gen.") || k.hasPrefix("ve.") { return nil }

    if k.hasPrefix("tfmr.layers.") {
        k = String(k.dropFirst("tfmr.".count))
    } else if k.hasPrefix("tfmr.model.") {
        k = String(k.dropFirst("tfmr.model.".count))
        if k.hasPrefix("embed_tokens") { return nil }
    } else if k == "tfmr.norm.weight" {
        k = "norm.weight"
    } else if k.hasPrefix("tfmr.") {
        return nil
    }

    if k.hasPrefix("cond_enc.") {
        if k.hasPrefix("cond_enc.spkr_enc.") {
            return k.replacingOccurrences(of: "cond_enc.spkr_enc", with: "speakerProj")
        }
        if k.hasPrefix("cond_enc.perceiver.") {
            return k.replacingOccurrences(of: "cond_enc.perceiver", with: "perceiver")
        }
        if k.hasPrefix("cond_enc.emotion_adv_fc.") {
            return k.replacingOccurrences(of: "cond_enc.emotion_adv_fc", with: "emotionAdvFC")
        }
        return nil
    }

    k = k.replacingOccurrences(of: "self_attn", with: "selfAttn")
    k = k.replacingOccurrences(of: "q_proj", with: "qProj")
    k = k.replacingOccurrences(of: "k_proj", with: "kProj")
    k = k.replacingOccurrences(of: "v_proj", with: "vProj")
    k = k.replacingOccurrences(of: "o_proj", with: "oProj")
    k = k.replacingOccurrences(of: "input_layernorm", with: "inputLayernorm")
    k = k.replacingOccurrences(of: "post_attention_layernorm", with: "postAttentionLayernorm")
    k = k.replacingOccurrences(of: "gate_proj", with: "gateProj")
    k = k.replacingOccurrences(of: "up_proj", with: "upProj")
    k = k.replacingOccurrences(of: "down_proj", with: "downProj")
    k = k.replacingOccurrences(of: "text_emb", with: "textEmb")
    k = k.replacingOccurrences(of: "speech_emb", with: "speechEmb")
    k = k.replacingOccurrences(of: "text_head", with: "textHead")
    k = k.replacingOccurrences(of: "speech_head", with: "speechHead")
    k = k.replacingOccurrences(of: "text_pos_emb.emb", with: "textPosEmb.embedding")
    k = k.replacingOccurrences(of: "speech_pos_emb.emb", with: "speechPosEmb.embedding")

    return k
}

func maxDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    return abs(a - b).max().item(Float.self)
}

// MARK: - Main Test

print(String(repeating: "=", count: 80))
print("SPEECH EMBEDDING INPUT VERIFICATION - Swift")
print(String(repeating: "=", count: 80))

let voicePath = "/Users/a10n/Projects/chatterbox claude/ChatterboxApp/AppAssets/voices/samantha"
let voiceURL = URL(fileURLWithPath: voicePath)
let refPath = "/Users/a10n/Projects/nightingale/verification_outputs/speech_embedding_debug"
let refURL = URL(fileURLWithPath: refPath)

// Load Python reference
let t3_cond_tokens_ref = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("t3_cond_tokens.npy"))
let cond_positions_ref = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("cond_positions.npy"))
let speech_emb_weight_ref = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("speech_emb_weight.npy"))
let speech_pos_emb_weight_ref = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("speech_pos_emb_weight.npy"))
let speech_emb_output_ref = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("speech_emb_output.npy"))
let speech_pos_emb_output_ref = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("speech_pos_emb_output.npy"))
let combined_output_ref = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("combined_output.npy"))

// Load Swift data
let t3_cond_tokens_swift = try NPYLoader.load(contentsOf: voiceURL.appendingPathComponent("t3_cond_tokens.npy"))

// Load T3 model
print("\n📦 Loading T3 model...")
let modelPath = "/Users/a10n/Projects/chatterbox claude/ChatterboxApp/AppAssets/models/chatterbox"
let modelURL = URL(fileURLWithPath: modelPath)
let t3URL = modelURL.appendingPathComponent("t3_fp32.safetensors")
let configURL = modelURL.appendingPathComponent("t3_config.json")
let ropeFreqsURL = modelURL.appendingPathComponent("rope_freqs_llama3.safetensors")

let configData = try Data(contentsOf: configURL)
let config = try JSONDecoder().decode(T3Config.self, from: configData)

let rawWeights = try MLX.loadArrays(url: t3URL)
let weights = remapT3Keys(rawWeights)
let t3 = T3Model(config: config, weights: weights, ropeFreqsURL: ropeFreqsURL)
print("✅ T3 model loaded\n")

// TEST 1: Compare t3_cond_tokens (speech codes)
print(String(repeating: "=", count: 80))
print("TEST 1: SPEECH CODES (t3_cond_tokens)")
print(String(repeating: "=", count: 80))

let tokensSwift = t3_cond_tokens_swift.asArray(Int32.self)
let tokensPython = t3_cond_tokens_ref.asArray(Int32.self)

print("\nPython reference:")
print("  Shape: \(t3_cond_tokens_ref.shape)")
print("  First 20 tokens: \(Array(tokensPython.prefix(20)))")

print("\nSwift values:")
print("  Shape: \(t3_cond_tokens_swift.shape)")
print("  First 20 tokens: \(Array(tokensSwift.prefix(20)))")

let tokens_match = tokensSwift == tokensPython
print("\n📊 RESULTS:")
print("  Tokens match: \(tokens_match ? "✅ IDENTICAL" : "❌ DIFFERENT")")
if !tokens_match {
    print("  ⚠️ CRITICAL: Speech code tokens differ between Python and Swift!")
    print("  This will cause completely different embeddings to be looked up.")
}

// TEST 2: Compare position indices
print("\n" + String(repeating: "=", count: 80))
print("TEST 2: POSITION INDICES (condPositions)")
print(String(repeating: "=", count: 80))

let condLen = t3_cond_tokens_swift.shape[1]
let condPositions = MLXArray(0..<condLen).asType(.int32).expandedDimensions(axis: 0)

print("\nPython reference:")
print("  Shape: \(cond_positions_ref.shape)")
print("  First 20: \(Array(cond_positions_ref.asArray(Int32.self).prefix(20)))")

print("\nSwift values:")
print("  Shape: \(condPositions.shape)")
print("  First 20: \(Array(condPositions.flattened().asArray(Int32.self).prefix(20)))")

let positions_diff = maxDiff(condPositions, cond_positions_ref)
print("\n📊 RESULTS:")
print("  Positions max_diff: \(String(format: "%.8f", positions_diff)) \(positions_diff == 0 ? "✅" : "❌")")

// TEST 3: Compare speech embedding weights
print("\n" + String(repeating: "=", count: 80))
print("TEST 3: SPEECH EMBEDDING WEIGHT MATRIX")
print(String(repeating: "=", count: 80))

let speech_emb_weight_swift = t3.speechEmb.weight

print("\nPython reference:")
print("  Shape: \(speech_emb_weight_ref.shape)")
let ref_mean = speech_emb_weight_ref.mean().item(Float.self)
let ref_variance = ((speech_emb_weight_ref - ref_mean) * (speech_emb_weight_ref - ref_mean)).mean().item(Float.self)
let ref_std = sqrt(ref_variance)
print("  Mean: \(String(format: "%.8f", ref_mean))")
print("  Std: \(String(format: "%.8f", ref_std))")
print("  First value: \(String(format: "%.8f", speech_emb_weight_ref[0, 0].item(Float.self)))")

print("\nSwift values:")
print("  Shape: \(speech_emb_weight_swift.shape)")
print("  Mean: \(String(format: "%.8f", speech_emb_weight_swift.mean().item(Float.self)))")
let variance_swift = ((speech_emb_weight_swift - speech_emb_weight_swift.mean()) * (speech_emb_weight_swift - speech_emb_weight_swift.mean())).mean().item(Float.self)
let std_swift = sqrt(variance_swift)
print("  Std: \(String(format: "%.8f", std_swift))")
print("  First value: \(String(format: "%.8f", speech_emb_weight_swift[0, 0].item(Float.self)))")

let weight_diff = maxDiff(speech_emb_weight_swift, speech_emb_weight_ref)
print("\n📊 RESULTS:")
print("  Weight matrix max_diff: \(String(format: "%.8f", weight_diff)) \(weight_diff < 1e-4 ? "✅" : "❌")")

// TEST 4: Compare speech position embedding weights
print("\n" + String(repeating: "=", count: 80))
print("TEST 4: SPEECH POSITION EMBEDDING WEIGHT")
print(String(repeating: "=", count: 80))

let speech_pos_emb_weight_swift = t3.speechPosEmb.embedding.weight

print("\nPython reference:")
print("  Shape: \(speech_pos_emb_weight_ref.shape)")
print("  Mean: \(String(format: "%.8f", speech_pos_emb_weight_ref.mean().item(Float.self)))")
print("  First value: \(String(format: "%.8f", speech_pos_emb_weight_ref[0, 0].item(Float.self)))")

print("\nSwift values:")
print("  Shape: \(speech_pos_emb_weight_swift.shape)")
print("  Mean: \(String(format: "%.8f", speech_pos_emb_weight_swift.mean().item(Float.self)))")
print("  First value: \(String(format: "%.8f", speech_pos_emb_weight_swift[0, 0].item(Float.self)))")

let pos_weight_diff = maxDiff(speech_pos_emb_weight_swift, speech_pos_emb_weight_ref)
print("\n📊 RESULTS:")
print("  Position weight max_diff: \(String(format: "%.8f", pos_weight_diff)) \(pos_weight_diff < 1e-4 ? "✅" : "❌")")

// TEST 5: Compare embedding outputs
print("\n" + String(repeating: "=", count: 80))
print("TEST 5: EMBEDDING OUTPUTS")
print(String(repeating: "=", count: 80))

let speech_emb_swift = t3.speechEmb(t3_cond_tokens_swift)
let speech_pos_emb_swift = t3.speechPosEmb(condPositions)

print("\nPython speechEmb:")
let ref_emb_mean = speech_emb_output_ref.mean().item(Float.self)
let ref_emb_variance = ((speech_emb_output_ref - ref_emb_mean) * (speech_emb_output_ref - ref_emb_mean)).mean().item(Float.self)
let ref_emb_std = sqrt(ref_emb_variance)
print("  Mean: \(String(format: "%.8f", ref_emb_mean))")
print("  Std: \(String(format: "%.8f", ref_emb_std))")
print("  First token, first 5 dims: \(speech_emb_output_ref[0, 0, 0..<5].asArray(Float.self))")

print("\nSwift speechEmb:")
print("  Mean: \(String(format: "%.8f", speech_emb_swift.mean().item(Float.self)))")
let variance_emb = ((speech_emb_swift - speech_emb_swift.mean()) * (speech_emb_swift - speech_emb_swift.mean())).mean().item(Float.self)
let std_emb = sqrt(variance_emb)
print("  Std: \(String(format: "%.8f", std_emb))")
print("  First token, first 5 dims: \(speech_emb_swift[0, 0, 0..<5].asArray(Float.self))")

let emb_diff = maxDiff(speech_emb_swift, speech_emb_output_ref)
print("\n📊 speechEmb max_diff: \(String(format: "%.8f", emb_diff)) \(emb_diff < 1e-4 ? "✅" : "❌")")

// Position embedding - need to match shapes
let speech_pos_emb_expanded = speech_pos_emb_swift.expandedDimensions(axis: 0)  // [1, T, C]
print("\nPython speechPosEmb:")
print("  Shape: \(speech_pos_emb_output_ref.shape)")
print("  Mean: \(String(format: "%.8f", speech_pos_emb_output_ref.mean().item(Float.self)))")

print("\nSwift speechPosEmb:")
print("  Shape: \(speech_pos_emb_expanded.shape)")
print("  Mean: \(String(format: "%.8f", speech_pos_emb_expanded.mean().item(Float.self)))")

// Reshape reference to match
let speech_pos_emb_ref_reshaped = speech_pos_emb_output_ref.reshaped([1, speech_pos_emb_output_ref.shape[0], speech_pos_emb_output_ref.shape[1]])
let pos_emb_diff = maxDiff(speech_pos_emb_expanded, speech_pos_emb_ref_reshaped)
print("\n📊 speechPosEmb max_diff: \(String(format: "%.8f", pos_emb_diff)) \(pos_emb_diff < 1e-4 ? "✅" : "❌")")

// Combined
let combined_swift = speech_emb_swift + speech_pos_emb_expanded
print("\nPython combined (speechEmb + speechPosEmb):")
print("  Mean: \(String(format: "%.8f", combined_output_ref.mean().item(Float.self)))")

print("\nSwift combined:")
print("  Mean: \(String(format: "%.8f", combined_swift.mean().item(Float.self)))")

let combined_diff = maxDiff(combined_swift, combined_output_ref)
print("\n📊 Combined max_diff: \(String(format: "%.8f", combined_diff)) \(combined_diff < 0.01 ? "✅" : "❌")")

// SUMMARY
print("\n" + String(repeating: "=", count: 80))
print("SUMMARY")
print(String(repeating: "=", count: 80))

let tests = [
    ("Speech code tokens", tokens_match ? 0.0 : 1.0, 0.0),
    ("Position indices", Float(positions_diff), 0.0),
    ("Speech embedding weight", weight_diff, 1e-4),
    ("Position embedding weight", pos_weight_diff, 1e-4),
    ("Speech embedding output", emb_diff, 1e-4),
    ("Position embedding output", pos_emb_diff, 1e-4),
    ("Combined output", combined_diff, 0.01)
]

for (name, diff, threshold) in tests {
    let status = diff <= Float(threshold) ? "✅" : "❌"
    print("\(status) \(name): \(String(format: "%.8f", diff))")
}

print("\n" + String(repeating: "=", count: 80))
