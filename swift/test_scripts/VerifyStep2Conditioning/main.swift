import Foundation
import MLX
import Nightingale

// MARK: - NPY Loader

struct NPYLoader {
    static func load(contentsOf url: URL) throws -> MLXArray {
        let data = try Data(contentsOf: url)
        var offset = 0

        guard data.count >= 6 else { throw NPYError.invalidFormat }
        offset = 8

        let major = data[6]
        var headerLen: Int
        if major == 1 {
            guard data.count >= offset + 2 else { throw NPYError.invalidFormat }
            headerLen = Int(data[offset]) | (Int(data[offset + 1]) << 8)
            offset += 2
        } else if major == 2 || major == 3 {
            guard data.count >= offset + 4 else { throw NPYError.invalidFormat }
            headerLen = Int(data[offset]) | (Int(data[offset + 1]) << 8) |
                       (Int(data[offset + 2]) << 16) | (Int(data[offset + 3]) << 24)
            offset += 4
        } else {
            throw NPYError.unsupportedVersion
        }

        guard data.count >= offset + headerLen else { throw NPYError.invalidFormat }
        let headerData = data[offset..<(offset + headerLen)]
        guard let headerStr = String(data: headerData, encoding: .utf8) else {
            throw NPYError.invalidFormat
        }
        offset += headerLen

        guard let shapeMatch = headerStr.range(of: "'shape':\\s*\\(([^)]*)\\)", options: .regularExpression) else {
            throw NPYError.invalidFormat
        }
        let shapeStr = String(headerStr[shapeMatch]).replacingOccurrences(of: "'shape': (", with: "").replacingOccurrences(of: ")", with: "")
        let shape = shapeStr.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }

        guard let dtypeMatch = headerStr.range(of: "'descr':\\s*'([^']*)'", options: .regularExpression) else {
            throw NPYError.invalidFormat
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
            throw NPYError.unsupportedDtype(dtypeStr)
        }

        return shape.isEmpty ? array : array.reshaped(shape)
    }

    enum NPYError: Error {
        case invalidFormat
        case unsupportedVersion
        case unsupportedDtype(String)
    }
}

// MARK: - T3 Key Remapping

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

// MARK: - Paths

let projectRoot = URL(fileURLWithPath: #file)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .path

let modelsPath = "\(projectRoot)/models"
let voicePath = "\(projectRoot)/baked_voices/samantha_full"
let refPath = "\(projectRoot)/verification_outputs/step2"

// MARK: - Main

print(String(repeating: "=", count: 80))
print("STEP 2: T3 CONDITIONING")
print(String(repeating: "=", count: 80))

do {
    let modelsURL = URL(fileURLWithPath: modelsPath)
    let voiceURL = URL(fileURLWithPath: voicePath)
    let refURL = URL(fileURLWithPath: refPath)

    let t3URL = modelsURL.appendingPathComponent("t3_fp32.safetensors")
    let configURL = modelsURL.appendingPathComponent("t3_config.json")
    let ropeFreqsURL = modelsURL.appendingPathComponent("rope_freqs_llama3.safetensors")

    let configData = try Data(contentsOf: configURL)
    let config = try JSONDecoder().decode(T3Config.self, from: configData)

    let rawWeights = try MLX.loadArrays(url: t3URL)
    let weights = remapT3Keys(rawWeights)
    let t3 = T3Model(config: config, weights: weights, ropeFreqsURL: ropeFreqsURL)

    let soul_t3 = try NPYLoader.load(contentsOf: voiceURL.appendingPathComponent("soul_t3_256.npy"))
    let t3_cond_tokens = try NPYLoader.load(contentsOf: voiceURL.appendingPathComponent("t3_cond_tokens.npy"))

    print("soul_t3: \(soul_t3.shape)")
    print("t3_cond_tokens: \(t3_cond_tokens.shape)")

    // Run conditioning
    let spkToken = t3.speakerProj(soul_t3).expandedDimensions(axis: 1)
    let condLen = t3_cond_tokens.shape[1]
    let condPositions = MLXArray(0..<condLen).asType(.int32).expandedDimensions(axis: 0)

    let speechEmb = t3.speechEmb(t3_cond_tokens)
    let speechPosEmb = t3.speechPosEmb(condPositions)
    let condSpeechEmb = speechEmb + speechPosEmb
    let perceiverOut = t3.perceiver!(condSpeechEmb)

    let emotionValue: Float = 0.5
    let emotionInput = MLXArray([emotionValue]).reshaped([1, 1, 1])
    let emotionToken = t3.emotionAdvFC!(emotionInput)

    let finalCond = concatenated([spkToken, perceiverOut, emotionToken], axis: 1)

    print("speaker_token: \(spkToken.shape)")
    print("perceiver_out: \(perceiverOut.shape)")
    print("emotion_token: \(emotionToken.shape)")
    print("final_cond: \(finalCond.shape)")

    // Load Python reference
    let refSpeaker = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("speaker_token.npy"))
    let refPerceiver = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("perceiver_out.npy"))
    let refEmotion = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("emotion_token.npy"))
    let refFinal = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("final_cond.npy"))

    // Compare
    func maxDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        return abs(a - b).max().item(Float.self)
    }

    let speakerDiff = maxDiff(spkToken, refSpeaker)
    let perceiverDiff = maxDiff(perceiverOut, refPerceiver)
    let emotionDiff = maxDiff(emotionToken, refEmotion)
    let finalDiff = maxDiff(finalCond, refFinal)

    print(String(repeating: "=", count: 80))
    print("speaker_token max_diff: \(speakerDiff)")
    print("perceiver_out max_diff: \(perceiverDiff)")
    print("emotion_token max_diff: \(emotionDiff)")
    print("final_cond max_diff: \(finalDiff)")

    let threshold: Float = 0.001
    let allPass = speakerDiff < threshold && perceiverDiff < threshold &&
                  emotionDiff < threshold && finalDiff < threshold

    print(String(repeating: "=", count: 80))
    if allPass {
        print("✅ PASSED (max_diff < \(threshold))")
    } else {
        print("❌ FAILED (max_diff >= \(threshold))")
        exit(1)
    }
    print(String(repeating: "=", count: 80))

} catch {
    print("❌ ERROR: \(error)")
    exit(1)
}
