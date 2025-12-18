import Foundation
import MLX
import Nightingale

// MARK: - Tokenization

func loadVocab(from url: URL) throws -> ([String: Int], [(String, String)]) {
    let data = try Data(contentsOf: url)
    let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    guard let model = json?["model"] as? [String: Any],
          let vocabDict = model["vocab"] as? [String: Int] else {
        fatalError("Invalid tokenizer.json format")
    }

    // Load BPE merges
    var merges: [(String, String)] = []
    if let mergeStrings = model["merges"] as? [String] {
        for mergeStr in mergeStrings {
            let parts = mergeStr.split(separator: " ", maxSplits: 1)
            if parts.count == 2 {
                merges.append((String(parts[0]), String(parts[1])))
            }
        }
    }
    return (vocabDict, merges)
}

func bpeEncode(word: String, vocab: [String: Int], merges: [(String, String)]) -> [Int] {
    var symbols = word.map { String($0) }
    var mergeRanks: [String: Int] = [:]
    for (index, merge) in merges.enumerated() {
        let key = "\(merge.0) \(merge.1)"
        mergeRanks[key] = index
    }

    while symbols.count > 1 {
        var bestPair: (Int, String, String)? = nil
        var bestRank = Int.max

        for i in 0..<(symbols.count - 1) {
            let pair = "\(symbols[i]) \(symbols[i + 1])"
            if let rank = mergeRanks[pair], rank < bestRank {
                bestRank = rank
                bestPair = (i, symbols[i], symbols[i + 1])
            }
        }

        guard let (idx, left, right) = bestPair else { break }
        symbols[idx] = left + right
        symbols.remove(at: idx + 1)
    }

    var tokenIds: [Int] = []
    for symbol in symbols {
        if let tokenId = vocab[symbol] {
            tokenIds.append(tokenId)
        } else {
            tokenIds.append(1)  // [UNK]
        }
    }
    return tokenIds
}

func tokenize(_ text: String, vocab: [String: Int], merges: [(String, String)]) -> [Int] {
    var tokens: [Int] = []
    let words = text.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
    for word in words {
        let wordTokens = bpeEncode(word: word, vocab: vocab, merges: merges)
        tokens.append(contentsOf: wordTokens)
    }
    return tokens
}

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

// MARK: - Utility

func maxDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    return abs(a - b).max().item(Float.self)
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
let refPath = "\(projectRoot)/verification_outputs/step3"

// MARK: - Main

let testText = "Hello world"

print(String(repeating: "=", count: 80))
print("STEP 3: T3 TRANSFORMER VERIFICATION")
print(String(repeating: "=", count: 80))
print("Test text: \"\(testText)\"")

do {
    let modelsURL = URL(fileURLWithPath: modelsPath)
    let voiceURL = URL(fileURLWithPath: voicePath)
    let refURL = URL(fileURLWithPath: refPath)

    // Load tokenizer
    let tokenizerURL = modelsURL.appendingPathComponent("tokenizer.json")
    let (vocab, merges) = try loadVocab(from: tokenizerURL)

    // Tokenize text
    let tokens = tokenize(testText, vocab: vocab, merges: merges)
    print("text_tokens: \(tokens)")

    // Add SOT/EOT tokens
    let sot: Int32 = 3  // start_text_token
    let eot: Int32 = 4  // stop_text_token
    var tokensWithSotEot = [sot] + tokens.map { Int32($0) } + [eot]
    let textTokens = MLXArray(tokensWithSotEot).expandedDimensions(axis: 0)
    print("text_tokens with SOT/EOT: \(tokensWithSotEot)")

    // Load T3 model
    let t3URL = modelsURL.appendingPathComponent("t3_fp32.safetensors")
    let configURL = modelsURL.appendingPathComponent("t3_config.json")
    let ropeFreqsURL = modelsURL.appendingPathComponent("rope_freqs_llama3.safetensors")

    let configData = try Data(contentsOf: configURL)
    let config = try JSONDecoder().decode(T3Config.self, from: configData)

    let rawWeights = try MLX.loadArrays(url: t3URL)
    let weights = remapT3Keys(rawWeights)
    let t3 = T3Model(config: config, weights: weights, ropeFreqsURL: ropeFreqsURL)

    // Load voice data
    let soul_t3 = try NPYLoader.load(contentsOf: voiceURL.appendingPathComponent("soul_t3_256.npy"))
    let t3_cond_tokens = try NPYLoader.load(contentsOf: voiceURL.appendingPathComponent("t3_cond_tokens.npy"))
    let emotion_adv = try NPYLoader.load(contentsOf: voiceURL.appendingPathComponent("emotion_adv.npy"))
    let emotionValue = emotion_adv.reshaped([-1]).asArray(Float.self)[0]
    print("emotion_adv: \(emotionValue)")

    // =========================================================================
    // STEP 2: T3 CONDITIONING (already verified)
    // =========================================================================
    let spkToken = t3.speakerProj(soul_t3).expandedDimensions(axis: 1)
    let condLen = t3_cond_tokens.shape[1]
    let condPositions = MLXArray(0..<condLen).asType(.int32).expandedDimensions(axis: 0)

    let speechEmb = t3.speechEmb(t3_cond_tokens)
    let speechPosEmb = t3.speechPosEmb(condPositions)
    let condSpeechEmb = speechEmb + speechPosEmb
    let perceiverOut = t3.perceiver!(condSpeechEmb)

    // emotionValue is loaded from NPY file above (not hardcoded 0.5)
    let emotionInput = MLXArray([emotionValue]).reshaped([1, 1, 1])
    let emotionToken = t3.emotionAdvFC!(emotionInput)

    let finalCond = concatenated([spkToken, perceiverOut, emotionToken], axis: 1)
    print("final_cond: \(finalCond.shape)")

    // =========================================================================
    // STEP 3: T3 TRANSFORMER
    // =========================================================================
    print("\nPreparing transformer input...")

    // Text embeddings
    let textEmb = t3.textEmb(textTokens)
    let textLen = textTokens.shape[1]
    let textPositions = MLXArray(0..<textLen).asType(.int32).expandedDimensions(axis: 0)
    let textPosEmb = t3.textPosEmb(textPositions)
    let textEmbWithPos = textEmb + textPosEmb
    print("text_emb (with pos): \(textEmbWithPos.shape)")

    // Concatenate: [conditioning | text]
    let transformerInput = concatenated([finalCond, textEmbWithPos], axis: 1)
    print("transformer_input: \(transformerInput.shape)")

    // Create causal mask - conditioning can attend bidirectionally
    let numCondTokens = finalCond.shape[1]  // 34
    let seqLen = transformerInput.shape[1]  // 42
    print("Creating causal mask (cond_len=\(numCondTokens), seq_len=\(seqLen))")

    // Create upper triangular mask with -inf (causal)
    var maskData = [Float](repeating: 0.0, count: seqLen * seqLen)
    for i in 0..<seqLen {
        for j in 0..<seqLen {
            if j > i {
                maskData[i * seqLen + j] = -Float.infinity
            }
        }
    }
    // Allow conditioning tokens to attend bidirectionally (set their rows to 0)
    for i in 0..<numCondTokens {
        for j in 0..<seqLen {
            maskData[i * seqLen + j] = 0.0
        }
    }
    let mask = MLXArray(maskData).reshaped([seqLen, seqLen])
    print("attention_mask: \(mask.shape)")

    // Run through transformer layers manually (matching Python reference)
    print("\nRunning transformer forward pass...")
    var hiddenStates = transformerInput
    for (i, layer) in t3.layers.enumerated() {
        if let transformerBlock = layer as? TransformerBlock {
            hiddenStates = transformerBlock.forward(hiddenStates, mask: mask, cache: nil)
        } else if let inspectableBlock = layer as? InspectableTransformerBlock {
            hiddenStates = inspectableBlock.forward(hiddenStates, mask: mask, cache: nil)
        }
        if i == 0 || i == 14 || i == 29 {  // First, middle, last
            let mean = hiddenStates.mean().item(Float.self)
            let variance = ((hiddenStates - mean) * (hiddenStates - mean)).mean().item(Float.self)
            let std = sqrt(variance)
            print("  After layer \(i): mean=\(String(format: "%.8f", mean)), std=\(String(format: "%.8f", std))")
        }
    }

    // Apply final layer norm (matching Python: hidden_states = tfmr_out.hidden_states[-1])
    hiddenStates = t3.norm(hiddenStates)
    let transformerOutput = hiddenStates
    print("transformer_output (after norm): \(transformerOutput.shape)")
    let mean = transformerOutput.mean().item(Float.self)
    let variance = ((transformerOutput - mean) * (transformerOutput - mean)).mean().item(Float.self)
    let std = sqrt(variance)
    print("  mean: \(String(format: "%.8f", mean))")
    print("  std: \(String(format: "%.8f", std))")


    // Extract text portion (skip conditioning tokens)
    let textHidden = transformerOutput[0..., numCondTokens..., 0...]
    print("text_hidden (text portion only): \(textHidden.shape)")

    // =========================================================================
    // COMPARE AGAINST PYTHON REFERENCE
    // =========================================================================
    print("\n" + String(repeating: "=", count: 80))
    print("COMPARISON AGAINST PYTHON REFERENCE")
    print(String(repeating: "=", count: 80))

    let refTransformerInput = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("transformer_input.npy"))
    let refTransformerOutput = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("transformer_output.npy"))
    let refTextHidden = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("text_hidden.npy"))

    let inputDiff = maxDiff(transformerInput, refTransformerInput)
    let outputDiff = maxDiff(transformerOutput, refTransformerOutput)
    let textHiddenDiff = maxDiff(textHidden, refTextHidden)

    print("transformer_input max_diff: \(String(format: "%.2e", inputDiff))")
    print("transformer_output max_diff: \(String(format: "%.2e", outputDiff))")
    print("text_hidden max_diff: \(String(format: "%.2e", textHiddenDiff))")

    let threshold: Float = 0.001
    let step3Pass = inputDiff < threshold && outputDiff < threshold && textHiddenDiff < threshold

    // =========================================================================
    // SUMMARY
    // =========================================================================
    print("\n" + String(repeating: "=", count: 80))
    print("VERIFICATION SUMMARY")
    print(String(repeating: "=", count: 80))
    print("Step 3 (Transformer): \(step3Pass ? "✅ PASSED (max_diff < \(threshold))" : "❌ FAILED (max_diff >= \(threshold))")")
    print(String(repeating: "=", count: 80))

    if !step3Pass {
        exit(1)
    }

} catch {
    print("❌ ERROR: \(error)")
    exit(1)
}
