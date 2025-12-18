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
    // Start with individual characters
    var symbols = word.map { String($0) }

    // Create a set of merge rules for fast lookup, preserving order via index
    var mergeRanks: [String: Int] = [:]
    for (index, merge) in merges.enumerated() {
        let key = "\(merge.0) \(merge.1)"
        mergeRanks[key] = index
    }

    // Apply merges iteratively
    while symbols.count > 1 {
        // Find the pair with the lowest merge rank
        var bestPair: (Int, String, String)? = nil  // (index, left, right)
        var bestRank = Int.max

        for i in 0..<(symbols.count - 1) {
            let pair = "\(symbols[i]) \(symbols[i + 1])"
            if let rank = mergeRanks[pair], rank < bestRank {
                bestRank = rank
                bestPair = (i, symbols[i], symbols[i + 1])
            }
        }

        // If no merge found, we're done
        guard let (idx, left, right) = bestPair else { break }

        // Apply the merge
        symbols[idx] = left + right
        symbols.remove(at: idx + 1)
    }

    // Convert symbols to token IDs
    var tokenIds: [Int] = []
    for symbol in symbols {
        if let tokenId = vocab[symbol] {
            tokenIds.append(tokenId)
        } else {
            // Unknown token - use [UNK] = 1
            tokenIds.append(1)
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

// MARK: - Main

let testText = "Hello world"
let modelsPath = "/Users/a10n/Projects/nightingale/models"
let voicePath = "/Users/a10n/Projects/nightingale/baked_voices/samantha_full"
let refPath = "/Users/a10n/Projects/nightingale/verification_outputs/e2e_steps1_4"

print(String(repeating: "=", count: 80))
print("END-TO-END VERIFICATION: STEPS 1-4")
print(String(repeating: "=", count: 80))
print("Test text: \"\(testText)\"")

do {
    let modelsURL = URL(fileURLWithPath: modelsPath)
    let voiceURL = URL(fileURLWithPath: voicePath)
    let refURL = URL(fileURLWithPath: refPath)

    // Load tokenizer
    let tokenizerURL = modelsURL.appendingPathComponent("tokenizer.json")
    let (vocab, merges) = try loadVocab(from: tokenizerURL)

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
    // STEP 1: TEXT TOKENIZATION
    // =========================================================================
    print("\n" + String(repeating: "=", count: 80))
    print("STEP 1: TEXT TOKENIZATION")
    print(String(repeating: "=", count: 80))

    let tokens = tokenize(testText, vocab: vocab, merges: merges)
    print("Token IDs: \(tokens)")
    print("Token count: \(tokens.count)")

    // Load Python reference
    let refTokens = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("step1_text_tokens.npy"))
    let refTokensArray = refTokens.asArray(Int32.self)

    // Compare
    let tokensInt32 = tokens.map { Int32($0) }
    let tokensMatch = tokensInt32 == refTokensArray
    let step1Pass = tokensMatch

    print("\nComparison:")
    print("  Python tokens: \(refTokensArray)")
    print("  Swift tokens:  \(tokens)")
    print("  Match: \(tokensMatch ? "✅ YES" : "❌ NO")")

    // =========================================================================
    // STEP 2: T3 CONDITIONING
    // =========================================================================
    print("\n" + String(repeating: "=", count: 80))
    print("STEP 2: T3 CONDITIONING")
    print(String(repeating: "=", count: 80))

    // Run conditioning
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

    print("speaker_token: \(spkToken.shape)")
    print("perceiver_out: \(perceiverOut.shape)")
    print("emotion_token: \(emotionToken.shape)")
    print("final_cond: \(finalCond.shape)")

    // Load Python reference
    let refSpeaker = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("step2_speaker_token.npy"))
    let refPerceiver = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("step2_perceiver_out.npy"))
    let refEmotion = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("step2_emotion_token.npy"))
    let refFinal = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("step2_final_cond.npy"))

    // Compare
    let speakerDiff = maxDiff(spkToken, refSpeaker)
    let perceiverDiff = maxDiff(perceiverOut, refPerceiver)
    let emotionDiff = maxDiff(emotionToken, refEmotion)
    let finalDiff = maxDiff(finalCond, refFinal)

    print("\nComparison:")
    print("  speaker_token max_diff: \(String(format: "%.2e", speakerDiff))")
    print("  perceiver_out max_diff: \(String(format: "%.2e", perceiverDiff))")
    print("  emotion_token max_diff: \(String(format: "%.2e", emotionDiff))")
    print("  final_cond max_diff: \(String(format: "%.2e", finalDiff))")

    let threshold: Float = 0.001
    let step2Pass = speakerDiff < threshold && perceiverDiff < threshold &&
                    emotionDiff < threshold && finalDiff < threshold

    // =========================================================================
    // STEP 3: T3 TRANSFORMER
    // =========================================================================
    print("\n" + String(repeating: "=", count: 80))
    print("STEP 3: T3 TRANSFORMER")
    print(String(repeating: "=", count: 80))

    // Add SOT/EOT tokens
    let sot: Int32 = 3
    let eot: Int32 = 4
    var tokensWithSotEot = [sot] + tokens.map { Int32($0) } + [eot]
    let textTokens = MLXArray(tokensWithSotEot).expandedDimensions(axis: 0)
    print("text_tokens with SOT/EOT: \(tokensWithSotEot)")

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

    // Create causal mask
    let numCondTokens = finalCond.shape[1]
    let seqLen = transformerInput.shape[1]
    var maskData = [Float](repeating: 0.0, count: seqLen * seqLen)
    for i in 0..<seqLen {
        for j in 0..<seqLen {
            if j > i {
                maskData[i * seqLen + j] = -Float.infinity
            }
        }
    }
    for i in 0..<numCondTokens {
        for j in 0..<seqLen {
            maskData[i * seqLen + j] = 0.0
        }
    }
    let mask = MLXArray(maskData).reshaped([seqLen, seqLen])

    // Run through transformer
    var hiddenStates = transformerInput
    for layer in t3.layers {
        if let transformerBlock = layer as? TransformerBlock {
            hiddenStates = transformerBlock.forward(hiddenStates, mask: mask, cache: nil)
        } else if let inspectableBlock = layer as? InspectableTransformerBlock {
            hiddenStates = inspectableBlock.forward(hiddenStates, mask: mask, cache: nil)
        }
    }

    // Apply final norm
    hiddenStates = t3.norm(hiddenStates)
    let transformerOutput = hiddenStates
    print("transformer_output: \(transformerOutput.shape)")

    // Extract text portion
    let textHidden = transformerOutput[0..., numCondTokens..., 0...]
    print("text_hidden: \(textHidden.shape)")

    // Load Python reference
    let refTransformerInput = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("step3_transformer_input.npy"))
    let refTransformerOutput = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("step3_transformer_output.npy"))
    let refTextHidden = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("step3_text_hidden.npy"))

    // Compare
    let transformerInputDiff = maxDiff(transformerInput, refTransformerInput)
    let transformerOutputDiff = maxDiff(transformerOutput, refTransformerOutput)
    let textHiddenDiff = maxDiff(textHidden, refTextHidden)

    print("\nComparison:")
    print("  transformer_input max_diff: \(String(format: "%.2e", transformerInputDiff))")
    print("  transformer_output max_diff: \(String(format: "%.2e", transformerOutputDiff))")
    print("  text_hidden max_diff: \(String(format: "%.2e", textHiddenDiff))")

    let step3Pass = transformerInputDiff < threshold && transformerOutputDiff < threshold && textHiddenDiff < threshold

    // =========================================================================
    // STEP 4: T3 TOKEN GENERATION (First Step)
    // =========================================================================
    print("\n" + String(repeating: "=", count: 80))
    print("STEP 4: T3 TOKEN GENERATION (First Step)")
    print(String(repeating: "=", count: 80))

    let cfgWeight: Float = 0.5
    print("CFG weight: \(cfgWeight)")

    // BOS token uses speech position 0
    let bosToken: Int32 = 6561  // start_speech_token
    let bosTokenArray = MLXArray([bosToken]).reshaped([1, 1])
    let bosEmb = t3.speechEmb(bosTokenArray)
    let bosPosEmb = t3.speechPosEmb(MLXArray([Int32(0)]).reshaped([1, 1]))
    let bosEmbWithPos = bosEmb + bosPosEmb
    print("BOS token: \(bosToken)")
    print("bos_emb (with pos): \(bosEmbWithPos.shape)")

    // Build conditioned input: [conditioning | text | BOS]
    let condInput = concatenated([finalCond, textEmbWithPos, bosEmbWithPos], axis: 1)
    print("cond_input: \(condInput.shape)")

    // Build unconditioned input: [null_conditioning | text | BOS]
    let nullSoulT3 = MLXArray.zeros(like: soul_t3)
    let nullSpkToken = t3.speakerProj(nullSoulT3).expandedDimensions(axis: 1)
    let nullFinalCond = concatenated([nullSpkToken, perceiverOut, emotionToken], axis: 1)
    let uncondInput = concatenated([nullFinalCond, textEmbWithPos, bosEmbWithPos], axis: 1)
    print("uncond_input: \(uncondInput.shape)")

    // Stack for CFG: [cond, uncond]
    let batchedInput = concatenated([condInput, uncondInput], axis: 0)
    print("batched_input (CFG): \(batchedInput.shape)")

    // Create hybrid mask: conditioning (34) bidirectional, text+BOS (9) causal
    let step4SeqLen = batchedInput.shape[1]
    let step4TotalElements = step4SeqLen * step4SeqLen
    var step4MaskData = [Float](repeating: 0.0, count: step4TotalElements)
    for i in 0..<step4SeqLen {
        for j in 0..<step4SeqLen {
            if j > i {
                step4MaskData[i * step4SeqLen + j] = -Float.infinity
            }
        }
    }
    // Allow conditioning tokens to attend bidirectionally
    for i in 0..<numCondTokens {
        for j in 0..<step4SeqLen {
            step4MaskData[i * step4SeqLen + j] = 0.0
        }
    }
    let step4Mask = MLXArray(step4MaskData).reshaped([step4SeqLen, step4SeqLen])
    print("Hybrid mask: \(step4Mask.shape)")

    // Run through transformer
    var step4HiddenStates = batchedInput
    for layer in t3.layers {
        if let transformerBlock = layer as? TransformerBlock {
            let dummyCache: KVCache? = nil
            step4HiddenStates = transformerBlock.forward(step4HiddenStates, mask: step4Mask, cache: dummyCache)
        } else if let inspectableBlock = layer as? InspectableTransformerBlock {
            let dummyCache: KVCache? = nil
            step4HiddenStates = inspectableBlock.forward(step4HiddenStates, mask: step4Mask, cache: dummyCache)
        }
    }

    // Apply final norm
    step4HiddenStates = t3.norm(step4HiddenStates)
    print("transformer_output: \(step4HiddenStates.shape)")

    // Get logits for last position (BOS token) - predicts first speech token
    let lastHidden = step4HiddenStates[0..., (step4HiddenStates.shape[1]-1)..<step4HiddenStates.shape[1], 0...]
    let allLogits = t3.speechHead(lastHidden)  // [2, 1, vocab_size]
    print("speech_head logits: \(allLogits.shape)")

    // Extract conditioned and unconditioned logits
    let condLogits = allLogits[0, 0, 0...]  // [vocab_size]
    let uncondLogits = allLogits[1, 0, 0...]  // [vocab_size]

    // Apply CFG: final = uncond + cfg_weight * (cond - uncond)
    let cfgLogits = uncondLogits + MLXArray(cfgWeight) * (condLogits - uncondLogits)
    print("  cfg_logits: \(cfgLogits.shape)")

    // Greedy prediction
    let greedyToken = argMax(cfgLogits, axis: -1).item(Int32.self)
    print("Greedy prediction (argmax): \(greedyToken)")

    // Load Python reference
    let refCondInput = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("step4_cond_input.npy"))
    let refUncondInput = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("step4_uncond_input.npy"))
    let refBatchedInput = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("step4_batched_input.npy"))
    let refStep4TransformerOutput = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("step4_transformer_output.npy"))
    let refCondLogits = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("step4_cond_logits.npy"))
    let refUncondLogits = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("step4_uncond_logits.npy"))
    let refCFGLogits = try NPYLoader.load(contentsOf: refURL.appendingPathComponent("step4_cfg_logits.npy"))

    // Compare
    let condInputDiff = maxDiff(condInput, refCondInput)
    let uncondInputDiff = maxDiff(uncondInput, refUncondInput)
    let batchedInputDiff = maxDiff(batchedInput, refBatchedInput)
    let step4TransformerOutputDiff = maxDiff(step4HiddenStates, refStep4TransformerOutput)
    let condLogitsDiff = maxDiff(condLogits, refCondLogits)
    let uncondLogitsDiff = maxDiff(uncondLogits, refUncondLogits)
    let cfgLogitsDiff = maxDiff(cfgLogits, refCFGLogits)

    print("\nComparison:")
    print("  cond_input max_diff: \(String(format: "%.2e", condInputDiff))")
    print("  uncond_input max_diff: \(String(format: "%.2e", uncondInputDiff))")
    print("  batched_input max_diff: \(String(format: "%.2e", batchedInputDiff))")
    print("  transformer_output max_diff: \(String(format: "%.2e", step4TransformerOutputDiff))")
    print("  cond_logits max_diff: \(String(format: "%.2e", condLogitsDiff))")
    print("  uncond_logits max_diff: \(String(format: "%.2e", uncondLogitsDiff))")
    print("  cfg_logits max_diff: \(String(format: "%.2e", cfgLogitsDiff))")

    let step4InputsPass = condInputDiff < threshold && uncondInputDiff < threshold && batchedInputDiff < threshold
    let step4TransformerPass = step4TransformerOutputDiff < threshold
    let step4LogitsPass = condLogitsDiff < threshold && uncondLogitsDiff < threshold && cfgLogitsDiff < threshold
    let step4GreedyPass = greedyToken == 3704  // Python greedy prediction
    let step4Pass = step4InputsPass && step4TransformerPass && step4LogitsPass && step4GreedyPass

    // =========================================================================
    // SUMMARY
    // =========================================================================
    print("\n" + String(repeating: "=", count: 80))
    print("VERIFICATION SUMMARY")
    print(String(repeating: "=", count: 80))

    print("Step 1 (Tokenization): \(step1Pass ? "✅ PASSED" : "❌ FAILED")")
    print("Step 2 (Conditioning): \(step2Pass ? "✅ PASSED (max_diff < \(threshold))" : "❌ FAILED (max_diff >= \(threshold))")")
    print("Step 3 (Transformer): \(step3Pass ? "✅ PASSED (max_diff < \(threshold))" : "❌ FAILED (max_diff >= \(threshold))")")
    print("Step 4 (Token Generation): \(step4Pass ? "✅ PASSED (max_diff < \(threshold), token == 3704)" : "❌ FAILED")")

    let allPass = step1Pass && step2Pass && step3Pass && step4Pass

    print(String(repeating: "=", count: 80))
    if allPass {
        print("✅ ALL TESTS PASSED")
    } else {
        print("❌ SOME TESTS FAILED")
        exit(1)
    }
    print(String(repeating: "=", count: 80))

} catch {
    print("❌ ERROR: \(error)")
    exit(1)
}
