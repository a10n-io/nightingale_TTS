import Foundation
import MLX
import MLXRandom
import Nightingale

// MARK: - T3 Key Remapping (copied from ChatterboxEngine)

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

    // Handle FP32 HuggingFace format: tfmr.layers.* → layers.*
    if k.hasPrefix("tfmr.layers.") {
        k = String(k.dropFirst("tfmr.".count))
    }
    // Handle Q4 format: tfmr.model.* (but skip embed_tokens)
    else if k.hasPrefix("tfmr.model.") {
        k = String(k.dropFirst("tfmr.model.".count))
        if k.hasPrefix("embed_tokens") { return nil }
    }
    // Handle final norm: tfmr.norm.weight → norm.weight
    else if k == "tfmr.norm.weight" {
        k = "norm.weight"
    }
    // Skip other tfmr.* keys
    else if k.hasPrefix("tfmr.") {
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

// MARK: - Main

print(String(repeating: "=", count: 80))
print("NIGHTINGALE - TEST T3 GENERATION")
print(String(repeating: "=", count: 80))
print()

// Paths
let modelsPath = "/Users/a10n/Projects/nightingale/models"
let voicePath = "/Users/a10n/Projects/nightingale/baked_voices/samantha_full"

print("Models: \(modelsPath)")
print("Voice: \(voicePath)")
print()

do {
    // Load T3 model
    print("Loading T3 model...")
    let modelsURL = URL(fileURLWithPath: modelsPath)
    let t3URL = modelsURL.appendingPathComponent("t3_fp32.safetensors")
    let configURL = modelsURL.appendingPathComponent("t3_config.json")
    let tokenizerURL = modelsURL.appendingPathComponent("tokenizer.json")

    // Load config
    let configData = try Data(contentsOf: configURL)
    let config = try JSONDecoder().decode(T3Config.self, from: configData)
    print("✅ T3 config loaded")

    // Load T3 weights and create model
    print("Loading T3 weights and creating model (~2GB, this may take a moment)...")
    let rawWeights = try MLX.loadArrays(url: t3URL)
    print("✅ Raw T3 weights loaded (\(rawWeights.count) arrays)")

    // Remap keys from Python naming to Swift naming
    let weights = remapT3Keys(rawWeights)
    print("✅ Keys remapped to Swift naming (\(weights.count) arrays)")

    let ropeFreqsURL = modelsURL.appendingPathComponent("rope_freqs_llama3.safetensors")
    let t3 = T3Model(config: config, weights: weights, ropeFreqsURL: ropeFreqsURL)
    print("✅ T3 model created")
    print()

    // Load prebaked voice
    print("Loading prebaked voice...")
    let voiceURL = URL(fileURLWithPath: voicePath)

    let soul_t3 = try NPYLoader.load(contentsOf: voiceURL.appendingPathComponent("soul_t3_256.npy"))
    print("✅ soul_t3_256.npy: \(soul_t3.shape)")

    let t3_cond_tokens = try NPYLoader.load(contentsOf: voiceURL.appendingPathComponent("t3_cond_tokens.npy"))
    print("✅ t3_cond_tokens.npy: \(t3_cond_tokens.shape)")
    print()

    // Load tokenizer for proper BPE encoding
    print("Loading tokenizer...")

    // For this test, we'll use ChatterboxEngine which has loadModels that loads tokenizer
    // Simpler approach: just use fixed expected Python tokens for now
    let testText = "Hello world"
    print("Test text: \"\(testText)\"")

    // Use Python's COMPLETE tokenization (with SOT/EOT tokens)
    // Python tokenization process:
    // 1. punc_norm("Hello world") → "Hello world."
    // 2. BPE tokenize "Hello world." → [284, 18, 84, 28, 2, 179, 79, 9]
    // 3. Add SOT (255) at start → [255, 284, 18, 84, 28, 2, 179, 79, 9]
    // 4. Add EOT (0) at end → [255, 284, 18, 84, 28, 2, 179, 79, 9, 0]
    let tokens = [255, 284, 18, 84, 28, 2, 179, 79, 9, 0]  // Full tokenization with SOT/EOT (10 tokens)
    let textTokens = MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0)
    print("Full tokens (SOT + BPE + EOT): \(tokens) (count: \(tokens.count))")
    print("Text tokens shape: \(textTokens.shape)")
    print()

    print("✅ Using complete Python tokenization:")
    print("   Token 255 = START_TEXT (SOT)")
    print("   Tokens [284, 18, 84, 28, 2, 179, 79, 9] = \"Hello world.\" (BPE with period)")
    print("   Token 0 = END_TEXT (EOT)")
    print("   Total: 10 tokens (1 + 8 + 1)")
    print()

    // Run T3 generation
    print("Running T3 generation...")
    print("Parameters:")
    print("  maxTokens: 150 (default)")
    print("  temperature: 0.4")
    print("  repetitionPenalty: 1.2 (FP32 default)")
    print("  seed: 42")
    print()

    MLXRandom.seed(42)
    let startTime = CFAbsoluteTimeGetCurrent()

    let speechTokens = t3.generate(
        textTokens: textTokens,
        speakerEmb: soul_t3,
        condTokens: t3_cond_tokens,
        maxTokens: 150,
        temperature: 0.4,
        repetitionPenalty: 1.2  // Match ChatterboxEngine (FP32 default)
    )

    let elapsed = CFAbsoluteTimeGetCurrent() - startTime

    print()
    print(String(repeating: "=", count: 80))
    print("✅ T3 GENERATION COMPLETE")
    print(String(repeating: "=", count: 80))
    print()
    print("Generated speech tokens: \(speechTokens.count)")
    print("Time: \(String(format: "%.2f", elapsed))s")
    print("Tokens/sec: \(String(format: "%.1f", Double(speechTokens.count) / elapsed))")
    print()
    print("First 20 tokens: \(Array(speechTokens.prefix(20)))")
    print("Last 20 tokens: \(Array(speechTokens.suffix(20)))")
    print()
    print("Token range: [\(speechTokens.min()!), \(speechTokens.max()!)]")

    // Save for comparison with Python
    let outputURL = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale/test_audio/swift_speech_tokens.npy")
    let tokensArray = MLXArray(speechTokens.map { Int32($0) })
    try tokensArray.save(npy: outputURL)
    print()
    print("Saved tokens to: \(outputURL.path)")
    print()
    print("Next step: Test S3Gen to convert these tokens to audio!")

} catch {
    print()
    print("❌ ERROR: \(error)")
    print()
    exit(1)
}
