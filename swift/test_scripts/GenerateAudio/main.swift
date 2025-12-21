import Foundation
import MLX
import MLXNN
import MLXRandom
import Nightingale

// MARK: - WAV File Writing

func writeWAV(audio: [Float], sampleRate: Int, to url: URL) throws {
    var data = Data()
    let numChannels: UInt16 = 1
    let bitsPerSample: UInt16 = 16
    let byteRate = UInt32(sampleRate * Int(numChannels) * Int(bitsPerSample) / 8)
    let blockAlign = UInt16(numChannels * bitsPerSample / 8)
    let dataSize = UInt32(audio.count * Int(bitsPerSample) / 8)
    let fileSize = 36 + dataSize

    data.append(contentsOf: "RIFF".utf8)
    data.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
    data.append(contentsOf: "WAVE".utf8)
    data.append(contentsOf: "fmt ".utf8)
    data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })
    data.append(contentsOf: "data".utf8)
    data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

    for sample in audio {
        let scaled = Int16(max(-1.0, min(1.0, sample)) * 32767.0)
        data.append(contentsOf: withUnsafeBytes(of: scaled.littleEndian) { Array($0) })
    }
    try data.write(to: url)
}

// MARK: - Tokenizer (from VerifyLive)

func loadTokenizer(from url: URL) throws -> ([String: Int], [(String, String)]) {
    let data = try Data(contentsOf: url)
    let json = try JSONSerialization.jsonObject(with: data, options: []) as! [String: Any]
    
    if let vocab = json["model"] as? [String: Int] {
        return (vocab, [])
    }
    
    guard let model = json["model"] as? [String: Any],
          let vocabRaw = model["vocab"] as? [String: Int],
          let mergesRaw = model["merges"] as? [String] else {
        fatalError("Invalid tokenizer.json format")
    }
    
    let merges = mergesRaw.map { line -> (String, String) in
        let parts = line.split(separator: " ")
        return (String(parts[0]), String(parts[1]))
    }
    return (vocabRaw, merges)
}

func normalizeTextForTokenizer(_ text: String) -> String {
    var result = text.lowercased()
    result = result.replacingOccurrences(of: "\n", with: " ")
    result = result.replacingOccurrences(of: "\t", with: " ")
    while result.contains("  ") {
        result = result.replacingOccurrences(of: "  ", with: " ")
    }
    result = result.trimmingCharacters(in: .whitespaces)
    return result
}

func tokenize(_ text: String, vocab: [String: Int], merges: [(String, String)], languageId: String? = nil) -> [Int] {
    var result: [Int] = []
    
    if let langId = languageId, let langToken = vocab["<|lang:\(langId)|>"] {
        result.append(langToken)
    }
    
    for char in text {
        let charStr = String(char)
        if let tokenId = vocab[charStr] {
            result.append(tokenId)
        } else if let unkId = vocab["<unk>"] {
            result.append(unkId)
        }
    }
    
    return result
}

// MARK: - T3 Key Remapping

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

func remapT3Keys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var remapped: [String: MLXArray] = [:]
    for (key, value) in weights {
        if let newKey = remapT3Key(key) {
            remapped[newKey] = value
        }
    }
    return remapped
}

// MARK: - Main

print(String(repeating: "=", count: 80))
print("NIGHTINGALE SWIFT TTS - AUDIO GENERATION")
print(String(repeating: "=", count: 80))

let PROJECT_ROOT = "/Users/a10n/Projects/nightingale_TTS"
let modelDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/models")
let outputDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/test_audio/swift")

try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

print("\nLoading models...")

// Load tokenizer
let tokenizerURL = URL(fileURLWithPath: "\(PROJECT_ROOT)/models/chatterbox/grapheme_mtl_merged_expanded_v1.json")
let (vocab, merges) = try loadTokenizer(from: tokenizerURL)
print("  ✅ Tokenizer loaded (\(vocab.count) tokens)")

// Load T3 model - use weights-based init to get InspectableTransformerBlocks
let t3URL = modelDir.appendingPathComponent("mlx/t3_fp32.safetensors")
let t3RawWeights = try MLX.loadArrays(url: t3URL)
let t3Weights = remapT3Keys(t3RawWeights)  // Remap Python key names to Swift
print("  Remapped \(t3Weights.count) T3 weights")
// Also load pre-computed RoPE frequencies for exact Python match
let ropeURL = modelDir.appendingPathComponent("mlx/rope_freqs.npy")
let t3 = T3Model(config: T3Config.default, weights: t3Weights, ropeFreqsURL: ropeURL)
eval(t3)
print("  ✅ T3 model loaded")

// Load voice conditioning (Samantha)
// T3 uses files from baked_voices/<voice>/npy/
let t3VoiceDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/baked_voices/samantha/npy")
// S3Gen uses files from baked_voices/<voice>/npy_original/
let s3VoiceDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/baked_voices/samantha/npy_original")

let condTokens = try NPYLoader.load(contentsOf: t3VoiceDir.appendingPathComponent("t3_cond_tokens.npy"))
let speakerEmb = try NPYLoader.load(contentsOf: t3VoiceDir.appendingPathComponent("soul_t3_256.npy"))
let emotionAdv = try NPYLoader.load(contentsOf: t3VoiceDir.appendingPathComponent("emotion_adv.npy"))
let s3Embedding = try NPYLoader.load(contentsOf: s3VoiceDir.appendingPathComponent("soul_s3_192.npy"))
let s3PromptFeat = try NPYLoader.load(contentsOf: s3VoiceDir.appendingPathComponent("prompt_feat.npy"))
let s3PromptToken = try NPYLoader.load(contentsOf: s3VoiceDir.appendingPathComponent("prompt_token.npy"))

eval(condTokens, speakerEmb, emotionAdv, s3Embedding, s3PromptFeat, s3PromptToken)
print("  ✅ Voice conditioning loaded (Samantha)")

// Load S3Gen (reuse loading logic from VerifyLive)
let flowURL = modelDir.appendingPathComponent("mlx/python_flow_weights.safetensors")
let vocoderURL = modelDir.appendingPathComponent("mlx/vocoder_weights.safetensors")

func remapS3Key(_ key: String) -> String? {
    var k = key
    if k.hasPrefix("conv_pre") || k.hasPrefix("conv_post") || k.hasPrefix("resblocks") || k.hasPrefix("ups") {
        k = "vocoder." + k
        k = k.replacingOccurrences(of: "conv_pre", with: "convPre")
        k = k.replacingOccurrences(of: "conv_post", with: "convPost")
        k = k.replacingOccurrences(of: "activations1", with: "acts1")
        k = k.replacingOccurrences(of: "activations2", with: "acts2")
    }
    if k.contains("f0_predictor.") {
        k = k.replacingOccurrences(of: "f0_predictor.condnet.0.", with: "vocoder.f0Predictor.convs.0.")
        k = k.replacingOccurrences(of: "f0_predictor.condnet.2.", with: "vocoder.f0Predictor.convs.1.")
        k = k.replacingOccurrences(of: "f0_predictor.condnet.4.", with: "vocoder.f0Predictor.convs.2.")
        k = k.replacingOccurrences(of: "f0_predictor.condnet.6.", with: "vocoder.f0Predictor.convs.3.")
        k = k.replacingOccurrences(of: "f0_predictor.condnet.8.", with: "vocoder.f0Predictor.convs.4.")
        k = k.replacingOccurrences(of: "f0_predictor.classifier.", with: "vocoder.f0Predictor.classifier.")
        return k
    }
    if k.contains("m_source.") { k = k.replacingOccurrences(of: "m_source.l_linear.", with: "vocoder.mSource.linear."); return k }
    if k.contains("source_downs.") { k = k.replacingOccurrences(of: "source_downs.", with: "vocoder.sourceDowns."); return k }
    if k.contains("source_resblocks.") {
        k = k.replacingOccurrences(of: "source_resblocks.", with: "vocoder.sourceResBlocks.")
        k = k.replacingOccurrences(of: "activations1", with: "acts1")
        k = k.replacingOccurrences(of: "activations2", with: "acts2")
        return k
    }
    if k.hasPrefix("decoder.") {
        k = k.replacingOccurrences(of: ".attn1.", with: ".attention.")
        k = k.replacingOccurrences(of: ".to_q.", with: ".queryProj.")
        k = k.replacingOccurrences(of: ".to_k.", with: ".keyProj.")
        k = k.replacingOccurrences(of: ".to_v.", with: ".valueProj.")
        k = k.replacingOccurrences(of: ".to_out.0.", with: ".outProj.")
        k = k.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.layers.0.")
        k = k.replacingOccurrences(of: ".ff.net.2.", with: ".ff.layers.1.")
        k = k.replacingOccurrences(of: ".norm3.", with: ".norm2.")
        k = k.replacingOccurrences(of: "down_blocks_", with: "downBlocks.")
        k = k.replacingOccurrences(of: "mid_blocks_", with: "midBlocks.")
        k = k.replacingOccurrences(of: "up_blocks_", with: "upBlocks.")
        k = k.replacingOccurrences(of: ".transformer_", with: ".transformers.")
    }
    if k.hasPrefix("encoder.") {
        k = k.replacingOccurrences(of: "encoder.encoder.", with: "encoder.downEncoder.")
        k = k.replacingOccurrences(of: "encoder.decoder.", with: "encoder.upEncoder.")
    }
    return k
}

func remapS3Keys(_ weights: [String: MLXArray], transposeConv1d: Bool = false) -> [String: MLXArray] {
    var remapped: [String: MLXArray] = [:]
    for (key, value) in weights {
        if let newKey = remapS3Key(key) {
            let isEmbedding = newKey.contains("Embedding.weight") || newKey.contains("speechEmb.weight")
            let isLinear = newKey.hasSuffix(".weight") && value.ndim == 2 && !isEmbedding
            let isConv1d = newKey.hasSuffix(".weight") && value.ndim == 3
            if isLinear {
                remapped[newKey] = value.T
            } else if isConv1d && transposeConv1d {
                let isConvTranspose = newKey.contains("ups.") && newKey.hasSuffix(".weight")
                remapped[newKey] = isConvTranspose ? value.transposed(1, 2, 0) : value.transposed(0, 2, 1)
            } else {
                remapped[newKey] = value
            }
        }
    }
    return remapped
}

var s3gen = S3Gen(flowWeights: [:], vocoderWeights: nil)
let vocoderWeights = try MLX.loadArrays(url: vocoderURL)

// Debug: Print mSource keys being remapped
print("  DEBUG: Checking mSource key remapping...")
for key in vocoderWeights.keys where key.contains("m_source") {
    if let remapped = remapS3Key(key), let value = vocoderWeights[key] {
        print("    '\(key)' -> '\(remapped)' (shape \(value.shape))")
    }
}

s3gen.update(parameters: ModuleParameters.unflattened(remapS3Keys(vocoderWeights, transposeConv1d: true)))

// Debug: Verify mSource.linear weights after loading
print("  DEBUG: mSource.linear weights after update:")
let mSourceWeight = s3gen.vocoder.mSource.linear.weight
eval(mSourceWeight)
print("    shape: \(mSourceWeight.shape)")
print("    values: \(mSourceWeight.flattened().asArray(Float.self))")
print("    Expected Python: [-0.00117903, -0.00026658, -0.00039365, ...]")

let flowWeights = try MLX.loadArrays(url: flowURL)
s3gen.update(parameters: ModuleParameters.unflattened(remapS3Keys(flowWeights, transposeConv1d: false)))
eval(s3gen)

// Enable vocoder debug output
Mel2Wav.debugEnabled = true
print("  ✅ S3Gen loaded")

let speechEmbMatrix = t3.speechEmb.weight
eval(speechEmbMatrix)

print("\n" + String(repeating: "=", count: 80))
print("GENERATING AUDIO (E2E Step 9)")
print(String(repeating: "=", count: 80))

// Set deterministic seed to match Python (SEED = 42)
MLXRandom.seed(42)

// User requested text
let testText = "Wow! I absolutely cannot believe that it worked on the first try!"
let lang = "en"

print("\nGenerating audio for:")
print("  Text: \(testText)")

// Tokenize
let normalized = normalizeTextForTokenizer(testText)
var tokens = tokenize(normalized, vocab: vocab, merges: merges, languageId: lang)
if let sotToken = vocab["<|startoftranscript|>"] { tokens.insert(sotToken, at: 0) }
if let eotToken = vocab["<|endoftranscript|>"] { tokens.append(eotToken) }
let textTokens = MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0)
print("  Tokens: \(tokens.count)")

// T3 Generation - use realistic settings for natural speech
print("  Generating speech tokens (temp=0.7, topP=0.95)...")
let speechTokensRaw = t3.generate(
    textTokens: textTokens,
    speakerEmb: speakerEmb,
    condTokens: condTokens,
    maxTokens: 500,
    temperature: 0.7,
    emotionValue: emotionAdv[0, 0].item(Float.self),
    cfgWeight: 0.5,
    repetitionPenalty: 1.2,
    topP: 0.95,
    minP: 0.05
)
var speechTokensClean = speechTokensRaw.filter { $0 != 6561 && $0 != 6562 }
if speechTokensClean.isEmpty { speechTokensClean = [1] }
let speechTokens = MLXArray(speechTokensClean.map { Int32($0) }).expandedDimensions(axis: 0)
eval(speechTokens)
print("  Generated \(speechTokensClean.count) speech tokens")

// S3Gen
print("  Running S3Gen...")
let audio = s3gen.generate(
    tokens: speechTokens,
    speakerEmb: s3Embedding,
    speechEmbMatrix: speechEmbMatrix,
    promptToken: s3PromptToken,
    promptFeat: s3PromptFeat
)
eval(audio)

let audioSamples = audio.squeezed().asArray(Float.self)
let duration = Float(audioSamples.count) / 24000.0
print("  Audio: \(audioSamples.count) samples (\(String(format: "%.2f", duration))s)")

// Frequency analysis to verify correct output
var lowEnergy: Float = 0
var highEnergy: Float = 0
let sampleRate: Float = 24000
for freq in stride(from: 100, through: 500, by: 50) {
    var realSum: Float = 0, imagSum: Float = 0
    for (i, sample) in audioSamples.enumerated() {
        let angle = 2.0 * Float.pi * Float(freq) * Float(i) / sampleRate
        realSum += sample * cos(angle)
        imagSum += sample * sin(angle)
    }
    lowEnergy += sqrt(realSum * realSum + imagSum * imagSum)
}
for freq in stride(from: 5000, through: 10000, by: 500) {
    var realSum: Float = 0, imagSum: Float = 0
    for (i, sample) in audioSamples.enumerated() {
        let angle = 2.0 * Float.pi * Float(freq) * Float(i) / sampleRate
        realSum += sample * cos(angle)
        imagSum += sample * sin(angle)
    }
    highEnergy += sqrt(realSum * realSum + imagSum * imagSum)
}
let totalEnergy = lowEnergy + highEnergy
print("  Frequency analysis:")
print("    Low freq (100-500 Hz): \(String(format: "%.1f", 100 * lowEnergy / totalEnergy))%")
print("    High freq (5k-10k Hz): \(String(format: "%.1f", 100 * highEnergy / totalEnergy))%")
if lowEnergy > highEnergy {
    print("    ✅ Correct: Low frequency dominant (speech)")
} else {
    print("    ⚠️  Warning: High frequency dominant (possible frequency inversion)")
}

// Save to test_audio folder
let outputPath = URL(fileURLWithPath: "\(PROJECT_ROOT)/test_audio/swift_output.wav")
try writeWAV(audio: audioSamples, sampleRate: 24000, to: outputPath)
print("  ✅ Saved: \(outputPath.path)")

print("\n" + String(repeating: "=", count: 80))
print("✅ AUDIO GENERATION COMPLETE!")
print("   Output: \(outputPath.path)")
print(String(repeating: "=", count: 80))
