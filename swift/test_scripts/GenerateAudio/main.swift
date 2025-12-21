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

// MARK: - Main

print(String(repeating: "=", count: 80))
print("NIGHTINGALE SWIFT TTS - AUDIO GENERATION")
print(String(repeating: "=", count: 80))

let PROJECT_ROOT = "/Users/a10n/Projects/nightingale_TTS"
let modelDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/models")
let outputDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/test_audio")

try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

let texts = [
    "Hello! This is a test of the Swift text to speech system.",
    "The quick brown fox jumps over the lazy dog.",
    "Welcome to Nightingale."
]

print("\nLoading models...")

// Load tokenizer
let tokenizerURL = URL(fileURLWithPath: "\(PROJECT_ROOT)/models/chatterbox/grapheme_mtl_merged_expanded_v1.json")
let (vocab, merges) = try loadTokenizer(from: tokenizerURL)
print("  ✅ Tokenizer loaded (\(vocab.count) tokens)")

// Load T3 model
let t3URL = modelDir.appendingPathComponent("mlx/t3_fp32.safetensors")
let t3Weights = try MLX.loadArrays(url: t3URL)
let t3 = T3Model(config: T3Config.default)
let t3Params = ModuleParameters.unflattened(t3Weights)
t3.update(parameters: t3Params)
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
s3gen.update(parameters: ModuleParameters.unflattened(remapS3Keys(vocoderWeights, transposeConv1d: true)))
let flowWeights = try MLX.loadArrays(url: flowURL)
s3gen.update(parameters: ModuleParameters.unflattened(remapS3Keys(flowWeights, transposeConv1d: false)))
eval(s3gen)
print("  ✅ S3Gen loaded")

let speechEmbMatrix = t3.speechEmb.weight
eval(speechEmbMatrix)

print("\n" + String(repeating: "=", count: 80))
print("GENERATING AUDIO")
print(String(repeating: "=", count: 80))

for (index, text) in texts.enumerated() {
    print("\n[\(index + 1)/\(texts.count)] \"\(text)\"")

    // Tokenize
    let normalized = normalizeTextForTokenizer(text)
    var tokens = tokenize(normalized, vocab: vocab, merges: merges, languageId: "en")
    if let sotToken = vocab["<|startoftranscript|>"] { tokens.insert(sotToken, at: 0) }
    if let eotToken = vocab["<|endoftranscript|>"] { tokens.append(eotToken) }
    let textTokens = MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0)
    print("  Tokens: \(tokens.count)")

    // T3 Generation
    print("  Generating speech tokens...")
    let speechTokensRaw = t3.generate(
        textTokens: textTokens,
        speakerEmb: speakerEmb,
        condTokens: condTokens,
        maxTokens: 800,
        temperature: 0.8,
        emotionValue: emotionAdv[0, 0].item(Float.self),
        cfgWeight: 0.5,
        repetitionPenalty: 1.05,
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

    let outputPath = outputDir.appendingPathComponent("swift_audio_\(index + 1).wav")
    try writeWAV(audio: audioSamples, sampleRate: 24000, to: outputPath)
    print("  ✅ Saved: \(outputPath.path)")
}

print("\n" + String(repeating: "=", count: 80))
print("✅ AUDIO GENERATION COMPLETE!")
print("   Output directory: \(outputDir.path)")
print(String(repeating: "=", count: 80))
