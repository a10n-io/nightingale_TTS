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
print("  DEBUG: s3PromptToken shape: \(s3PromptToken.shape)")
print("  DEBUG: s3PromptFeat shape: \(s3PromptFeat.shape)")

// Load S3Gen (reuse loading logic from VerifyLive)
// Load FULL S3Gen weights (encoder + decoder + embeddings)
// NOTE: Using *_fixed.safetensors with all Conv1d in consistent MLX format [out, kernel, in]
let flowURL = modelDir.appendingPathComponent("mlx/s3gen_fp16_fixed.safetensors")
let vocoderURL = modelDir.appendingPathComponent("mlx/vocoder_weights_fixed_v2.safetensors")

func remapS3Key(_ key: String) -> String? {
    var k = key

    // Strip "s3gen.flow." prefix if present
    if k.hasPrefix("s3gen.flow.") {
        k = k.replacingOccurrences(of: "s3gen.flow.", with: "")
    } else if k.hasPrefix("flow.") {
        k = k.replacingOccurrences(of: "flow.", with: "")
    }

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
        // CRITICAL: Strip "estimator." - Python has decoder.estimator.*, Swift has decoder.*
        k = k.replacingOccurrences(of: "decoder.estimator.", with: "decoder.")

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
        // CRITICAL: Convert final_proj and final_block to camelCase
        k = k.replacingOccurrences(of: ".final_proj.", with: ".finalProj.")
        k = k.replacingOccurrences(of: ".final_block.", with: ".finalBlock.")
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
            } else if isConv1d {
                // Only auto-detect and transpose if transposeConv1d is true
                // For pre-converted weights (like vocoder), pass transposeConv1d: false
                if transposeConv1d {
                    // PyTorch Conv1d: [out, in, kernel] where in > kernel usually
                    // MLX Conv1d: [out, kernel, in] where kernel < in usually
                    // Detect format by comparing middle vs last dimension
                    let dim1 = value.shape[1]
                    let dim2 = value.shape[2]
                    let needsTranspose = dim1 > dim2  // PyTorch has larger middle dim (in_channels)
                    remapped[newKey] = needsTranspose ? value.swappedAxes(1, 2) : value
                } else {
                    // Don't transpose - weights already in correct format
                    remapped[newKey] = value
                }
            } else {
                remapped[newKey] = value
            }
        }
    }
    return remapped
}

// Load weights FIRST before initializing S3Gen
print("Loading flow weights from: \(flowURL)")
let flowWeights = try MLX.loadArrays(url: flowURL)
print("  Loaded \(flowWeights.count) flow weight tensors")

print("Loading vocoder weights from: \(vocoderURL)")
let vocoderWeights = try MLX.loadArrays(url: vocoderURL)
print("  Loaded \(vocoderWeights.count) vocoder weight tensors")

// Debug: Print mSource keys being remapped
print("  DEBUG: Checking mSource key remapping...")
for key in vocoderWeights.keys where key.contains("m_source") {
    if let remapped = remapS3Key(key), let value = vocoderWeights[key] {
        print("    '\(key)' -> '\(remapped)' (shape \(value.shape))")
    }
}

// Initialize S3Gen with PROPER weights (not empty dict)
// S3Gen.init will handle encoder weight remapping internally
print("Initializing S3Gen with flowWeights...")
var s3gen = S3Gen(flowWeights: flowWeights, vocoderWeights: nil)

// Load vocoder weights via update (vocoder doesn't need special init handling)
// CRITICAL: vocoder safetensors ALSO has Conv1d in MLX format - do NOT transpose
print("Loading vocoder weights via update()...")
s3gen.update(parameters: ModuleParameters.unflattened(remapS3Keys(vocoderWeights, transposeConv1d: false)))

// Load ONLY decoder weights via update (not encoder, which was loaded in init)
// NOTE: s3gen_fp16.safetensors already has weights in MLX format (Conv1d are [out,kernel,in])
print("Loading decoder weights via update()...")
let decoderOnlyWeights = flowWeights.filter { $0.key.contains("decoder") }
print("  DEBUG: Filtered to \(decoderOnlyWeights.count) decoder-only keys")
print("  Sample decoder keys:")
for key in Array(decoderOnlyWeights.keys.sorted().prefix(5)) {
    print("    \(key)")
}
// Check for final_proj specifically
print("  DEBUG: Checking final_proj keys...")
for key in decoderOnlyWeights.keys where key.contains("final_proj") {
    print("    Before remap: '\(key)'")
    if let remapped = remapS3Key(key) {
        print("    After remap:  '\(remapped)'")
    }
}
// CRITICAL: s3gen_fp16.safetensors ALREADY has Conv1d in MLX format [out,kernel,in]
// Do NOT transpose! The file was pre-converted for MLX.
// Decoder weights need auto-detection transpose (mix of PyTorch and pre-converted formats)
let remappedDecoderWeights = remapS3Keys(decoderOnlyWeights, transposeConv1d: true)
print("  After remapping: \(remappedDecoderWeights.count) decoder keys")
print("  Sample remapped keys:")
for key in Array(remappedDecoderWeights.keys.sorted().prefix(5)) {
    print("    \(key)")
}
// Check if finalProj keys are in remapped weights
print("  DEBUG: Checking for finalProj in remapped weights...")
for key in remappedDecoderWeights.keys where key.contains("finalProj") {
    print("    Found: '\(key)'")
}
// DEBUG: Check a decoder weight BEFORE update
if let firstWeight = s3gen.decoder.downBlocks[0].resnet.block1.norm.weight {
    eval(firstWeight)
    let beforeMean = firstWeight.mean().item(Float.self)
    let beforeMin = firstWeight.min().item(Float.self)
    let beforeMax = firstWeight.max().item(Float.self)
    print("  DEBUG: Decoder downBlocks[0].resnet.block1.norm.weight BEFORE update:")
    print("    mean=\(beforeMean), range=[\(beforeMin), \(beforeMax)]")
}

// DEBUG: Check if Conv1d weight was transposed
if let convWeight = remappedDecoderWeights["decoder.estimator.downBlocks.0.resnet.block1.conv.conv.weight"] {
    eval(convWeight)
    print("  DEBUG: Conv1d weight shape AFTER remapping: \(convWeight.shape)")
    print("    Expected [256, 3, 320] if transposed correctly")
    print("    Original PyTorch format: [256, 320, 3]")
}

// Strip "decoder.estimator." prefix and call update() directly on decoder module
// Python has decoder.estimator.downBlocks but Swift has decoder.downBlocks
print("  Stripping 'decoder.estimator.' prefix and loading directly to decoder...")
var decoderWeightsNoPrefix: [String: MLXArray] = [:]
for (key, value) in remappedDecoderWeights {
    if key.hasPrefix("decoder.estimator.") {
        let keyWithoutPrefix = String(key.dropFirst("decoder.estimator.".count))
        decoderWeightsNoPrefix[keyWithoutPrefix] = value
    } else if key.hasPrefix("decoder.") {
        // Some keys might not have "estimator" (e.g., decoder.convIn, decoder.timeMLP)
        let keyWithoutPrefix = String(key.dropFirst("decoder.".count))
        decoderWeightsNoPrefix[keyWithoutPrefix] = value
    }
}

// DEBUG: Check the same Conv1d weight after prefix stripping
if let convWeight = decoderWeightsNoPrefix["downBlocks.0.resnet.block1.conv.conv.weight"] {
    eval(convWeight)
    print("  DEBUG: Conv1d weight shape AFTER prefix strip: \(convWeight.shape)")
}
print("  Stripped to \(decoderWeightsNoPrefix.count) keys without 'decoder.' prefix")
print("  Sample stripped keys:")
for key in Array(decoderWeightsNoPrefix.keys.sorted().prefix(5)) {
    print("    \(key)")
}

// DEBUG: Check Conv1d weight shapes in the dict before loading
for (key, value) in decoderWeightsNoPrefix {
    if key.contains("downBlocks.0") && key.contains(".conv.") && key.hasSuffix(".weight") && value.ndim == 3 {
        eval(value)
        print("  DEBUG Conv1d in dict: \(key) → shape \(value.shape)")
    }
}

// CRITICAL: Transpose ALL FixedLinear weights from PyTorch [out, in] to MLX [in, out]
// This includes:
// - timeMLP.linear_1, timeMLP.linear_2
// - ResNet mlp.1.weight (mlpLinear)
// - Attention projections (queryProj, keyProj, valueProj, outProj)
// - Feedforward layers (FlowMLP layers[0], layers[1])
// BUT NOT LayerNorm weights, embeddings, or Conv1d weights
var transposedDecoderWeights: [String: MLXArray] = [:]
for (key, value) in decoderWeightsNoPrefix {
    // Transpose ALL Linear/FixedLinear weights (2D, excluding LayerNorm and embeddings)
    // LayerNorm weights don't have "linear" or "Proj" or "mlp" or "layers" in their key
    // Embeddings would be handled separately
    let is2DWeight = key.hasSuffix(".weight") && value.ndim == 2
    let isLinearLayer = is2DWeight && (
        key.contains("timeMLP.linear") ||      // TimeMLP linear layers
        key.contains(".mlp.") ||               // ResNet MLP layers (mlpLinear)
        key.contains("Proj.weight") ||         // Attention projections (queryProj, keyProj, etc.)
        key.contains(".layers.") ||            // FlowMLP feedforward layers
        key.contains(".ff.layers.")            // Alternative FF layer naming
    )

    // Explicitly exclude LayerNorm, GroupNorm, embeddings
    let isNormLayer = key.contains(".norm") && !key.contains(".norm.") // Avoid false positives like "norm1.weight"
    let shouldTranspose = isLinearLayer && !isNormLayer

    if shouldTranspose {
        transposedDecoderWeights[key] = value.T
        print("  Transposed \(key): \(value.shape) -> \(value.T.shape)")
    } else {
        transposedDecoderWeights[key] = value
    }
}

// Load directly to decoder
s3gen.decoder.update(parameters: ModuleParameters.unflattened(transposedDecoderWeights))

// DEBUG: Check the same decoder weight AFTER update
if let firstWeight = s3gen.decoder.downBlocks[0].resnet.block1.norm.weight {
    eval(firstWeight)
    let afterMean = firstWeight.mean().item(Float.self)
    let afterMin = firstWeight.min().item(Float.self)
    let afterMax = firstWeight.max().item(Float.self)
    print("  DEBUG: Decoder downBlocks[0].resnet.block1.norm.weight AFTER update:")
    print("    mean=\(afterMean), range=[\(afterMin), \(afterMax)]")
    print("    Expected Python: mean=0.535156")
}

// Debug: Verify mSource.linear weights after loading
print("  DEBUG: mSource.linear weights after update:")
let mSourceWeight = s3gen.vocoder.mSource.linear.weight
eval(mSourceWeight)
print("    shape: \(mSourceWeight.shape)")
print("    values: \(mSourceWeight.flattened().asArray(Float.self))")
print("    Expected Python: [-0.00117903, -0.00026658, -0.00039365, ...]")

eval(s3gen)

// Enable vocoder debug output
Mel2Wav.debugEnabled = true
print("  ✅ S3Gen loaded (encoder+decoder+vocoder)")

// Verify encoder and encoderProj weights are loaded correctly
print("\n🔍 DEBUG: Weight verification:")

// Check encoder embedding
let encEmbWeight = s3gen.inputEmbedding.weight
eval(encEmbWeight)
print("  inputEmbedding.weight: shape=\(encEmbWeight.shape), range=[\(encEmbWeight.min().item(Float.self)), \(encEmbWeight.max().item(Float.self))]")

// Check encoder first layer (encoders[0])
let encFirstLayerW = s3gen.encoder.encoders[0].feedForward.w1.weight
eval(encFirstLayerW)
print("  encoder.encoders[0].feedForward.w1.weight:")
print("    Shape: \(encFirstLayerW.shape) (expected: [2048, 512])")
print("    Range: [\(encFirstLayerW.min().item(Float.self)), \(encFirstLayerW.max().item(Float.self))] (expected: ≈[-0.35, 0.39])")

// Check encoderProj
let epWeight = s3gen.encoderProj.weight
eval(epWeight)
let epMean = epWeight.mean().item(Float.self)
let epMin = epWeight.min().item(Float.self)
let epMax = epWeight.max().item(Float.self)
print("  encoderProj.weight:")
print("    Shape: \(epWeight.shape) (expected: [512, 80])")
print("    Mean: \(epMean) (expected: ≈0.000109)")
print("    Range: [\(epMin), \(epMax)] (expected: ≈[-0.131, 0.128])")
if abs(epMean) > 0.01 || abs(epMin) > 0.2 || abs(epMax) > 0.2 {
    print("    ⚠️ WARNING: encoderProj weights look wrong!")
} else {
    print("    ✅ encoderProj weights correct!")
}

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

// CROSS-TEST MODE: Use Python tokens instead of generating with Swift T3
let usePythonTokens = true  // Set to true to test Python tokens → Swift audio
let pythonTokens = [1732, 2068, 2186, 1457, 680, 1460, 3647, 5834, 5915, 5266, 5509, 5429, 3242, 569, 572, 599, 683, 719, 1448, 2087, 1976, 1946, 3890, 5192, 4814, 1277, 1736, 3650, 6077, 4780, 2163, 4752, 6377, 6373, 4771, 5014, 5015, 2021, 2020, 4448, 2671, 2112, 411, 2517, 5109, 5838, 5845, 5837, 2367, 4718, 6238, 1226, 2145, 1431, 5006, 3479, 1288, 2267, 5021, 4778, 1855, 4194, 2246, 193, 157, 1679, 2096, 4373, 4349, 6050, 1686, 1032, 2331, 5672, 586, 3990, 38, 4032, 6534, 4320, 4106, 3863, 3146, 4671, 5648, 627, 5325, 1620, 1892, 1865, 5510, 4789, 5186, 731, 734, 737, 116, 386]

let speechTokens: MLXArray
if usePythonTokens {
    print("  🔄 Using Python-generated tokens for cross-test")
    speechTokens = MLXArray(pythonTokens.map { Int32($0) }).expandedDimensions(axis: 0)
    print("  Loaded \(pythonTokens.count) Python tokens")
} else {
    // T3 Generation - match Python settings for deterministic output
    // Note: Using maxTokens=100 to match Python output (Python generated 98 tokens)
    print("  Generating speech tokens (temp=0.001, topP=1.0, repPen=2.0, maxTokens=100)...")
    let speechTokensRaw = t3.generate(
        textTokens: textTokens,
        speakerEmb: speakerEmb,
        condTokens: condTokens,
        maxTokens: 100,  // Match Python: generates ~98 tokens for this length text
        temperature: 0.001,  // Match Python's near-deterministic sampling
        emotionValue: emotionAdv[0, 0].item(Float.self),
        cfgWeight: 0.5,
        repetitionPenalty: 2.0,  // Match Python's repetition penalty
        topP: 1.0,  // Match Python (no nucleus sampling)
        minP: 0.05
    )
    var speechTokensClean = speechTokensRaw.filter { $0 != 6561 && $0 != 6562 }
    if speechTokensClean.isEmpty { speechTokensClean = [1] }
    speechTokens = MLXArray(speechTokensClean.map { Int32($0) }).expandedDimensions(axis: 0)
    print("  Generated \(speechTokensClean.count) speech tokens")
}
eval(speechTokens)

// Save tokens for cross-testing with Python (only when using Swift tokens)
let tokensSaveURL = URL(fileURLWithPath: "\(PROJECT_ROOT)/E2E/swift_generated_tokens.safetensors")
try MLX.save(arrays: ["tokens": speechTokens], url: tokensSaveURL)
print("  ✅ Saved Swift tokens to E2E/swift_generated_tokens.safetensors")

// S3Gen - SKIP VOCODER, JUST GET MEL
print("  Running S3Gen (just encoder+ODE, no vocoder)...")
let (encOut, mel) = s3gen.getEncoderAndFlowOutput(
    tokens: speechTokens,
    speakerEmb: s3Embedding,
    speechEmbMatrix: speechEmbMatrix,
    promptToken: s3PromptToken,
    promptFeat: s3PromptFeat
)
eval(mel)

// Save mel for Python comparison
let melSaveURL = URL(fileURLWithPath: "\(PROJECT_ROOT)/E2E/swift_generated_mel_raw.safetensors")
try MLX.save(arrays: ["mel": mel], url: melSaveURL)
print("  Saved mel to E2E/swift_generated_mel_raw.safetensors")

// Now run full pipeline for audio
print("  Running full S3Gen with vocoder...")
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
let outputFilename = usePythonTokens ? "python_tokens_swift_audio.wav" : "swift_output.wav"
let outputPath = URL(fileURLWithPath: "\(PROJECT_ROOT)/test_audio/\(outputFilename)")
try writeWAV(audio: audioSamples, sampleRate: 24000, to: outputPath)
print("  ✅ Saved: \(outputPath.path)")

print("\n" + String(repeating: "=", count: 80))
print("✅ AUDIO GENERATION COMPLETE!")
print("   Output: \(outputPath.path)")
print(String(repeating: "=", count: 80))
