import Foundation
import MLX
import MLXNN
import MLXRandom
import Nightingale
import AVFoundation

// MARK: - Key Remapping (copied from ChatterboxEngine)

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

func remapS3Keys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var remapped: [String: MLXArray] = [:]
    for (key, value) in weights {
        if let newKey = remapS3Key(key) {
            remapped[newKey] = value
        }
    }
    return remapped
}

func remapS3Key(_ key: String) -> String? {
    var k = key

    // Map root components from flow.* (handle both with and without s3gen. prefix)
    if k.hasPrefix("s3gen.flow.input_embedding.") {
        return k.replacingOccurrences(of: "s3gen.flow.input_embedding.", with: "inputEmbedding.")
    }
    if k.hasPrefix("flow.input_embedding.") {
        return k.replacingOccurrences(of: "flow.input_embedding.", with: "inputEmbedding.")
    }
    if k.hasPrefix("s3gen.flow.spk_embed_affine_layer.") {
        return k.replacingOccurrences(of: "s3gen.flow.spk_embed_affine_layer.", with: "spkEmbedAffine.")
    }
    if k.hasPrefix("flow.spk_embed_affine_layer.") {
        return k.replacingOccurrences(of: "flow.spk_embed_affine_layer.", with: "spkEmbedAffine.")
    }
    if k.hasPrefix("s3gen.flow.encoder_proj.") {
        return k.replacingOccurrences(of: "s3gen.flow.encoder_proj.", with: "encoderProj.")
    }
    if k.hasPrefix("flow.encoder_proj.") {
        return k.replacingOccurrences(of: "flow.encoder_proj.", with: "encoderProj.")
    }

    // Map Encoder
    var isEncoderKey = false
    if k.hasPrefix("s3gen.flow.encoder.") {
        k = k.replacingOccurrences(of: "s3gen.flow.encoder.", with: "encoder.")
        isEncoderKey = true
    } else if k.hasPrefix("flow.encoder.") {
        k = k.replacingOccurrences(of: "flow.encoder.", with: "encoder.")
        isEncoderKey = true
    }

    if isEncoderKey {
        k = k.replacingOccurrences(of: "pre_lookahead_layer", with: "preLookaheadLayer")
        k = k.replacingOccurrences(of: "up_layer", with: "upLayer")
        for i in 0..<6 { k = k.replacingOccurrences(of: "encoders_\(i).", with: "encoders.\(i).") }
        for i in 0..<4 { k = k.replacingOccurrences(of: "up_encoders.\(i).", with: "upEncoders.\(i).") }
        k = k.replacingOccurrences(of: "after_norm", with: "afterNorm")
    }

    // Remap mel2wav.* -> vocoder.*
    if k.hasPrefix("mel2wav.") {
        k = k.replacingOccurrences(of: "mel2wav.", with: "vocoder.")
        k = k.replacingOccurrences(of: "conv_pre", with: "convPre")
        k = k.replacingOccurrences(of: "conv_post", with: "convPost")
    }

    // F0 Predictor Mapping
    if k.contains("f0_predictor.") {
        k = k.replacingOccurrences(of: "f0_predictor.condnet.0.", with: "vocoder.f0Predictor.convs.0.")
        k = k.replacingOccurrences(of: "f0_predictor.condnet.1.", with: "vocoder.f0Predictor.convs.1.")
        k = k.replacingOccurrences(of: "f0_predictor.condnet.2.", with: "vocoder.f0Predictor.convs.2.")
        k = k.replacingOccurrences(of: "f0_predictor.condnet.3.", with: "vocoder.f0Predictor.convs.3.")
        k = k.replacingOccurrences(of: "f0_predictor.condnet.4.", with: "vocoder.f0Predictor.convs.4.")
        k = k.replacingOccurrences(of: "f0_predictor.classifier.", with: "vocoder.f0Predictor.classifier.")
        return k
    }

    // Source Module Mapping
    if k.contains("m_source.") {
        k = k.replacingOccurrences(of: "m_source.l_linear.", with: "vocoder.mSource.linear.")
        return k
    }

    // Source Downs
    if k.contains("source_downs.") {
        k = k.replacingOccurrences(of: "source_downs.", with: "vocoder.sourceDowns.")
        return k
    }

    // Source ResBlocks
    if k.contains("source_resblocks.") {
        k = k.replacingOccurrences(of: "source_resblocks.", with: "vocoder.sourceResBlocks.")
        k = k.replacingOccurrences(of: "activations1", with: "acts1")
        k = k.replacingOccurrences(of: "activations2", with: "acts2")
        return k
    }

    // Handle direct vocoder weights
    if k.hasPrefix("conv_pre") || k.hasPrefix("conv_post") || k.hasPrefix("resblocks") || k.hasPrefix("ups") {
        k = "vocoder." + k
        k = k.replacingOccurrences(of: "conv_pre", with: "convPre")
        k = k.replacingOccurrences(of: "conv_post", with: "convPost")
        k = k.replacingOccurrences(of: "activations1", with: "acts1")
        k = k.replacingOccurrences(of: "activations2", with: "acts2")
    }

    // Transform flow.decoder.estimator.* -> decoder.*
    if k.hasPrefix("s3gen.flow.decoder.estimator.") {
        k = k.replacingOccurrences(of: "s3gen.flow.decoder.estimator.", with: "decoder.")
    } else if k.hasPrefix("flow.decoder.estimator.") {
        k = k.replacingOccurrences(of: "flow.decoder.estimator.", with: "decoder.")
    }

    if k.contains("rand_noise") { return nil }

    // Block names
    k = k.replacingOccurrences(of: "down_blocks_", with: "downBlocks.")
    k = k.replacingOccurrences(of: "mid_blocks_", with: "midBlocks.")
    k = k.replacingOccurrences(of: "up_blocks_", with: "upBlocks.")

    k = k.replacingOccurrences(of: "mlp_linear", with: "mlpLinear")
    k = k.replacingOccurrences(of: "res_conv", with: "resConv")

    // Transform transformer components
    k = k.replacingOccurrences(of: ".transformer_", with: ".transformers.")
    k = k.replacingOccurrences(of: ".attn.", with: ".attention.")
    k = k.replacingOccurrences(of: "query_proj", with: "queryProj")
    k = k.replacingOccurrences(of: "key_proj", with: "keyProj")
    k = k.replacingOccurrences(of: "value_proj", with: "valueProj")
    k = k.replacingOccurrences(of: "out_proj", with: "outProj")

    // Conformer attention
    k = k.replacingOccurrences(of: "linear_q", with: "queryProj")
    k = k.replacingOccurrences(of: "linear_k", with: "keyProj")
    k = k.replacingOccurrences(of: "linear_v", with: "valueProj")
    k = k.replacingOccurrences(of: "linear_out", with: "outProj")

    if k.contains(".norm3.") {
        k = k.replacingOccurrences(of: ".norm3.", with: ".norm2.")
    }
    k = k.replacingOccurrences(of: "ff.net.0.", with: "ff.layers.0.")
    k = k.replacingOccurrences(of: "ff.net.2.", with: "ff.layers.1.")

    k = k.replacingOccurrences(of: "time_mlp", with: "timeMLP")
    k = k.replacingOccurrences(of: "timeMLP.0.", with: "timeMLP.linear1.")
    k = k.replacingOccurrences(of: "timeMLP.2.", with: "timeMLP.linear2.")
    k = k.replacingOccurrences(of: ".linear_1.", with: ".linear1.")
    k = k.replacingOccurrences(of: ".linear_2.", with: ".linear2.")

    k = k.replacingOccurrences(of: "downsample", with: "downLayer")
    k = k.replacingOccurrences(of: "upsample", with: "upLayer")
    k = k.replacingOccurrences(of: "final_block", with: "finalBlock")
    k = k.replacingOccurrences(of: "final_proj", with: "finalProj")
    k = k.replacingOccurrences(of: "act_post", with: "actPost")

    return k
}

// MARK: - Main

print(String(repeating: "=", count: 80))
print("NIGHTINGALE - TEST S3GEN VOCODING")
print(String(repeating: "=", count: 80))
print()

// Paths
let modelsPath = "/Users/a10n/Projects/nightingale/models"
let voicePath = "/Users/a10n/Projects/nightingale/baked_voices/samantha_full"
let tokensPath = "/Users/a10n/Projects/nightingale/test_audio/swift_speech_tokens.npy"

print("Models: \(modelsPath)")
print("Voice: \(voicePath)")
print("Speech tokens: \(tokensPath)")
print()

do {
    // Load S3Gen model
    print("Loading S3Gen model...")
    let modelsURL = URL(fileURLWithPath: modelsPath)
    let s3genFP16URL = modelsURL.appendingPathComponent("s3gen_fp16.safetensors")
    let flowWeightsURL = modelsURL.appendingPathComponent("python_flow_weights.safetensors")
    let vocoderURL = modelsURL.appendingPathComponent("vocoder_weights_python.safetensors")

    print("Loading S3Gen FP16 weights...")
    let s3genWeights = try MLX.loadArrays(url: s3genFP16URL)
    print("✅ S3Gen weights loaded (\(s3genWeights.count) arrays)")

    // Load vocoder weights (CRITICAL - without this we get single tone!)
    var vocoderWeights: [String: MLXArray]? = nil
    if FileManager.default.fileExists(atPath: vocoderURL.path) {
        print("Loading vocoder weights...")
        vocoderWeights = try MLX.loadArrays(url: vocoderURL)
        print("✅ Vocoder weights loaded (\(vocoderWeights!.count) arrays)")
    } else {
        print("⚠️  WARNING: vocoder_weights_python.safetensors not found!")
        print("   This will likely produce a single tone instead of speech!")
    }

    // Create S3Gen WITH vocoder weights
    let s3gen = S3Gen(flowWeights: s3genWeights, vocoderWeights: vocoderWeights)
    print("✅ S3Gen model created with vocoder")

    // Update with python_flow_weights
    print("Loading Python flow decoder weights...")
    let pythonFlow = try MLX.loadArrays(url: flowWeightsURL)
    print("✅ Python flow weights loaded (\(pythonFlow.count) arrays)")

    // Use proper key remapping (copied from ChatterboxEngine)
    let remapped = remapS3Keys(pythonFlow)
    print("✅ Keys remapped (\(remapped.count) arrays)")

    let params = ModuleParameters.unflattened(remapped)
    s3gen.update(parameters: params)
    print("✅ S3Gen weights updated")

    // Load fixed noise from Python (for deterministic generation)
    let fixedNoiseURL = modelsURL.appendingPathComponent("python_fixed_noise.safetensors")
    if FileManager.default.fileExists(atPath: fixedNoiseURL.path) {
        try s3gen.loadFixedNoise(from: fixedNoiseURL)
    } else {
        print("⚠️  Python fixed noise not found, using Swift RNG (may differ from Python)")
    }
    print()

    // Load T3 model (just need speechEmb.weight matrix)
    print("Loading T3 model (for speech embedding matrix)...")
    let t3URL = modelsURL.appendingPathComponent("t3_fp32.safetensors")
    let configURL = modelsURL.appendingPathComponent("t3_config.json")
    let ropeFreqsURL = modelsURL.appendingPathComponent("rope_freqs_llama3.safetensors")

    let configData = try Data(contentsOf: configURL)
    let config = try JSONDecoder().decode(T3Config.self, from: configData)

    let rawT3Weights = try MLX.loadArrays(url: t3URL)
    let t3Weights = remapT3Keys(rawT3Weights)
    let t3 = T3Model(config: config, weights: t3Weights, ropeFreqsURL: ropeFreqsURL)
    print("✅ T3 model loaded (for embedding matrix)")

    // Get speech embedding matrix
    let speechEmbMatrix = t3.speechEmb.weight
    print("✅ Speech embedding matrix: \(speechEmbMatrix.shape)")
    print()

    // Load prebaked voice S3Gen components
    print("Loading prebaked voice S3Gen components...")
    let voiceURL = URL(fileURLWithPath: voicePath)

    let soul_s3 = try NPYLoader.load(contentsOf: voiceURL.appendingPathComponent("soul_s3_192.npy"))
    print("✅ soul_s3_192.npy: \(soul_s3.shape)")

    let prompt_token = try NPYLoader.load(contentsOf: voiceURL.appendingPathComponent("prompt_token.npy"))
    print("✅ prompt_token.npy: \(prompt_token.shape)")

    let prompt_feat = try NPYLoader.load(contentsOf: voiceURL.appendingPathComponent("prompt_feat.npy"))
    print("✅ prompt_feat.npy: \(prompt_feat.shape)")
    print()

    // Load speech tokens from TestT3Generate
    print("Loading speech tokens from T3 generation...")
    let tokens_file = URL(fileURLWithPath: tokensPath)
    var tokens = try NPYLoader.load(contentsOf: tokens_file)
    print("✅ Loaded tokens: \(tokens.shape)")

    // Ensure batch dimension [N] -> [1, N]
    if tokens.ndim == 1 {
        tokens = tokens.expandedDimensions(axis: 0)
    }

    // Filter valid tokens (< 6561)
    eval(tokens)
    let tokensFlat = tokens.reshaped([-1]).asArray(Int32.self)
    let validTokens = tokensFlat.filter { $0 < 6561 }
    print("   Total tokens: \(tokensFlat.count)")
    print("   Valid tokens: \(validTokens.count)")
    print("   Invalid tokens: \(tokensFlat.count - validTokens.count)")
    print("   First 10: \(validTokens.prefix(10))")
    print("   Last 10: \(validTokens.suffix(10))")
    print()

    // Create valid token array
    let validTokenArray = MLXArray(validTokens).expandedDimensions(axis: 0)

    // Run S3Gen generation
    print("Running S3Gen generation...")
    print("Parameters:")
    print("  Tokens: \(validTokenArray.shape)")
    print("  Speaker emb: \(soul_s3.shape)")
    print("  Speech emb matrix: \(speechEmbMatrix.shape)")
    print("  Prompt token: \(prompt_token.shape)")
    print("  Prompt feat: \(prompt_feat.shape)")
    print()

    MLXRandom.seed(42)
    GPU.clearCache()

    let startTime = CFAbsoluteTimeGetCurrent()

    let audio = s3gen.generate(
        tokens: validTokenArray,
        speakerEmb: soul_s3,
        speechEmbMatrix: speechEmbMatrix,
        promptToken: prompt_token,
        promptFeat: prompt_feat
    )

    eval(audio)
    let elapsed = CFAbsoluteTimeGetCurrent() - startTime

    print()
    print(String(repeating: "=", count: 80))
    print("✅ S3GEN GENERATION COMPLETE")
    print(String(repeating: "=", count: 80))
    print()

    // Audio stats
    let audioArray = audio.asArray(Float.self)
    let duration = Float(audioArray.count) / 24000.0

    print("Audio generated:")
    print("  Samples: \(audioArray.count)")
    print("  Duration: \(String(format: "%.2f", duration))s")
    print("  Sample rate: 24000 Hz")
    print("  Time: \(String(format: "%.2f", elapsed))s")
    print("  Real-time factor: \(String(format: "%.2f", duration / Float(elapsed)))x")
    print()

    let minVal = audioArray.min() ?? 0
    let maxVal = audioArray.max() ?? 0
    let absMax = max(abs(minVal), abs(maxVal))
    print("  Range: [\(String(format: "%.4f", minVal)), \(String(format: "%.4f", maxVal))]")
    print("  Peak amplitude: \(String(format: "%.4f", absMax))")
    print()
    print("  First 10 samples: \(audioArray.prefix(10).map { String(format: "%.4f", $0) })")
    print()

    // Save audio to WAV
    let outputURL = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale/test_audio/swift_generated.wav")
    try ChatterboxEngine.saveWav(audioArray, to: outputURL)
    print()
    print("🎵 Audio saved to: \(outputURL.path)")
    print()
    print("Next step: Compare with Python-generated audio!")
    print("Run: python generate.py \"Hello world\" to compare")

} catch {
    print()
    print("❌ ERROR: \(error)")
    print()
    exit(1)
}
