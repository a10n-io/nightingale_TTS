import Foundation
import MLX
import MLXRandom
import Nightingale

@main
struct SaveDecoderIntermediates {
    static func main() async throws {
        print("Loading ChatterboxEngine...")
        let modelsPath = URL(fileURLWithPath: "models/chatterbox")
        let engine = ChatterboxEngine()
        try await engine.loadModels(modelsURL: modelsPath)

        guard let s3gen = await engine.s3gen else {
            fatalError("S3Gen not loaded!")
        }

        // Load baked voice
        let voiceDir = "baked_voices/samantha"
        let voiceArrays = try! MLX.loadArrays(url: URL(fileURLWithPath: voiceDir + "/baked_voice.safetensors"))
        let speakerEmb = voiceArrays["t3.speaker_emb"]!
        let speechEmbMatrix = voiceArrays["gen.embedding"]!
        let promptToken = voiceArrays["gen.prompt_token"]!
        let promptFeat = voiceArrays["gen.prompt_feat"]!

        // Load Python tokens
        let tokensPath = "test_audio/cross_validate/python_speech_tokens.safetensors"
        let tokensArrays = try! MLX.loadArrays(url: URL(fileURLWithPath: tokensPath))
        let generatedTokens = tokensArrays["speech_tokens"]!

        print("\n" + String(repeating: "=", count: 80))
        print("TRACING SWIFT DECODER INTERMEDIATE OUTPUTS")
        print(String(repeating: "=", count: 80))

        // Manually trace through decoder to save intermediates
        // 1. Token embedding + encoder
        let fullTokens = concatenated([promptToken.squeezed(axis: 0), generatedTokens], axis: 0)
        let tokenEmb = take(s3gen.inputEmbedding.weight, fullTokens.asType(.int32), axis: 0)
        let x = tokenEmb.expandedDimensions(axis: 0)

        let h = s3gen.encoder(x)
        let mu = s3gen.encoderProj(h)
        eval(mu)

        print("\n1. Encoder output (mu):")
        print("   Shape: \(mu.shape)")
        print("   Mean: \(mu.mean().item(Float.self))")
        print("   Std: \(mu.variance().sqrt().item(Float.self))")
        print("   Range: [\(mu.min().item(Float.self)), \(mu.max().item(Float.self))]")

        // Save mu
        try! MLX.save(
            arrays: ["mu": mu.squeezed(axis: 0)],
            url: URL(fileURLWithPath: "test_audio/forensic/swift_decoder_mu.safetensors")
        )

        // 2. Speaker embedding projection
        var speechEmb = speechEmbMatrix
        let norm = sqrt(sum(speechEmb * speechEmb, axis: 1, keepDims: true)) + 1e-8
        speechEmb = speechEmb / norm
        let spk = matmul(speechEmb, s3gen.spkEmbedAffine.weight) + s3gen.spkEmbedAffine.bias!
        eval(spk)

        print("\n2. Speaker embedding projection (spk):")
        print("   Input (speechEmbMatrix): \(speechEmbMatrix.shape), mean=\(speechEmbMatrix.mean().item(Float.self))")
        print("   After normalization: mean=\(speechEmb.mean().item(Float.self))")
        print("   After projection (spk): \(spk.shape)")
        print("   Mean: \(spk.mean().item(Float.self))")
        print("   Std: \(spk.variance().sqrt().item(Float.self))")
        let spkFirst5 = spk[0, 0..<5]
        eval(spkFirst5)
        print("   First 5 values: [\(spkFirst5[0].item(Float.self)), \(spkFirst5[1].item(Float.self)), \(spkFirst5[2].item(Float.self)), \(spkFirst5[3].item(Float.self)), \(spkFirst5[4].item(Float.self))]")

        try! MLX.save(
            arrays: ["spk": spk],
            url: URL(fileURLWithPath: "test_audio/forensic/swift_decoder_spk.safetensors")
        )

        // 3. Decoder conditioning (conds)
        let promptLen = promptFeat.shape[1]
        let L_total = mu.shape[1]
        let zeros = MLXArray.zeros([1, L_total - promptLen, 80], dtype: mu.dtype)
        let conds = concatenated([promptFeat, zeros], axis: 1)
        eval(conds)

        print("\n3. Decoder conditioning (conds):")
        print("   Shape: \(conds.shape)")
        let promptRegion = conds[0, 0..<promptLen, 0...]
        let generatedRegion = conds[0, promptLen..., 0...]
        eval(promptRegion, generatedRegion)
        print("   Prompt region mean: \(promptRegion.mean().item(Float.self))")
        print("   Generated region (zeros) mean: \(generatedRegion.mean().item(Float.self))")

        // 4. Initial noise - USE THE ACTUAL NOISE FROM S3GEN (contains Python noise!)
        let noise = s3gen.fixedNoise[0..., 0..., 0..<L_total]
        eval(noise)

        print("\n4. Initial noise (from s3gen.fixedNoise):")
        print("   Shape: \(noise.shape)")
        print("   Mean: \(noise.mean().item(Float.self))")
        print("   Std: \(noise.variance().sqrt().item(Float.self))")
        let noiseFirst5 = noise[0, 0, 0..<5]
        eval(noiseFirst5)
        print("   First 5 values [0,0,:5]: [\(noiseFirst5[0].item(Float.self)), \(noiseFirst5[1].item(Float.self)), \(noiseFirst5[2].item(Float.self)), \(noiseFirst5[3].item(Float.self)), \(noiseFirst5[4].item(Float.self))]")

        try! MLX.save(
            arrays: ["noise": noise],
            url: URL(fileURLWithPath: "test_audio/forensic/swift_decoder_noise.safetensors")
        )

        // 5. Run full decoder
        let (_, mel) = s3gen.getEncoderAndFlowOutput(
            tokens: generatedTokens.expandedDimensions(axis: 0),
            speakerEmb: speakerEmb,
            speechEmbMatrix: speechEmbMatrix,
            promptToken: promptToken,
            promptFeat: promptFeat
        )

        // Extract generated portion
        let generatedMel = mel[0..., 0..., promptLen...]
        eval(generatedMel)

        print("\n5. Final decoder output:")
        print("   Shape: \(generatedMel.shape)")
        print("   Mean: \(generatedMel.mean().item(Float.self))")
        print("   Std: \(generatedMel.variance().sqrt().item(Float.self))")

        print("\n✅ Saved intermediate outputs to forensic/")
    }
}
