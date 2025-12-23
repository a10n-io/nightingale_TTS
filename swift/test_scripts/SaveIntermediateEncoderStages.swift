import Foundation
import MLX
import Nightingale

@main
struct SaveIntermediateEncoderStages {
    static func main() async throws {
        print("Loading ChatterboxEngine...")
        let modelsPath = URL(fileURLWithPath: "../models/chatterbox")
        let engine = ChatterboxEngine()
        try await engine.loadModels(modelsURL: modelsPath)

        // Load test data
        let voiceDir = "../baked_voices/samantha"
        let voiceArrays = try! MLX.loadArrays(url: URL(fileURLWithPath: voiceDir + "/baked_voice.safetensors"))
        let tokensPath = "../test_audio/cross_validate/python_speech_tokens.safetensors"
        let tokensArrays = try! MLX.loadArrays(url: URL(fileURLWithPath: tokensPath))

        let promptToken = voiceArrays["gen.prompt_token"]!
        let speechTokens = tokensArrays["speech_tokens"]!

        guard let s3gen = await engine.s3gen else {
            fatalError("S3Gen not loaded!")
        }

        // Concatenate tokens
        let fullTokens = concatenated([promptToken.squeezed(axis: 0), speechTokens], axis: 0)

        // Get input embeddings
        let tokenEmbs = take(s3gen.inputEmbedding.weight, fullTokens.asType(.int32), axis: 0)
        eval(tokenEmbs)

        print("Input embeddings: \(tokenEmbs.shape)")
        print("  Mean: \(tokenEmbs.mean().item(Float.self))")

        // Add batch dimension
        var x = tokenEmbs.expandedDimensions(axis: 0)  // [1, 348, 512]

        // Run through embedLinear
        x = s3gen.encoder.embedLinear(x)
        eval(x)

        print("\nAfter embedLinear: \(x.shape)")
        print("  Mean: \(x.mean().item(Float.self))")
        print("  Std: \(x.variance().sqrt().item(Float.self))")
        print("  Range: [\(x.min().item(Float.self)), \(x.max().item(Float.self))]")

        // Save
        let forensicDir = "../test_audio/forensic"
        try? FileManager.default.createDirectory(atPath: forensicDir, withIntermediateDirectories: true)

        try! MLX.save(
            arrays: ["after_embedlinear": x.squeezed(axis: 0)],  // Remove batch dim
            url: URL(fileURLWithPath: forensicDir + "/swift_after_embedlinear.safetensors")
        )

        print("\n✅ Saved to: \(forensicDir)/swift_after_embedlinear.safetensors")
    }
}
