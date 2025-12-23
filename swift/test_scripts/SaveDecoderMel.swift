import Foundation
import MLX
import Nightingale

@main
struct SaveDecoderMel {
    static func main() async throws {
        print("Loading ChatterboxEngine...")
        let modelsPath = URL(fileURLWithPath: "../models/chatterbox")
        let engine = ChatterboxEngine()
        try await engine.loadModels(modelsURL: modelsPath)

        guard let s3gen = await engine.s3gen else {
            fatalError("S3Gen not loaded!")
        }

        // Load baked voice
        let voiceDir = "../baked_voices/samantha"
        let voiceArrays = try! MLX.loadArrays(url: URL(fileURLWithPath: voiceDir + "/baked_voice.safetensors"))
        let speakerEmb = voiceArrays["t3.speaker_emb"]!
        let speechEmbMatrix = voiceArrays["gen.embedding"]!
        let promptToken = voiceArrays["gen.prompt_token"]!
        let promptFeat = voiceArrays["gen.prompt_feat"]!

        // Load Python tokens
        let tokensPath = "../test_audio/cross_validate/python_speech_tokens.safetensors"
        let tokensArrays = try! MLX.loadArrays(url: URL(fileURLWithPath: tokensPath))
        let generatedTokens = tokensArrays["speech_tokens"]!

        print("\nInputs:")
        print("  speakerEmb: \(speakerEmb.shape)")
        print("  speechEmbMatrix: \(speechEmbMatrix.shape)")
        print("  promptToken: \(promptToken.shape)")
        print("  promptFeat: \(promptFeat.shape)")
        print("  generatedTokens: \(generatedTokens.shape)")

        print("\nRunning full pipeline (encoder + decoder)...")
        // Get encoder output (mu) and decoder output (mel) separately
        // This matches Python's flow.inference() which returns mel before vocoder
        let (_, mel) = s3gen.getEncoderAndFlowOutput(
            tokens: generatedTokens.expandedDimensions(axis: 0),
            speakerEmb: speakerEmb,
            speechEmbMatrix: speechEmbMatrix,
            promptToken: promptToken,
            promptFeat: promptFeat
        )

        eval(mel)

        print("\nSwift decoder mel output (full including prompt):")
        print("  Shape: \(mel.shape)")
        print("  Mean: \(mel.mean().item(Float.self))")
        print("  Std: \(mel.variance().sqrt().item(Float.self))")
        print("  Range: [\(mel.min().item(Float.self)), \(mel.max().item(Float.self))]")

        // Extract generated portion (without prompt) to match Python
        let promptLen = promptFeat.shape[1]  // 500
        let generatedMel = mel[0..., 0..., promptLen...]
        eval(generatedMel)

        print("\nGenerated portion only (without prompt):")
        print("  Shape: \(generatedMel.shape)")
        print("  Mean: \(generatedMel.mean().item(Float.self))")

        // Save in [C, T] format to match Python (remove batch dim)
        let outputDir = "../test_audio/forensic"
        try? FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

        try! MLX.save(
            arrays: ["mel": generatedMel.squeezed(axis: 0)],  // Remove batch dim: [1, 80, T] -> [80, T]
            url: URL(fileURLWithPath: outputDir + "/swift_decoder_mel.safetensors")
        )

        print("\n✅ Saved to: \(outputDir)/swift_decoder_mel.safetensors")
    }
}
