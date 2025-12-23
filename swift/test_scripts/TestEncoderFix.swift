import Foundation
import MLX
import Nightingale

@main
struct TestEncoderFix {
    static func main() async throws {
        print(String(repeating: "=", count: 80))
        print("TEST ENCODER FIX - Verify key remapping works")
        print(String(repeating: "=", count: 80))

        // Load model through ChatterboxEngine (production flow)
        print("\n1. Loading ChatterboxEngine...")
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

        print("\n2. Running encoder through production flow...")

        guard let s3gen = await engine.s3gen else {
            fatalError("S3Gen not loaded!")
        }

        // Concatenate tokens (same as production)
        let fullTokens = concatenated([promptToken.squeezed(axis: 0), speechTokens], axis: 0)

        // Get embeddings
        let tokenEmbs = take(s3gen.inputEmbedding.weight, fullTokens.asType(.int32), axis: 0)
        eval(tokenEmbs)

        print("\n📊 Input embeddings (token lookups):")
        print("   Shape: \(tokenEmbs.shape)")
        let embMean = tokenEmbs.mean()
        let embStd = tokenEmbs.variance().sqrt()
        eval(embMean, embStd)
        print("   Mean: \(embMean.item(Float.self))")
        print("   Std:  \(embStd.item(Float.self))")

        // Add batch dimension
        let x = tokenEmbs.expandedDimensions(axis: 0)

        // Run encoder
        let encoderOutput = s3gen.encoder(x)
        eval(encoderOutput)

        print("\n📊 Swift Encoder output (BEFORE encoder_proj):")
        print("   Shape: \(encoderOutput.shape)")

        let encMean = encoderOutput.mean()
        let encStd = encoderOutput.variance().sqrt()
        let encMin = encoderOutput.min()
        let encMax = encoderOutput.max()
        eval(encMean, encStd, encMin, encMax)

        print("   Mean: \(encMean.item(Float.self))")
        print("   Std:  \(encStd.item(Float.self))")
        print("   Range: [\(encMin.item(Float.self)), \(encMax.item(Float.self))]")

        // Run through encoder_proj
        let encoderProjOutput = s3gen.encoderProj(encoderOutput)
        eval(encoderProjOutput)

        print("\n📊 Swift Encoder output (AFTER encoder_proj):")
        print("   Shape: \(encoderProjOutput.shape)")

        let projMean = encoderProjOutput.mean()
        let projStd = encoderProjOutput.variance().sqrt()
        let projMin = encoderProjOutput.min()
        let projMax = encoderProjOutput.max()
        eval(projMean, projStd, projMin, projMax)

        print("   Mean: \(projMean.item(Float.self))")
        print("   Std:  \(projStd.item(Float.self))")
        print("   Range: [\(projMin.item(Float.self)), \(projMax.item(Float.self))]")

        // Save for comparison
        let forensicDir = "../test_audio/forensic"
        try? FileManager.default.createDirectory(atPath: forensicDir, withIntermediateDirectories: true)

        try! MLX.save(
            arrays: [
                "input_embeddings": tokenEmbs,
                "encoder_output_before_proj": encoderOutput,
                "encoder_output_after_proj": encoderProjOutput,
            ],
            url: URL(fileURLWithPath: forensicDir + "/swift_encoder_fixed.safetensors")
        )

        print("\n✅ Saved to: \(forensicDir)/swift_encoder_fixed.safetensors")

        print("\n" + String(repeating: "=", count: 80))
        print("EXPECTED VALUES (from Python):")
        print(String(repeating: "=", count: 80))
        print("\nEncoder BEFORE encoder_proj:")
        print("  mean=-0.007, std=0.311")
        print("\nEncoder AFTER encoder_proj:")
        print("  mean=-0.007, std=0.435")

        print("\n" + String(repeating: "=", count: 80))
        print("DIAGNOSIS:")
        print(String(repeating: "=", count: 80))

        let stdRatioBefore = encStd.item(Float.self) / 0.311
        let stdRatioAfter = projStd.item(Float.self) / 0.435

        print("\nStd ratio BEFORE encoder_proj: \(stdRatioBefore)")
        if abs(stdRatioBefore - 1.0) < 0.15 {
            print("✅ Encoder variance matches! (ratio ~1.0)")
        } else {
            print("❌ Encoder variance mismatch!")
        }

        print("\nStd ratio AFTER encoder_proj: \(stdRatioAfter)")
        if abs(stdRatioAfter - 1.0) < 0.15 {
            print("✅ Encoder_proj variance matches! (ratio ~1.0)")
            print("🎉 FIX VERIFIED - Key remapping is working!")
        } else {
            print("❌ Encoder_proj variance mismatch!")
        }

        print("\n" + String(repeating: "=", count: 80))
    }
}
