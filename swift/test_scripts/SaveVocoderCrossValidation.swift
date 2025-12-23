import Foundation
import MLX
import Nightingale

@main
struct SaveVocoderCrossValidation {
    static func main() async throws {
        print("=" + String(repeating: "=", count: 79))
        print("VOCODER CROSS-VALIDATION: Swift Vocoder on Python Decoder Mel")
        print("=" + String(repeating: "=", count: 79))

        print("\nLoading ChatterboxEngine...")
        let modelsPath = URL(fileURLWithPath: "../models/chatterbox")
        let engine = ChatterboxEngine()
        try await engine.loadModels(modelsURL: modelsPath)

        guard let s3gen = await engine.s3gen else {
            fatalError("S3Gen not loaded!")
        }

        // 🔍 DEBUG: Check if mSource.linear weights loaded or are still random
        print("\n" + String(repeating: "=", count: 80))
        print("🔍 DEBUGGING: mSource.linear Weight Check")
        print(String(repeating: "=", count: 80))

        let sourceLinearWeight = s3gen.vocoder.mSource.linear.weight
        eval(sourceLinearWeight)

        let weightFlat = sourceLinearWeight.flattened()
        let absWeights = abs(weightFlat)
        let meanAbs = absWeights.mean().item(Float.self)
        let stdWeight = weightFlat.variance().sqrt().item(Float.self)

        print("mSource.linear.weight:")
        print("  Shape: \(sourceLinearWeight.shape)")
        print("  Mean (absolute): \(meanAbs)")
        print("  Std: \(stdWeight)")
        print("  First 5 weights: \(Array(weightFlat[0..<min(5, weightFlat.shape[0])].asArray(Float.self)))")

        if let bias = s3gen.vocoder.mSource.linear.bias {
            eval(bias)
            print("mSource.linear.bias:")
            print("  Shape: \(bias.shape)")
            print("  Values: \(Array(bias.asArray(Float.self)))")
        } else {
            print("mSource.linear.bias: None")
        }

        print("\n⚠️  Expected (if weights loaded correctly):")
        print("   - Mean abs should be small (< 0.1) for trained weights")
        print("   - If mean ≈ 0.3-0.6, weights are likely still RANDOM!")
        print(String(repeating: "=", count: 80))

        print("\nLoading PYTHON decoder mel output...")
        // Load the decoder mel from Python
        let melPath = "../test_audio/cross_validate/python_decoder_mel_for_swift_vocoder.safetensors"
        let melArrays = try! MLX.loadArrays(url: URL(fileURLWithPath: melPath))
        var pythonDecoderMel = melArrays["mel"]!  // [80, T]

        print("Python decoder mel:")
        print("  Shape: \(pythonDecoderMel.shape)")
        print("  Mean: \(pythonDecoderMel.mean().item(Float.self))")
        print("  Std: \(pythonDecoderMel.variance().sqrt().item(Float.self))")
        print("  Range: [\(pythonDecoderMel.min().item(Float.self)), \(pythonDecoderMel.max().item(Float.self))]")

        // CRITICAL: Match Python's input format!
        // Python does: mel[80,T] -> unsqueeze(0) -> [1,80,T] -> transpose(1,2) -> [1,T,80]
        // So Python vocoder receives [1, T, 80]
        // Swift vocoder also expects [B, C, T] = [1, 80, T], but let's match Python's actual call
        pythonDecoderMel = pythonDecoderMel.transposed(1, 0).expandedDimensions(axis: 0)  // [80, T] -> [T, 80] -> [1, T, 80]
        eval(pythonDecoderMel)

        print("  After transpose to match Python: \(pythonDecoderMel.shape) - should be [1, T, 80]")

        // Now transpose back to [1, 80, T] for Swift vocoder which expects [B, C, T]
        pythonDecoderMel = pythonDecoderMel.transposed(0, 2, 1)  // [1, T, 80] -> [1, 80, T]
        eval(pythonDecoderMel)

        print("\nRunning SWIFT vocoder on Python decoder mel...")
        let audio = s3gen.vocoder(pythonDecoderMel)
        eval(audio)

        print("  Audio shape after vocoder: \(audio.shape)")

        // Flatten audio to 1D
        let audioFlat: MLXArray
        if audio.ndim == 3 {
            audioFlat = audio.squeezed(axis: 0).squeezed(axis: 0)
        } else if audio.ndim == 2 {
            audioFlat = audio.squeezed(axis: 0)
        } else {
            audioFlat = audio
        }
        eval(audioFlat)

        print("\nSwift vocoder audio output (from Python mel):")
        print("  Shape: \(audioFlat.shape)")
        print("  Mean: \(audioFlat.mean().item(Float.self))")
        print("  Std: \(audioFlat.variance().sqrt().item(Float.self))")
        print("  Range: [\(audioFlat.min().item(Float.self)), \(audioFlat.max().item(Float.self))]")
        print("  Sample rate: 24000 Hz")
        print("  Duration: \(Float(audioFlat.shape[0]) / 24000.0) seconds")

        // Save
        let outputDir = "../test_audio/forensic"
        try! MLX.save(
            arrays: ["audio": audioFlat],
            url: URL(fileURLWithPath: outputDir + "/python_mel_swift_vocoder.safetensors")
        )

        print("\n✅ Saved to: \(outputDir)/python_mel_swift_vocoder.safetensors")
        print("\nNext: Compare with python_mel_python_vocoder.safetensors to verify vocoder parity")
    }
}
