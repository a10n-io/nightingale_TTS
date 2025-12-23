import Foundation
import MLX
import Nightingale

@main
struct TraceVocoderLayers {
    static func main() async throws {
        print(String(repeating: "=", count: 80))
        print("SWIFT VOCODER LAYER-BY-LAYER TRACE")
        print(String(repeating: "=", count: 80))

        print("\nLoading ChatterboxEngine...")
        let modelsPath = URL(fileURLWithPath: "../models/chatterbox")
        let engine = ChatterboxEngine()
        try await engine.loadModels(modelsURL: modelsPath)

        guard let s3gen = await engine.s3gen else {
            fatalError("S3Gen not loaded!")
        }

        // Load Python's decoder mel (same input used for Python)
        print("\nLoading Python decoder mel...")
        let melPath = "../test_audio/cross_validate/python_decoder_mel_for_swift_vocoder.safetensors"
        let melArrays = try! MLX.loadArrays(url: URL(fileURLWithPath: melPath))
        let pythonDecoderMel = melArrays["mel"]!  // [80, T]

        print("Input mel: \(pythonDecoderMel.shape) = [80, T]")

        // Access vocoder
        let vocoder = s3gen.vocoder

        // Enable debugging to see all intermediate outputs
        Mel2Wav.debugEnabled = true

        // Prepare input: [80, T] -> [1, 80, T] for vocoder
        let mel = pythonDecoderMel.expandedDimensions(axis: 0)  // [1, 80, T]
        eval(mel)
        print("Vocoder input: \(mel.shape) = [B, C, T]")

        // Save input
        let outputDir = "../test_audio/vocoder_trace"
        try? FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

        try! MLX.save(
            arrays: ["mel": mel],
            url: URL(fileURLWithPath: outputDir + "/swift_0_input.safetensors")
        )

        // Now manually trace through the vocoder (copy logic from Mel2Wav.callAsFunction)
        print("\n" + String(repeating: "=", count: 80))
        print("TRACING VOCODER LAYERS")
        print(String(repeating: "=", count: 80))

        // Step 0: Transpose mel [B, 80, T] -> [B, T, 80]
        var x = mel.transposed(0, 2, 1)
        eval(x)
        print("\nStep 0: Transposed to MLX format: \(x.shape) = [B, T, C]")
        try! MLX.save(
            arrays: ["x": x],
            url: URL(fileURLWithPath: outputDir + "/swift_0_transposed.safetensors")
        )

        // Step 1: F0 Prediction
        let f0 = vocoder.f0Predictor(x)  // [B, T]
        eval(f0)
        print("\nStep 1: F0 prediction: \(f0.shape)")
        print("  F0 range: [\(f0.min().item(Float.self)), \(f0.max().item(Float.self))]")
        print("  F0 first 10: \(Array(f0[0, 0..<10].asArray(Float.self)))")
        try! MLX.save(
            arrays: ["f0": f0],
            url: URL(fileURLWithPath: outputDir + "/swift_1_f0.safetensors")
        )

        // Step 2: Upsample F0 (Swift uses tiling, Python uses Upsample)
        let f0Up = tiled(f0.expandedDimensions(axis: 2), repetitions: [1, 1, 480])  // [B, T, 480]
        let f0Flat = f0Up.reshaped([f0Up.shape[0], -1, 1])  // [B, T_high, 1]
        eval(f0Flat)
        print("\nStep 2: F0 upsampled: \(f0Flat.shape)")
        print("  F0_up range: [\(f0Flat.min().item(Float.self)), \(f0Flat.max().item(Float.self))]")
        try! MLX.save(
            arrays: ["f0_upsampled": f0Flat],
            url: URL(fileURLWithPath: outputDir + "/swift_2_f0_upsampled.safetensors")
        )

        // Step 3: Source Generation
        let (s, _, _) = vocoder.mSource(f0Flat)
        eval(s)
        print("\nStep 3: Source signal: \(s.shape)")
        print("  Source range: [\(s.min().item(Float.self)), \(s.max().item(Float.self))]")
        print("  Source first 10: \(Array(s[0, 0..<10, 0].asArray(Float.self)))")
        try! MLX.save(
            arrays: ["source": s],
            url: URL(fileURLWithPath: outputDir + "/swift_3_source.safetensors")
        )

        // Step 4: Conv Pre
        let xConvPre = vocoder.convPre(x)  // [B, T, 512]
        eval(xConvPre)
        print("\nStep 4: Conv pre: \(xConvPre.shape)")
        print("  Conv pre range: [\(xConvPre.min().item(Float.self)), \(xConvPre.max().item(Float.self))]")
        try! MLX.save(
            arrays: ["conv_pre": xConvPre],
            url: URL(fileURLWithPath: outputDir + "/swift_4_conv_pre.safetensors")
        )

        print("\n" + String(repeating: "=", count: 80))
        print("✅ Saved Swift vocoder intermediate outputs to:")
        print("   \(outputDir)")
        print(String(repeating: "=", count: 80))
    }
}
