import Foundation
import MLX
import MLXNN
import Nightingale

@main
struct TraceFusionLayer0 {
    static func main() async throws {
        print(String(repeating: "=", count: 80))
        print("SWIFT VOCODER FUSION LAYER 0 TRACE")
        print(String(repeating: "=", count: 80))
        print("Goal: Capture x (main path) and s (source path) BEFORE fusion")
        print(String(repeating: "=", count: 80))

        print("\nLoading ChatterboxEngine...")
        let modelsPath = URL(fileURLWithPath: "../models/chatterbox")
        let engine = ChatterboxEngine()
        try await engine.loadModels(modelsURL: modelsPath)

        guard let s3gen = await engine.s3gen else {
            fatalError("S3Gen not loaded!")
        }

        let vocoder = s3gen.vocoder

        // Load Python decoder mel
        print("Loading Python decoder mel...")
        let melPath = "../test_audio/cross_validate/python_decoder_mel_for_swift_vocoder.safetensors"
        let melArrays = try! MLX.loadArrays(url: URL(fileURLWithPath: melPath))
        let pythonDecoderMel = melArrays["mel"]!  // [80, T]

        print("\nInput mel: \(pythonDecoderMel.shape)")

        // Prepare input: [80, T] -> [1, 80, T] for vocoder
        let mel = pythonDecoderMel.expandedDimensions(axis: 0)  // [1, 80, T]
        eval(mel)
        print("Mel batched: \(mel.shape)")

        // Transpose: [B, 80, T] -> [B, T, 80]
        var x = mel.transposed(0, 2, 1)
        eval(x)
        print("Speech feat (transposed): \(x.shape)")

        // F0 Prediction
        let f0 = vocoder.f0Predictor(x)  // [B, T]
        eval(f0)
        print("\nF0: \(f0.shape)")

        // Upsample F0
        let f0Up = tiled(f0.expandedDimensions(axis: 2), repetitions: [1, 1, 480])  // [B, T, 480]
        let f0Flat = f0Up.reshaped([f0Up.shape[0], -1, 1])  // [B, T_high, 1]
        eval(f0Flat)
        print("F0 upsampled: \(f0Flat.shape)")

        // Generate Source
        let (s, _, _) = vocoder.mSource(f0Flat)  // [B, T_high, 1]
        eval(s)
        print("Source: \(s.shape)")

        // Source STFT
        // s is [B, T_high, 1], need to squeeze to [B, T_high] for STFT
        let sFlat = s.squeezed(axis: 2)  // [B, T_high]
        eval(sFlat)

        let (sStftReal, sStftImag) = vocoder.stft(x: sFlat, nFFT: Mel2Wav.nFFT, hopLength: Mel2Wav.hopLength, window: vocoder.stftWindow)
        // STFT returns [B, F, T'], but MLX Conv1d needs [B, T, C]
        // So transpose to [B, T', F], then concatenate along last dim
        let sStftRealT = sStftReal.transposed(0, 2, 1)  // [B, F, T'] -> [B, T', F]
        let sStftImagT = sStftImag.transposed(0, 2, 1)  // [B, F, T'] -> [B, T', F]
        let sStft = concatenated([sStftRealT, sStftImagT], axis: 2)  // [B, T', n_fft+2]
        eval(sStft)
        print("Source STFT: \(sStft.shape)")

        // Conv Pre (main path start)
        x = vocoder.convPre(x)  // [B, T, 512]
        eval(x)
        print("Conv pre: \(x.shape)")

        // ========== LAYER 0 FUSION TRACE ==========
        print("\n" + String(repeating: "=", count: 80))
        print("LAYER 0 FUSION - THE CRITICAL DIAGNOSTIC")
        print(String(repeating: "=", count: 80))

        let i = 0

        // 1. Main Path Upsample
        // Apply leaky ReLU (slope=0.1 from S3Gen.swift:2305)
        x = leakyRelu(x, negativeSlope: 0.1)
        eval(x)

        // Upsample
        var xUp = vocoder.ups[i](x)
        eval(xUp)

        print("\n🔍 SWIFT FUSION [Layer \(i)]")
        print("  x_up (Main Path after ups[\(i)]): shape=\(xUp.shape)")
        print("    mean=\(xUp.mean().item(Float.self))")
        print("    std=\(xUp.variance().sqrt().item(Float.self))")
        print("    range=[\(xUp.min().item(Float.self)), \(xUp.max().item(Float.self))]")

        // Get first 10 values (note: Swift is [B, T, C], Python is [B, C, T])
        let xUpFirst10 = Array(xUp[0, 0..<min(10, xUp.shape[1]), 0].asArray(Float.self))
        print("    first 10: \(xUpFirst10)")

        // 2. Source Path Downsample
        var sDown = vocoder.sourceDowns[i](sStft)
        eval(sDown)

        print("\n  s_down (Source Path after source_downs[\(i)]): shape=\(sDown.shape)")
        print("    mean=\(sDown.mean().item(Float.self))")
        print("    std=\(sDown.variance().sqrt().item(Float.self))")
        print("    range=[\(sDown.min().item(Float.self)), \(sDown.max().item(Float.self))]")

        let sDownFirst10 = Array(sDown[0, 0..<min(10, sDown.shape[1]), 0].asArray(Float.self))
        print("    first 10: \(sDownFirst10)")

        // Save for comparison with Python
        let outputDir = "../test_audio/fusion_trace"
        try? FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

        try! MLX.save(
            arrays: [
                "x_up": xUp,
                "s_down": sDown
            ],
            url: URL(fileURLWithPath: outputDir + "/swift_fusion_layer0_pre.safetensors")
        )

        // 3. The Fusion
        let xFused = xUp + sDown
        eval(xFused)

        print("\n  x_fused (After x + s): shape=\(xFused.shape)")
        print("    mean=\(xFused.mean().item(Float.self))")
        print("    std=\(xFused.variance().sqrt().item(Float.self))")
        print("    range=[\(xFused.min().item(Float.self)), \(xFused.max().item(Float.self))]")

        // 4. Source Resblock
        let xAfterResblock = vocoder.sourceResBlocks[i](xFused)
        eval(xAfterResblock)

        print("\n  x_after_resblock (After source_resblocks[\(i)]): shape=\(xAfterResblock.shape)")
        print("    mean=\(xAfterResblock.mean().item(Float.self))")
        print("    std=\(xAfterResblock.variance().sqrt().item(Float.self))")
        print("    range=[\(xAfterResblock.min().item(Float.self)), \(xAfterResblock.max().item(Float.self))]")

        try! MLX.save(
            arrays: [
                "x_fused": xFused,
                "x_after_resblock": xAfterResblock
            ],
            url: URL(fileURLWithPath: outputDir + "/swift_fusion_layer0_post.safetensors")
        )

        print("\n" + String(repeating: "=", count: 80))
        print("✅ Saved Swift Layer 0 fusion trace to:")
        print("   \(outputDir)/swift_fusion_layer0_pre.safetensors")
        print("   \(outputDir)/swift_fusion_layer0_post.safetensors")
        print(String(repeating: "=", count: 80))
    }
}
