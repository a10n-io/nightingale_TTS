import Foundation
import MLX
import MLXNN
import Nightingale

@main
struct SaveSTFTInput {
    static func main() async throws {
        print(String(repeating: "=", count: 80))
        print("SWIFT: SAVE STFT INPUT TO source_downs[0]")
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

        // Prepare input: [80, T] -> [1, 80, T] for vocoder
        let mel = pythonDecoderMel.expandedDimensions(axis: 0)  // [1, 80, T]
        eval(mel)

        // Transpose: [B, 80, T] -> [B, T, 80]
        var x = mel.transposed(0, 2, 1)
        eval(x)

        // F0 Prediction
        let f0 = vocoder.f0Predictor(x)  // [B, T]
        eval(f0)

        // Upsample F0
        let f0Up = tiled(f0.expandedDimensions(axis: 2), repetitions: [1, 1, 480])  // [B, T, 480]
        let f0Flat = f0Up.reshaped([f0Up.shape[0], -1, 1])  // [B, T_high, 1]
        eval(f0Flat)

        // Generate Source
        let (s, _, _) = vocoder.mSource(f0Flat)  // [B, T_high, 1]
        eval(s)

        print("\nSource signal shape: \(s.shape)")
        print("Source signal range: [\(s.min().item(Float.self)), \(s.max().item(Float.self))]")

        // ========== CRITICAL: RAW STFT OUTPUT ==========
        print("\n" + String(repeating: "=", count: 80))
        print("CAPTURING RAW STFT OUTPUT (INPUT TO source_downs[0])")
        print(String(repeating: "=", count: 80))

        // s is [B, T_high, 1], need to squeeze to [B, T_high] for STFT
        let sFlat = s.squeezed(axis: 2)  // [B, T_high]
        eval(sFlat)

        // Get raw STFT (returns [B, F, T'] format)
        let (sStftReal, sStftImag) = vocoder.stft(x: sFlat, nFFT: Mel2Wav.nFFT, hopLength: Mel2Wav.hopLength, window: vocoder.stftWindow)
        eval(sStftReal, sStftImag)

        print("\n🔍 SWIFT STFT OUTPUT (before transpose and concat):")
        print("  Real shape: \(sStftReal.shape)")
        print("  Imag shape: \(sStftImag.shape)")
        print("  Real mean: \(sStftReal.mean().item(Float.self))")
        print("  Imag mean: \(sStftImag.mean().item(Float.self))")
        print("  Real range: [\(sStftReal.min().item(Float.self)), \(sStftReal.max().item(Float.self))]")
        print("  Imag range: [\(sStftImag.min().item(Float.self)), \(sStftImag.max().item(Float.self))]")

        // Print first 10 values of first frequency bin
        let realFirst10 = Array(sStftReal[0, 0, 0..<min(10, sStftReal.shape[2])].asArray(Float.self))
        let imagFirst10 = Array(sStftImag[0, 0, 0..<min(10, sStftImag.shape[2])].asArray(Float.self))
        print("\n  Real bin 0, first 10 time steps: \(realFirst10)")
        print("  Imag bin 0, first 10 time steps: \(imagFirst10)")

        // Now transpose to match Python format: [B, F, T'] -> [B, T', F]
        // Then concatenate along last dim to match what source_downs[0] expects
        let sStftRealT = sStftReal.transposed(0, 2, 1)  // [B, F, T'] -> [B, T', F]
        let sStftImagT = sStftImag.transposed(0, 2, 1)  // [B, F, T'] -> [B, T', F]
        let sStftConcat = concatenated([sStftRealT, sStftImagT], axis: 2)  // [B, T', n_fft+2]
        eval(sStftConcat)

        print("\n🔍 SWIFT STFT INPUT TO source_downs[0] (after transpose and concat):")
        print("  Shape: \(sStftConcat.shape)")
        print("  Mean: \(sStftConcat.mean().item(Float.self))")

        // Save for comparison
        let outputDir = "../test_audio/stft_dump"
        try? FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

        // Save in PyTorch format [B, C, T] for easy comparison
        // Need to transpose back: [B, T', F] -> [B, F, T']
        let sStftRealPyFormat = sStftReal  // Already [B, F, T']
        let sStftImagPyFormat = sStftImag  // Already [B, F, T']
        let sStftConcatPyFormat = concatenated([sStftRealPyFormat, sStftImagPyFormat], axis: 1)  // [B, 2F, T']

        try! MLX.save(
            arrays: [
                "s_stft_real": sStftRealPyFormat,
                "s_stft_imag": sStftImagPyFormat,
                "s_stft_concat": sStftConcatPyFormat,
                "source_signal": s
            ],
            url: URL(fileURLWithPath: outputDir + "/swift_stft_input.safetensors")
        )

        print("\n" + String(repeating: "=", count: 80))
        print("✅ Saved to: \(outputDir)/swift_stft_input.safetensors")
        print(String(repeating: "=", count: 80))
        print("\n📋 NEXT: Run comparison script to analyze the difference")
        print(String(repeating: "=", count: 80))
    }
}
