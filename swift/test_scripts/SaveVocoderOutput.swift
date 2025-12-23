import Foundation
import MLX
import Nightingale

@main
struct SaveVocoderOutput {
    static func main() async throws {
        print("Loading ChatterboxEngine...")
        let modelsPath = URL(fileURLWithPath: "models/chatterbox")
        let engine = ChatterboxEngine()
        try await engine.loadModels(modelsURL: modelsPath)

        guard let s3gen = await engine.s3gen else {
            fatalError("S3Gen not loaded!")
        }

        print("\nLoading Swift decoder mel output...")
        // Load the decoder mel output from Swift
        let melPath = "test_audio/forensic/swift_decoder_mel.safetensors"
        let melArrays = try! MLX.loadArrays(url: URL(fileURLWithPath: melPath))
        var decoderMel = melArrays["mel"]!  // [80, T]

        print("Swift decoder mel input to vocoder:")
        print("  Shape: \(decoderMel.shape)")
        print("  Mean: \(decoderMel.mean().item(Float.self))")
        print("  Std: \(decoderMel.variance().sqrt().item(Float.self))")
        print("  Range: [\(decoderMel.min().item(Float.self)), \(decoderMel.max().item(Float.self))]")

        // Vocoder expects [B, C, T] format
        decoderMel = decoderMel.expandedDimensions(axis: 0)  // [1, 80, T]
        eval(decoderMel)

        print("\nRunning vocoder...")
        let audio = s3gen.vocoder(decoderMel)
        eval(audio)

        print("  Audio shape after vocoder: \(audio.shape)")

        // Flatten audio to 1D
        let audioFlat: MLXArray
        if audio.ndim == 3 {
            // [1, 1, T] -> [T]
            audioFlat = audio.squeezed(axis: 0).squeezed(axis: 0)
        } else if audio.ndim == 2 {
            // [1, T] -> [T]
            audioFlat = audio.squeezed(axis: 0)
        } else {
            // Already [T]
            audioFlat = audio
        }
        eval(audioFlat)

        print("\nSwift vocoder audio output:")
        print("  Shape: \(audioFlat.shape)")
        print("  Mean: \(audioFlat.mean().item(Float.self))")
        print("  Std: \(audioFlat.variance().sqrt().item(Float.self))")
        print("  Range: [\(audioFlat.min().item(Float.self)), \(audioFlat.max().item(Float.self))]")
        print("  Sample rate: 24000 Hz")
        print("  Duration: \(Float(audioFlat.shape[0]) / 24000.0) seconds")

        // Save
        let outputDir = "test_audio/forensic"
        try! MLX.save(
            arrays: ["audio": audioFlat],
            url: URL(fileURLWithPath: outputDir + "/swift_vocoder_audio.safetensors")
        )

        print("\n✅ Saved to: \(outputDir)/swift_vocoder_audio.safetensors")
    }
}
