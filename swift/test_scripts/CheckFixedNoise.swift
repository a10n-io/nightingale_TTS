import Foundation
import MLX
import MLXRandom
import Nightingale

@main
struct CheckFixedNoise {
    static func main() async throws {
        print("Checking fixed noise initialization...")

        // Load ChatterboxEngine which initializes S3Gen with fixed noise
        let modelsPath = URL(fileURLWithPath: "../models/chatterbox")
        let engine = ChatterboxEngine()
        try await engine.loadModels(modelsURL: modelsPath)

        guard let s3gen = await engine.s3gen else {
            fatalError("S3Gen not loaded!")
        }

        // Check the fixed noise
        let noise = s3gen.fixedNoise
        eval(noise)

        print("\nSwift fixed noise:")
        print("  Shape: \(noise.shape)")
        print("  Mean: \(noise.mean().item(Float.self))")
        print("  Std: \(noise.variance().sqrt().item(Float.self))")
        print("  Range: [\(noise.min().item(Float.self)), \(noise.max().item(Float.self))]")

        // Check first 5 values
        let first5 = noise[0, 0, 0..<5]
        eval(first5)
        print("  First 5 values [0,0,:5]:")
        for i in 0..<5 {
            print("    [\(i)]: \(first5[i].item(Float.self))")
        }

        // Compare with Python's expected values (seed 0)
        print("\n✅ Expected Python values (seed 0):")
        print("  First 5: [-1.1258, -1.1524, -0.2506, -0.4339, 0.8487]")
    }
}
