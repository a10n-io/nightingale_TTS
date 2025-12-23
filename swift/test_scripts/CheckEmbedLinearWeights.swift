import Foundation
import MLX
import Nightingale

@main
struct CheckEmbedLinearWeights {
    static func main() async throws {
        print("Loading ChatterboxEngine...")
        let modelsPath = URL(fileURLWithPath: "../models/chatterbox")
        let engine = ChatterboxEngine()
        try await engine.loadModels(modelsURL: modelsPath)

        guard let s3gen = await engine.s3gen else {
            fatalError("S3Gen not loaded!")
        }

        let w = s3gen.encoder.embedLinear.weight
        eval(w)

        print("=" + String(repeating: "=", count: 79))
        print("SWIFT EMBEDLINEAR WEIGHT (after loading via ChatterboxEngine)")
        print("=" + String(repeating: "=", count: 79))
        print("Shape: \(w.shape)")
        print("Sum: \(w.sum().item(Float.self))")
        print("Mean: \(w.mean().item(Float.self))")
        print("Std: \(w.variance().sqrt().item(Float.self))")
        print("Range: [\(w.min().item(Float.self)), \(w.max().item(Float.self))]")

        // Check first few VALUES to compare with Python
        print("\nFirst 5 values [0, :5]:")
        let row0 = w[0, 0..<5]
        eval(row0)
        for i in 0..<5 {
            print("  [\(i)]: \(row0[i].item(Float.self))")
        }

        print("\nFirst 5 values [:5, 0]:")
        let col0 = w[0..<5, 0]
        eval(col0)
        for i in 0..<5 {
            print("  [\(i)]: \(col0[i].item(Float.self))")
        }

        print("\n" + String(repeating: "=", count: 80))
        print("EXPECTED (Python transposed to MLX format [in, out]):")
        print(String(repeating: "=", count: 80))
        print("First 5 values [0, :5]: [-0.022813, 0.000403, -0.127293, -0.002905, 0.003286]")
        print("First 5 values [:5, 0]: [-0.022813, 0.015566, -0.000571, -0.013604, 0.017855]")
    }
}
