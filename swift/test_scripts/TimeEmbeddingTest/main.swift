import MLX
import MLXRandom
import MLXNN
import Foundation
import Nightingale

print(String(repeating: "=", count: 60))
print("🦅 SWIFT - Full TimeMLP Test")
print(String(repeating: "=", count: 60))

// Load model weights
let s3genPath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/models/chatterbox/s3gen.safetensors")
let flowWeights = try! MLX.loadArrays(url: s3genPath)
let s3gen = S3Gen(flowWeights: flowWeights, vocoderWeights: nil)

// Test with t=0.5
let t = MLXArray([0.5])

// Enable debug
TimeMLP.debugEnabled = true

// Run through TimeMLP
let tMlpOut = s3gen.decoder.timeMLP(t)
eval(tMlpOut)

print("\nAfter time_mlp (Sinusoidal→Linear→SiLU→Linear):")
print("  Shape: \(tMlpOut.shape)")
print("  Mean: \(tMlpOut.mean().item(Float.self))")
print("  Std: \(tMlpOut.variance().sqrt().item(Float.self))")
print("  Range: [\(tMlpOut.min().item(Float.self)), \(tMlpOut.max().item(Float.self))]")
print("  [:5]: \(tMlpOut[0, 0..<5].asArray(Float.self))")
print(String(repeating: "=", count: 60))
