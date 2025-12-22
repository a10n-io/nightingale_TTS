import Foundation
import MLX
import MLXNN
import Nightingale

print(String(repeating: "=", count: 80))
print("DETERMINISTIC DECODER TEST")
print(String(repeating: "=", count: 80))

// === Deterministic tensor generation (matches Python exactly) ===
func getDeterministicTensor(_ shape: [Int]) -> MLXArray {
    let count = shape.reduce(1, *)
    let flat = MLXArray(0..<count).asType(Float.self)
    let data = sin(flat / 10.0)
    return data.reshaped(shape)
}

// Helper for debugging
func debugStats(_ x: MLXArray, name: String) {
    eval(x)
    let mean = x.mean().item(Float.self)
    let variance = x.variance().item(Float.self)
    let std = sqrt(variance)
    let minVal = x.min().item(Float.self)
    let maxVal = x.max().item(Float.self)
    print(String(format: "🔍 [%@] Shape: %@ | Mean: %.4f | Std: %.4f | Range: [%.4f, %.4f]",
                 name, "\(x.shape)", mean, std, minVal, maxVal))
}

print("\n--- 🧪 GENERATING DETERMINISTIC INPUTS ---")

// Generate inputs (matches Python shapes)
let L_total = 696
let L_pm = 500

let mu = getDeterministicTensor([1, 80, L_total])
var conds = getDeterministicTensor([1, 80, L_pm])
conds = concatenated([conds, MLXArray.zeros([1, 80, L_total - L_pm])], axis: 2)
let speaker_emb = getDeterministicTensor([1, 80])
let xt = getDeterministicTensor([1, L_total, 80])

print("✅ Generated deterministic tensors")

// Prepare decoder inputs (matches Python)
let x_transposed = xt.transposed(0, 2, 1)  // [1, 80, L_total]
let x_batch = concatenated([x_transposed, x_transposed], axis: 0)  // [2, 80, L_total]
let mu_batch = concatenated([mu, mu], axis: 0)  // [2, 80, L_total]
let conds_batch = concatenated([conds, MLXArray.zeros(like: conds)], axis: 0)  // [2, 80, L_total]
let speaker_batch = tiled(speaker_emb, repetitions: [2, 1])  // [2, 80]
let mask_batch = MLXArray.ones([2, 1, L_total])

// Time parameter
let n_timesteps = 10
var timesteps: [Float] = []
for i in 0...n_timesteps {
    let angle = Float.pi / 2.0 * Float(i) / Float(n_timesteps)
    timesteps.append(cos(angle) * cos(angle))
}
let t_curr = timesteps[0]
let t_batch = MLXArray([t_curr, t_curr])

print("\n--- LOADING DECODER ---")
let PROJECT_ROOT = "/Users/a10n/Projects/nightingale_TTS"
let s3genPath = URL(fileURLWithPath: "\(PROJECT_ROOT)/models/chatterbox/s3gen.safetensors")
let flowWeights = try! MLX.loadArrays(url: s3genPath)
let s3gen = S3Gen(flowWeights: flowWeights, vocoderWeights: nil)
let decoder = s3gen.decoder

print("✅ Decoder loaded")

print("\n--- RUNNING DECODER ---")
FlowMatchingDecoder.debugStep = 1  // Enable checkpoints

let output = decoder(x: x_batch, mu: mu_batch, t: t_batch, speakerEmb: speaker_batch, cond: conds_batch, mask: mask_batch)
eval(output)

print("\n--- FINAL OUTPUT ---")
debugStats(output, name: "FINAL_OUTPUT")

print("\n" + String(repeating: "=", count: 80))
print("✅ TEST COMPLETE - Compare with Python bisect_decoder.py output")
print(String(repeating: "=", count: 80))
