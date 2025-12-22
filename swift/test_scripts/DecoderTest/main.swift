import Foundation
import MLX
import MLXNN
import Nightingale

print("================================================================================")
print("DECODER COMPARISON TEST: Python vs Swift")
print("================================================================================")

// Load Python decoder trace
let PROJECT_ROOT = "/Users/a10n/Projects/nightingale_TTS"
let tracePath = URL(fileURLWithPath: "\(PROJECT_ROOT)/test_audio/python_decoder_trace.safetensors")
print("\nLoading Python trace from: \(tracePath.path)")

let trace = try! MLX.loadArrays(url: tracePath)

let noise = trace["noise"]!
let mu = trace["mu"]!
let spk_cond = trace["spk_cond"]!
let conds = trace["conds"]!
let mask = trace["mask"]!
let t = trace["t"]!
let python_output = trace["decoder_output"]!

eval(noise, mu, spk_cond, conds, mask, t, python_output)

print("✅ Loaded Python trace:")
print("  noise: \(noise.shape), range=[\(noise.min().item(Float.self)), \(noise.max().item(Float.self))]")
print("  mu: \(mu.shape), range=[\(mu.min().item(Float.self)), \(mu.max().item(Float.self))]")
print("  spk_cond: \(spk_cond.shape), range=[\(spk_cond.min().item(Float.self)), \(spk_cond.max().item(Float.self))]")
print("  conds: \(conds.shape), range=[\(conds.min().item(Float.self)), \(conds.max().item(Float.self))]")
print("  mask: \(mask.shape), all ones")
print("  t: \(t.shape), value=\(t.item(Float.self))")
print("  python_output: \(python_output.shape), range=[\(python_output.min().item(Float.self)), \(python_output.max().item(Float.self))], mean=\(python_output.mean().item(Float.self))")

// Load S3Gen weights directly
print("\nLoading S3Gen weights...")
let modelDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/models/chatterbox")
let s3genPath = modelDir.appendingPathComponent("s3gen.safetensors")
let flowWeights = try! MLX.loadArrays(url: s3genPath)
print("✅ Loaded \(flowWeights.count) weight arrays from s3gen.safetensors")

// Initialize S3Gen
print("\nInitializing S3Gen...")
let s3gen = S3Gen(flowWeights: flowWeights, vocoderWeights: nil)
print("✅ S3Gen initialized")

// Get the decoder
let decoder = s3gen.decoder

// Run Swift decoder with EXACT same inputs
print("\n=== Running Swift Decoder ===")
FlowMatchingDecoder.debugStep = 1  // Enable debug output

let swift_output = decoder(x: noise, mu: mu, t: t, speakerEmb: spk_cond, cond: conds, mask: mask)
eval(swift_output)

print("\nSwift decoder output:")
print("  shape: \(swift_output.shape)")
print("  range: [\(swift_output.min().item(Float.self)), \(swift_output.max().item(Float.self))]")
print("  mean: \(swift_output.mean().item(Float.self))")

// Compare outputs
print("\n================================================================================")
print("COMPARISON:")
print("================================================================================")
print("Python output: range=[\(python_output.min().item(Float.self)), \(python_output.max().item(Float.self))], mean=\(python_output.mean().item(Float.self))")
print("Swift output:  range=[\(swift_output.min().item(Float.self)), \(swift_output.max().item(Float.self))], mean=\(swift_output.mean().item(Float.self))")

let diff = abs(swift_output - python_output)
eval(diff)
let max_diff = diff.max().item(Float.self)
let mean_diff = diff.mean().item(Float.self)

print("\nDifference (abs):")
print("  max: \(max_diff)")
print("  mean: \(mean_diff)")

if mean_diff < 0.01 {
    print("\n✅ PASS: Outputs match within tolerance!")
} else {
    print("\n❌ FAIL: Outputs diverge significantly!")
    print("   Outputs are NOT matching - there's a bug in the Swift decoder!")
}

print("================================================================================")
