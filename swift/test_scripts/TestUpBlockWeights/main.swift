import Foundation
import MLX
import MLXNN
import Nightingale

let PROJECT_ROOT = "/Users/a10n/Projects/nightingale_TTS"
let modelDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/models/chatterbox")

print("=" + String(repeating: "=", count: 79))
print("UP BLOCK WEIGHTS TEST")
print("=" + String(repeating: "=", count: 79))

// Load decoder weights
let decoderURL = modelDir.appendingPathComponent("decoder_weights.safetensors")
let weights = try MLX.loadArrays(url: decoderURL)
print("Loaded \(weights.count) tensors\n")

// Check up_blocks.0.0 (ResNet block)
let prefix = "s3gen.flow.decoder.estimator.up_blocks.0.0"

print("Checking up_blocks.0.0.block1 weights:")
if let w = weights["\(prefix).block1.block.0.weight"] {
    eval(w)
    print("  block1.block.0.weight: \(w.shape), range=[\(w.min().item(Float.self)), \(w.max().item(Float.self))]")
} else {
    print("  ❌ block1.block.0.weight NOT FOUND")
}

if let b = weights["\(prefix).block1.block.0.bias"] {
    eval(b)
    print("  block1.block.0.bias: \(b.shape), range=[\(b.min().item(Float.self)), \(b.max().item(Float.self))]")
} else {
    print("  ❌ block1.block.0.bias NOT FOUND")
}

print("\nChecking up_blocks.0.0.res_conv weights:")
if let w = weights["\(prefix).res_conv.weight"] {
    eval(w)
    print("  res_conv.weight: \(w.shape), range=[\(w.min().item(Float.self)), \(w.max().item(Float.self))]")
} else {
    print("  ❌ res_conv.weight NOT FOUND")
}

// Create a simple test input
print("\n" + String(repeating: "=", count: 80))
print("Testing CausalBlock1D with up block dimensions")
print(String(repeating: "=", count: 80))

// Create random input: [1, 512, 696] (after concatenation)
let testInput = MLXRandom.uniform(low: -1.0, high: 1.0, [1, 512, 696])
eval(testInput)
print("Test input: \(testInput.shape), range=[\(testInput.min().item(Float.self)), \(testInput.max().item(Float.self))]")

// Create CausalBlock1D(512 -> 256)
let block1 = CausalBlock1D(dim: 512, dimOut: 256)

// Try to load weights manually
if let w = weights["\(prefix).block1.block.0.weight"],
   let b = weights["\(prefix).block1.block.0.bias"],
   let nw = weights["\(prefix).block1.block.2.weight"],
   let nb = weights["\(prefix).block1.block.2.bias"] {
    // CausalBlock1D has: conv, activation (Mish), norm
    // Weight loading would happen through update(parameters:)
    print("\n✅ All block1 weights found!")
    print("   Conv weight: \(w.shape)")
    print("   Conv bias: \(b.shape)")
    print("   Norm weight: \(nw.shape)")
    print("   Norm bias: \(nb.shape)")
} else {
    print("\n❌ Some weights missing!")
}

print("\n" + String(repeating: "=", count: 80))
print("CONCLUSION")
print(String(repeating: "=", count: 80))
print("The up_blocks.0.0 weights exist in decoder_weights.safetensors")
print("Next: Check if weights are being loaded into the module correctly")
