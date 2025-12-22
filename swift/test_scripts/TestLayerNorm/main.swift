import Foundation
import MLX
import MLXNN

print("================================================================================")
print("LAYERNORM PARAMETER TEST")
print("================================================================================")

// Create a simple LayerNorm
let ln = LayerNorm(dimensions: 256, eps: 1e-5)

print("\n1. Default LayerNorm parameters:")
if let w = ln.weight {
    print("  weight exists: \(w.shape), values=[\(w[0].item(Float.self)), \(w[1].item(Float.self)), ...]")
} else {
    print("  weight is nil!")
}

if let b = ln.bias {
    print("  bias exists: \(b.shape), values=[\(b[0].item(Float.self)), \(b[1].item(Float.self)), ...]")
} else {
    print("  bias is nil!")
}

// Try loading custom weights
print("\n2. Loading custom weights:")
let customWeight = MLXArray.ones([256])
let customBias = MLXArray.zeros([256])

ln.update(parameters: ModuleParameters.unflattened(["weight": customWeight, "bias": customBias]))

print("After update:")
if let w = ln.weight {
    eval(w)
    print("  weight: \(w.shape), values=[\(w[0].item(Float.self)), \(w[1].item(Float.self)), ..., \(w[255].item(Float.self))]")
} else {
    print("  weight is nil!")
}

if let b = ln.bias {
    eval(b)
    print("  bias: \(b.shape), values=[\(b[0].item(Float.self)), \(b[1].item(Float.self)), ..., \(b[255].item(Float.self))]")
} else {
    print("  bias is nil!")
}

// Test forward pass
print("\n3. Forward pass test:")
let input = MLXArray.ones([1, 10, 256])  // Simple ones tensor
let output = ln(input)
eval(output)
print("  Input: all ones")
print("  Output range: [\(output.min().item(Float.self)), \(output.max().item(Float.self))]")
print("  Output mean: \(MLX.mean(output).item(Float.self))")
print("  Expected output if working: normalized to ~0 mean, then scaled by weight (0-0.255) and shifted by bias (0.1)")

// Check what parameters are actually used
print("\n4. Check parameters dictionary:")
let params = ln.parameters()
let flattened = params.flattened()
print("  Parameters count: \(flattened.count)")
for (key, value) in flattened {
    eval(value)
    print("  \(key): \(value.shape)")
}

print("================================================================================")
