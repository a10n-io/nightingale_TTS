#!/usr/bin/env swift

import Foundation
import MLX

print(String(repeating: "=", count: 80))
print("TEST EMBED.LINEAR ONLY - Swift")
print(String(repeating: "=", count: 80))

// Load model weights
let modelsPath = "models/chatterbox"
let flowPath = "\(modelsPath)/s3gen.safetensors"
let flowWeights = try! MLX.loadArrays(url: URL(fileURLWithPath: flowPath))

// Load tokens
let tokensPath = "test_audio/cross_validate/python_speech_tokens.safetensors"
let tokensData = try! MLX.loadArrays(url: URL(fileURLWithPath: tokensPath))
let tokens = tokensData["speech_tokens"]!  // [T]

// Load input_embedding
let inputEmbWeight = flowWeights["flow.input_embedding.weight"]!  // [vocab_size, 512]

// Load embed.linear weights
var embedLinearWeight: MLXArray? = nil
var embedLinearBias: MLXArray? = nil

for (key, value) in flowWeights {
    if key.contains("encoder.embed.out.0.weight") {
        embedLinearWeight = value
        print("✅ Found embed.linear.weight: \(key)")
    }
    if key.contains("encoder.embed.out.0.bias") {
        embedLinearBias = value
        print("✅ Found embed.linear.bias: \(key)")
    }
}

guard let weight = embedLinearWeight else {
    print("❌ embed.linear.weight not found!")
    exit(1)
}

print("\n📊 Weights loaded:")
print("   input_embedding: \(inputEmbWeight.shape)")
print("   embed.linear.weight: \(weight.shape)")
if let bias = embedLinearBias {
    print("   embed.linear.bias: \(bias.shape)")
}

// ==============================================================================
// Forward pass: tokens -> embedding lookup -> embed.linear
// ==============================================================================

print("\n" + String(repeating: "=", count: 80))
print("FORWARD PASS")
print(String(repeating: "=", count: 80))

// Step 1: Embedding lookup
print("\n1. Embedding lookup")
eval(tokens)
print("   tokens shape: \(tokens.shape)")
let firstTokens = tokens[0..<min(10, tokens.shape[0])]
eval(firstTokens)
let first10 = (0..<min(10, tokens.shape[0])).map { firstTokens[$0].item(Int32.self) }
print("   First 10 tokens: \(first10)")

// Gather embeddings
let tokenEmbs = take(inputEmbWeight, tokens.asType(.int32), axis: 0)  // [T, 512]
eval(tokenEmbs)
let embMean = tokenEmbs.mean()
let embStd = tokenEmbs.variance().sqrt()
eval(embMean, embStd)
print("   token_embs shape: \(tokenEmbs.shape)")
print("   token_embs stats: mean=\(embMean.item(Float.self)), std=\(embStd.item(Float.self))")

// Step 2: embed.linear
print("\n2. embed.linear (RAW WEIGHT - NO TRANSPOSE YET)")
eval(weight)
print("   Weight shape (PyTorch format): \(weight.shape)")

// PyTorch Linear weight is [out, in]
// We need to transpose to [in, out] for MLX
let weightTransposed = weight.transposed()
print("   Weight shape (after transpose): \(weightTransposed.shape)")

// Now do: output = tokenEmbs @ weightTransposed
var output = matmul(tokenEmbs, weightTransposed)
if let bias = embedLinearBias {
    output = output + bias
}

eval(output)
let outMean = output.mean()
let outStd = output.variance().sqrt()
let outMin = output.min()
let outMax = output.max()
eval(outMean, outStd, outMin, outMax)

print("   output shape: \(output.shape)")
print("   output stats: mean=\(outMean.item(Float.self)), std=\(outStd.item(Float.self)), range=[\(outMin.item(Float.self)), \(outMax.item(Float.self))]")

// Save for comparison
try? MLX.save(
    arrays: [
        "token_embs": tokenEmbs,
        "embed_linear_output": output,
    ],
    url: URL(fileURLWithPath: "test_audio/forensic/swift_embed_linear_only.safetensors")
)

print("\n✅ Saved to: test_audio/forensic/swift_embed_linear_only.safetensors")

print("\n" + String(repeating: "=", count: 80))
print("COMPARISON")
print(String(repeating: "=", count: 80))
print("Python: mean=0.000108, std=0.504991")
print("Swift:  mean=\(outMean.item(Float.self)), std=\(outStd.item(Float.self))")

if abs(outMean.item(Float.self) - 0.000108) < 0.001 && abs(outStd.item(Float.self) - 0.504991) < 0.01 {
    print("\n✅ MATCH! embed.linear works correctly")
} else {
    print("\n❌ MISMATCH! embed.linear diverges from Python")
    print("   This suggests weight transposition issue!")
}

print(String(repeating: "=", count: 80))
