import Foundation
import MLX
import Nightingale

print(String(repeating: "=", count: 80))
print("CHECK INPUT_EMBEDDING WEIGHTS - Swift")
print(String(repeating: "=", count: 80))

// Load model
let modelsPath = "../models/chatterbox"
let s3gen = try! S3Gen(weightsFolder: modelsPath)

// Access inputEmbedding.weight
let inputEmbWeight = s3gen.inputEmbedding.weight
eval(inputEmbWeight)

print("\n✅ Swift inputEmbedding.weight")
print("   Shape: \(inputEmbWeight.shape)")

let embMin = inputEmbWeight.min()
let embMax = inputEmbWeight.max()
let embMean = inputEmbWeight.mean()
let embStd = inputEmbWeight.variance().sqrt()
eval(embMin, embMax, embMean, embStd)

print("   Range: [\(embMin.item(Float.self)), \(embMax.item(Float.self))]")
print("   Mean: \(embMean.item(Float.self))")
print("   Std: \(embStd.item(Float.self))")

print("\n📊 First 10 embeddings for token 0:")
let token0 = inputEmbWeight[0, 0..<10]
eval(token0)
let token0Arr = (0..<10).map { token0[$0].item(Float.self) }
print("   \(token0Arr)")

if inputEmbWeight.shape[0] > 500 {
    print("\n📊 First 10 embeddings for token 500:")
    let token500 = inputEmbWeight[500, 0..<10]
    eval(token500)
    let token500Arr = (0..<10).map { token500[$0].item(Float.self) }
    print("   \(token500Arr)")
}

if inputEmbWeight.shape[0] > 1000 {
    print("\n📊 First 10 embeddings for token 1000:")
    let token1000 = inputEmbWeight[1000, 0..<10]
    eval(token1000)
    let token1000Arr = (0..<10).map { token1000[$0].item(Float.self) }
    print("   \(token1000Arr)")
}

print("\n" + String(repeating: "=", count: 80))
print("COMPARE WITH PYTHON")
print(String(repeating: "=", count: 80))
print("Python token 0[:10]:")
print("  [-0.387375, -0.462175, -0.161073, -0.120614, 0.464650, 0.281306, -0.095799, -0.919886, 0.057988, -0.600852]")
print("\nPython token 500[:10]:")
print("  [0.039843, 0.392456, -0.474283, 0.494111, -0.294568, 0.281250, -0.027729, -0.470035, -0.409847, -0.749390]")
print("\nPython token 1000[:10]:")
print("  [-0.365420, -0.164035, 0.154215, -0.458009, -0.074446, 0.284938, -0.182239, 0.213400, 0.090599, -0.902424]")

print("\n✅ If Swift values match Python → inputEmbedding loaded correctly")
print("❌ If Swift values differ → inputEmbedding loading issue!")

print("\n" + String(repeating: "=", count: 80))
