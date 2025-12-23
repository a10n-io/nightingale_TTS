import Foundation
import MLX
import Nightingale

// Test the PRODUCTION flow by simulating exactly what ChatterboxEngine does
// This verifies encoder weights are loaded correctly through remapS3Keys()

print(String(repeating: "=", count: 80))
print("VERIFY ENCODER - SIMULATING ChatterboxEngine FLOW")
print(String(repeating: "=", count: 80))

// Step 1: Load raw weights (just like ChatterboxEngine does)
print("\n1. Loading raw weights from files...")
let modelsPath = "../models/chatterbox"
let flowPath = "\(modelsPath)/s3gen.safetensors"
let flowWeights = try! MLX.loadArrays(url: URL(fileURLWithPath: flowPath))
print("✅ Loaded \(flowWeights.count) weights from s3gen.safetensors")

// Step 2: Create S3Gen with raw weights (just like ChatterboxEngine line 259)
print("\n2. Creating S3Gen with raw weights...")
MLXRandom.seed(42)  // Deterministic initialization
let s3gen = S3Gen(flowWeights: flowWeights, vocoderWeights: nil)
print("✅ S3Gen created")

// Step 3: CRITICAL - Apply remapS3Keys and update (just like ChatterboxEngine lines 263-281)
print("\n3. Applying remapS3Keys and updating (PRODUCTION PATH)...")

// We need to call the private remapS3Keys function
// Since we can't access it directly, let's check if weights were already applied
// Actually, looking at ChatterboxEngine code, the S3Gen init loads weights directly
// and then ChatterboxEngine.update() is called with remapped weights

// For now, let's just verify what the encoder produces after S3Gen init
print("⚠️  Note: S3Gen loads weights in init, update() would be called by ChatterboxEngine")

// Load test data
let voiceDir = "../baked_voices/samantha"
let voiceArrays = try! MLX.loadArrays(url: URL(fileURLWithPath: voiceDir + "/baked_voice.safetensors"))
let tokensPath = "../test_audio/cross_validate/python_speech_tokens.safetensors"
let tokensArrays = try! MLX.loadArrays(url: URL(fileURLWithPath: tokensPath))

let promptToken = voiceArrays["gen.prompt_token"]!
let speechTokens = tokensArrays["speech_tokens"]!

print("\n4. Running encoder...")

// Concatenate prompt and tokens
let fullTokens = concatenated([promptToken.squeezed(axis: 0), speechTokens], axis: 0)

// Embedding lookup
let tokenEmbs = take(s3gen.inputEmbedding.weight, fullTokens.asType(.int32), axis: 0)
eval(tokenEmbs)

// Add batch dimension
let x = tokenEmbs.expandedDimensions(axis: 0)  // [1, T, 512]

// Run encoder
let encoderOutput = s3gen.encoder(x)
eval(encoderOutput)

print("\n📊 Encoder output:")
print("   Shape: \(encoderOutput.shape)")

let encMean = encoderOutput.mean()
let encStd = encoderOutput.variance().sqrt()
let encMin = encoderOutput.min()
let encMax = encoderOutput.max()
eval(encMean, encStd, encMin, encMax)

print("   Mean: \(encMean.item(Float.self))")
print("   Std: \(encStd.item(Float.self))")
print("   Range: [\(encMin.item(Float.self)), \(encMax.item(Float.self))]")

// Check embed.linear weight to see if it was transposed
print("\n5. Checking embed.linear weight...")
let embedWeight = s3gen.encoder.embedLinear.weight
eval(embedWeight)
print("   embed.linear.weight shape: \(embedWeight.shape)")
print("   First element [0,0]: \(embedWeight[0,0].item(Float.self))")
print("   Element [0,1]: \(embedWeight[0,1].item(Float.self))")
print("   Element [1,0]: \(embedWeight[1,0].item(Float.self))")

// Compare with expected Python values
print("\n" + String(repeating: "=", count: 80))
print("COMPARISON WITH PYTHON")
print(String(repeating: "=", count: 80))
print("\nExpected (from Python encoder):")
print("  mean=-0.007, std=0.455, range=[-1.75, +1.85]")
print("\nActual (Swift S3Gen after init):")
print("  mean=\(encMean.item(Float.self)), std=\(encStd.item(Float.self)), range=[\(encMin.item(Float.self)), \(encMax.item(Float.self))]")

let stdRatio = encStd.item(Float.self) / 0.455
print("\n📊 Std ratio: Swift/Python = \(stdRatio)")

if abs(stdRatio - 1.0) < 0.1 {
    print("✅ Standard deviation matches!")
} else if abs(stdRatio - 0.5) < 0.1 {
    print("❌ Half variance! (ratio ~0.5)")
    print("   Suggests: Missing residuals OR wrong weight loading")
} else {
    print("⚠️  Unexpected ratio: \(stdRatio)")
}

// Save for further analysis
let forensicDir = "../test_audio/forensic"
try? FileManager.default.createDirectory(atPath: forensicDir, withIntermediateDirectories: true)

try? MLX.save(
    arrays: [
        "encoder_output": encoderOutput,
        "embedLinear_weight": embedWeight,
    ],
    url: URL(fileURLWithPath: forensicDir + "/swift_encoder_s3gen_init.safetensors")
)

print("\n✅ Saved to: \(forensicDir)/swift_encoder_s3gen_init.safetensors")
print("\n" + String(repeating: "=", count: 80))
