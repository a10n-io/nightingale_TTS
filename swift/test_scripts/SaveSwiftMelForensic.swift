import Foundation
import MLX
import MLXRandom
import MLXNN
import Nightingale

// Load the model
let modelsPath = "../models/chatterbox"
print("Loading Nightingale model from \(modelsPath)...")
let model = try! Nightingale(modelFolder: modelsPath)

// Load samantha voice
let voiceDir = "../baked_voices/samantha"
let voiceArrays = try! MLX.loadArrays(url: URL(fileURLWithPath: voiceDir + "/baked_voice.safetensors"))
let speakerEmb = voiceArrays["t3.speaker_emb"]!
let speechEmbMatrix = voiceArrays["gen.embedding"]!
let promptToken = voiceArrays["gen.prompt_token"]!
let promptFeat = voiceArrays["gen.prompt_feat"]!

print("\n📊 Loaded voice data:")
print("   speakerEmb: \(speakerEmb.shape)")
print("   speechEmbMatrix: \(speechEmbMatrix.shape)")
print("   promptToken: \(promptToken.shape)")
print("   promptFeat: \(promptFeat.shape)")

// Load Python tokens for exact comparison
let tokensPath = "../test_audio/cross_validate/python_speech_tokens.safetensors"
let tokensArrays = try! MLX.loadArrays(url: URL(fileURLWithPath: tokensPath))
let speechTokens = tokensArrays["speech_tokens"]!

print("\n📊 Loaded tokens:")
print("   speechTokens: \(speechTokens.shape)")

// Create output directory
let outputDir = "../test_audio/forensic"
try? FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

print("\n🔬 Generating RAW Swift mel (with forensic debugging)...")

// Generate mel with debugging enabled
let melFull = model.s3gen.generateMelFromTokens(
    speechTokens: speechTokens,
    speechEmbMatrix: speechEmbMatrix,
    speakerEmb: speakerEmb,
    promptToken: promptToken,
    promptFeat: promptFeat,
    temperature: 1.0,
    cfgStrength: 1.0,
    debug: true  // This will print forensic debug info
)

// Extract regions
let promptLen = promptToken.shape[1]
let melPrompt = melFull[0..., 0..., 0..<promptLen]
let melGen = melFull[0..., 0..., promptLen...]

eval(melFull, melPrompt, melGen)

print("\n📊 Swift Mel (RAW - no modifications):")
print("   Full shape: \(melFull.shape)")
print("   Full range: [\(melFull.min().item(Float.self)), \(melFull.max().item(Float.self))]")
print("   Full mean: \(melFull.mean().item(Float.self))")
print("   Full std: \(melFull.variance().sqrt().item(Float.self))")

print("\n   Prompt region (\(promptLen) frames):")
print("     Range: [\(melPrompt.min().item(Float.self)), \(melPrompt.max().item(Float.self))]")
print("     Mean: \(melPrompt.mean().item(Float.self))")

print("\n   Generated region (\(melGen.shape[2]) frames):")
print("     Range: [\(melGen.min().item(Float.self)), \(melGen.max().item(Float.self))]")
print("     Mean: \(melGen.mean().item(Float.self))")

// Per-channel statistics (generated region only)
print("\n   Per-channel statistics (generated region):")
for i in [0, 20, 40, 60, 79] {
    let chan = melGen[0, i, 0...]
    eval(chan)
    let chanMean = chan.mean().item(Float.self)
    let chanStd = chan.variance().sqrt().item(Float.self)
    let chanMin = chan.min().item(Float.self)
    let chanMax = chan.max().item(Float.self)
    print(String(format: "     Ch%2d: mean=%8.5f, std=%7.5f, range=[%8.5f, %7.5f]",
                 i, chanMean, chanStd, chanMin, chanMax))
}

// Save everything
let outputPath = outputDir + "/swift_mel_raw.safetensors"
try! MLX.save(
    arrays: [
        "mel_full": melFull,
        "mel_prompt": melPrompt,
        "mel_gen": melGen
    ],
    url: URL(fileURLWithPath: outputPath)
)

print("\n✅ Saved to: forensic/swift_mel_raw.safetensors")
print("=" + String(repeating: "=", count: 79))
