import MLX
import MLXRandom
import Foundation
import Nightingale

print(String(repeating: "=", count: 80))
print("SAVE SWIFT DECODER MEL OUTPUT")
print(String(repeating: "=", count: 80))

// Load Python tokens
let pythonTokensPath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/test_audio/cross_validate/python_speech_tokens.safetensors")
let pythonTokensData = try! MLX.loadArrays(url: pythonTokensPath)
let pythonTokens = pythonTokensData["speech_tokens"]!
print("\nPython tokens shape: \(pythonTokens.shape)")
print("Python tokens: \(Array(pythonTokens.asArray(Int32.self).prefix(20)))")

// Load models
let s3genPath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/models/chatterbox/s3gen.safetensors")
let weights = try! MLX.loadArrays(url: s3genPath)

print("\nLoading S3Gen...")
let s3gen = S3Gen(flowWeights: weights, vocoderWeights: weights)

// Load voice
let voicePath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/baked_voices/samantha/baked_voice.safetensors")
let voiceData = try! MLX.loadArrays(url: voicePath)
let speakerEmb = voiceData["speaker_emb"]!
let speechEmbMatrix = voiceData["speech_emb_matrix"]!
let promptToken = voiceData["prompt_token"]!
let promptFeat = voiceData["prompt_feat"]!

print("\nRunning decoder to get mel spectrogram...")
// Get mel output from decoder (before vocoder)
let (mel, _) = s3gen.getEncoderAndFlowOutput(
    tokens: pythonTokens,
    speakerEmb: speakerEmb,
    speechEmbMatrix: speechEmbMatrix,
    promptToken: promptToken,
    promptFeat: promptFeat
)

eval(mel)
print("\n📊 Decoder output mel:")
print("   Shape: \(mel.shape)")
print("   Mean: \(mel.mean().item(Float.self))")
print("   Std: \(mel.variance().sqrt().item(Float.self))")
print("   Range: [\(mel.min().item(Float.self)), \(mel.max().item(Float.self))]")

// Check a few mel channels
print("\n📊 Mel channel statistics:")
for i in [0, 20, 40, 60, 79] {
    let channel = mel[0, i, 0...]
    eval(channel)
    print("   Channel \(i): mean=\(channel.mean().item(Float.self)), std=\(channel.variance().sqrt().item(Float.self))")
}

// Save mel
let outputPath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/test_audio/cross_validate/swift_mel.safetensors")
try! MLX.save(arrays: ["mel": mel], url: outputPath)
print("\n✅ Saved Swift mel to: \(outputPath.path)")

print("\n" + String(repeating: "=", count: 80))
