import MLX
import MLXRandom
import Foundation
import Nightingale

print(String(repeating: "=", count: 80))
print("GENERATE RAW SWIFT MEL")
print(String(repeating: "=", count: 80))

// Load inputs from forensic directory (use same inputs as Python)
let inputPath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/test_audio/cross_validate/python_speech_tokens.safetensors")
let tokensData = try! MLX.loadArrays(url: inputPath)
let tokens = tokensData["speech_tokens"]!

// Load voice
let voicePath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/baked_voices/samantha/baked_voice.safetensors")
let voiceData = try! MLX.loadArrays(url: voicePath)
let speakerEmb = voiceData["t3.speaker_emb"]!
let speechEmbMatrix = voiceData["gen.embedding"]!
let promptToken = voiceData["gen.prompt_token"]!
let promptFeat = voiceData["gen.prompt_feat"]!

print("\n📊 Inputs:")
print("   tokens: \(tokens.shape)")
print("   speaker_emb: \(speakerEmb.shape)")
print("   speech_emb_matrix: \(speechEmbMatrix.shape)")
print("   prompt_token: \(promptToken.shape)")
print("   prompt_feat: \(promptFeat.shape)")

// Load S3Gen
let weightsPath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/models/chatterbox/s3gen.safetensors")
let weights = try! MLX.loadArrays(url: weightsPath)

print("\nLoading S3Gen...")
let s3gen = S3Gen(flowWeights: weights, vocoderWeights: weights)

print("\n🔬 Generating RAW Swift mel (decoder only, no vocoder)...")
let (melFull, _) = s3gen.getEncoderAndFlowOutput(
    tokens: tokens,
    speakerEmb: speakerEmb,
    speechEmbMatrix: speechEmbMatrix,
    promptToken: promptToken,
    promptFeat: promptFeat
)

eval(melFull)

// Extract regions
let promptLen = promptToken.shape[1]
let melPrompt = melFull[0..., 0..., 0..<promptLen]
let melGen = melFull[0..., 0..., promptLen...]

eval(melPrompt)
eval(melGen)

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

// Channel-by-channel stats
print("\n   Per-channel statistics (generated region):")
for i in [0, 20, 40, 60, 79] {
    let chan = melGen[0, i, 0...]
    eval(chan)
    let chanMean = chan.mean().item(Float.self)
    let chanStd = chan.variance().sqrt().item(Float.self)
    let chanMin = chan.min().item(Float.self)
    let chanMax = chan.max().item(Float.self)
    print(String(format: "     Ch%2d: mean=%8.5f, std=%7.5f, range=[%8.5f, %7.5f]", i, chanMean, chanStd, chanMin, chanMax))
}

// Save
let outputPath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/test_audio/forensic/swift_mel_raw.safetensors")
try! MLX.save(arrays: [
    "mel_full": melFull,
    "mel_prompt": melPrompt,
    "mel_gen": melGen
], url: outputPath)

print("\n✅ Saved to: forensic/swift_mel_raw.safetensors")
print(String(repeating: "=", count: 80))
