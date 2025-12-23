import Foundation
import MLX
import MLXNN
import MLXRandom
import Nightingale

print("Loading models...")
let s3gen = try! S3Gen(weightsURL: URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/models/chatterbox/s3gen.safetensors"))
print("✅ S3Gen loaded")

// Load sujano voice
let voicePath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/baked_voices/sujano/baked_voice.safetensors")
let voiceData = try! MLX.loadArrays(url: voicePath)
let speakerEmb = voiceData["speaker_emb"]!
let speechEmbMatrix = voiceData["speech_emb_matrix"]!
let promptToken = voiceData["prompt_token"]!
let promptFeat = voiceData["prompt_feat"]!

// Load Python speech tokens
let pythonTokensPath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/test_audio/cross_validate/python_speech_tokens.safetensors")
let pythonData = try! MLX.loadArrays(url: pythonTokensPath)
let pythonTokens = pythonData["speech_tokens"]!

print("\nRunning decoder to get mel spectrogram...")
let (mel, _) = s3gen.getEncoderAndFlowOutput(
    tokens: pythonTokens,
    speakerEmb: speakerEmb,
    speechEmbMatrix: speechEmbMatrix,
    promptToken: promptToken,
    promptFeat: promptFeat
)

eval(mel)
print("\n📊 Swift Sujano Mel:")
print("   Shape: \(mel.shape)")
print("   Range: [\(mel.min().item(Float.self)), \(mel.max().item(Float.self))]")
print("   Mean: \(mel.mean().item(Float.self)), std: \(mel.variance().sqrt().item(Float.self))")

// Save mel
let outputPath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/test_audio/swift_sujano_mel.safetensors")
try! MLX.save(arrays: ["mel": mel], url: outputPath)
print("\n✅ Saved Swift sujano mel to: \(outputPath.path)")
