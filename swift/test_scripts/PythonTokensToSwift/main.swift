import Foundation
import MLX
import MLXRandom
import MLXNN
import Nightingale

let PROJECT_ROOT = "/Users/a10n/Projects/nightingale_TTS"
let modelDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/models/chatterbox")

print("=" + String(repeating: "=", count: 79))
print("PYTHON TOKENS → SWIFT AUDIO")
print("=" + String(repeating: "=", count: 79))

// Python tokens from generate_python_tokens.py
let pythonTokens: [Int] = [1732, 2068, 2186, 1457, 680, 1460, 3647, 5834, 5915, 5266, 5509, 5429, 3242, 569, 572, 599, 683, 719, 1448, 2087, 1976, 1946, 3890, 5192, 4814, 1277, 1736, 3650, 6077, 4780, 2163, 4752, 6377, 6373, 4771, 5014, 5015, 2021, 2020, 4448, 2671, 2112, 411, 2517, 5109, 5838, 5845, 5837, 2367, 4718, 6238, 1226, 2145, 1431, 5006, 3479, 1288, 2267, 5021, 4778, 1855, 4194, 2246, 193, 157, 1679, 2096, 4373, 4349, 6050, 1686, 1032, 2331, 5672, 586, 3990, 38, 4032, 6534, 4320, 4106, 3863, 3146, 4671, 5648, 627, 5325, 1620, 1892, 1865, 5510, 4789, 5186, 731, 734, 737, 116, 386]

print("\nPython tokens: \(pythonTokens.count) tokens")
print("First 10: \(Array(pythonTokens.prefix(10)))")

// Convert to MLXArray
let speechTokens = MLXArray(pythonTokens.map { Int32($0) }).expandedDimensions(axis: 0)
eval(speechTokens)
print("Token tensor: \(speechTokens.shape)")

// Load S3Gen
print("\nLoading S3Gen...")
let voiceDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/baked_voices/samantha")
let s3gen = try S3GenMLX.loadFromSafetensors(
    modelDir: modelDir,
    voiceDir: voiceDir,
    device: Device.default
)
print("✅ S3Gen loaded")

// Generate audio
print("\nGenerating audio from Python tokens using Swift S3Gen...")
let audio = s3gen.generate(
    tokens: speechTokens,
    speakerEmb: s3gen.speakerEmbedding!,
    speechEmbMatrix: s3gen.speechEmbMatrix!,
    promptToken: s3gen.promptToken!,
    promptFeat: s3gen.promptFeat!
)
eval(audio)

let audioArray = audio.asArray(Float.self)
print("Generated audio: \(audioArray.count) samples (\(String(format: "%.2f", Float(audioArray.count)/24000.0))s)")
print("   Range: [\(audioArray.min()!), \(audioArray.max()!)]")

// Save as WAV
let outputPath = "\(PROJECT_ROOT)/test_audio/python_tokens_swift_audio.wav"
try saveWAV(samples: audioArray, sampleRate: 24000, path: outputPath)

print("\n✅ Saved: \(outputPath)")
print("\n" + String(repeating: "=", count: 80))
