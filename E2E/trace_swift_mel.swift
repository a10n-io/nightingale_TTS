#!/usr/bin/env swift
// Trace Swift mel generation to see intermediate values

import Foundation
import MLX
import MLXRandom
import MLXNN

#if canImport(Nightingale)
import Nightingale
#else
fatalError("Cannot import Nightingale module")
#endif

let PROJECT_ROOT = "/Users/a10n/Projects/nightingale_TTS"
let modelDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/models/chatterbox")
let voiceDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/baked_voices/samantha")

print("=" + String(repeating: "=", count: 79))
print("SWIFT MEL TRACE")
print("=" + String(repeating: "=", count: 79))

// Load Python tokens
let pythonTokens: [Int] = [1732, 2068, 2186, 1457, 680, 1460, 3647, 5834, 5915, 5266, 5509, 5429, 3242, 569, 572, 599, 683, 719, 1448, 2087, 1976, 1946, 3890, 5192, 4814, 1277, 1736, 3650, 6077, 4780, 2163, 4752, 6377, 6373, 4771, 5014, 5015, 2021, 2020, 4448, 2671, 2112, 411, 2517, 5109, 5838, 5845, 5837, 2367, 4718, 6238, 1226, 2145, 1431, 5006, 3479, 1288, 2267, 5021, 4778, 1855, 4194, 2246, 193, 157, 1679, 2096, 4373, 4349, 6050, 1686, 1032, 2331, 5672, 586, 3990, 38, 4032, 6534, 4320, 4106, 3863, 3146, 4671, 5648, 627, 5325, 1620, 1892, 1865, 5510, 4789, 5186, 731, 734, 737, 116, 386]

print("\nPython tokens: \(pythonTokens.count) tokens")
let speechTokens = MLXArray(pythonTokens.map { Int32($0) }).expandedDimensions(axis: 0)
eval(speechTokens)

// Load S3Gen
print("\nLoading S3Gen...")
let s3gen = try S3GenMLX.loadFromSafetensors(
    modelDir: modelDir,
    voiceDir: voiceDir,
    device: Device.default
)
print("✅ S3Gen loaded")

// Generate mel only (no vocoder)
print("\nGenerating mel from Python tokens...")
let mel = s3gen.generateMel(
    tokens: speechTokens,
    speakerEmb: s3gen.speakerEmbedding!,
    speechEmbMatrix: s3gen.speechEmbMatrix!,
    promptToken: s3gen.promptToken!,
    promptFeat: s3gen.promptFeat!
)
eval(mel)

print("\n" + String(repeating: "=", count: 80))
print("SWIFT MEL SPECTROGRAM")
print(String(repeating: "=", count: 80))
print("Shape: \(mel.shape)")
print("Range: [\(mel.min().item(Float.self)), \(mel.max().item(Float.self))]")
print("Mean: \(mel.mean().item(Float.self))")

// Check per-channel means
let channels = [0, 10, 20, 40, 60, 79]
for i in channels {
    let channelSlice = mel[0..., i..., 0...]
    let channelMean = channelSlice.mean().item(Float.self)
    print("Channel \(String(format: "%2d", i)) mean: \(String(format: "%7.4f", channelMean))")
}

print(String(repeating: "=", count: 80))
