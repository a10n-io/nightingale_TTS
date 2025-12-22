import MLX
import MLXRandom
import Foundation
import Nightingale

print(String(repeating: "=", count: 80))
print("VOCODER SHAPE TEST")
print(String(repeating: "=", count: 80))

// Load S3Gen with vocoder weights
let s3genPath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/models/chatterbox/s3gen.safetensors")
let weights = try! MLX.loadArrays(url: s3genPath)
let s3gen = S3Gen(flowWeights: weights, vocoderWeights: weights)

// Create a test mel spectrogram in the decoder output format
// Decoder outputs [B, 80, T]
let testMel = MLXRandom.normal([1, 80, 100])
eval(testMel)
print("\n🔍 Test Mel Shape (decoder output format): \(testMel.shape)")
print("   Expected: [1, 80, 100] (B, Channels, Time)")

// Enable vocoder debugging
Mel2Wav.debugEnabled = true

print("\n--- Running Vocoder ---")
let audio = s3gen.vocoder(testMel)
eval(audio)
print("\n🔍 Vocoder Output Shape: \(audio.shape)")
print("   Audio range: [\(audio.min().item(Float.self)), \(audio.max().item(Float.self))]")

print("\n" + String(repeating: "=", count: 80))
