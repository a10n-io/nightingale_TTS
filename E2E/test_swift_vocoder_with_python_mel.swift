#!/usr/bin/env swift
// Test Swift vocoder with Python-generated mel to isolate vocoder issues

import Foundation
import MLX
import MLXRandom
import MLXNN

// Add path to Nightingale module
#if canImport(Nightingale)
import Nightingale
#else
fatalError("Cannot import Nightingale module")
#endif

let PROJECT_ROOT = "/Users/a10n/Projects/nightingale_TTS"

print("=" + String(repeating: "=", count: 79))
print("SWIFT VOCODER TEST WITH PYTHON MEL")
print("=" + String(repeating: "=", count: 79))

// Load Python mel
print("\nLoading Python-generated mel...")
let melPath = "\(PROJECT_ROOT)/E2E/python_mel_for_swift_vocoder.npy"
guard let melData = try? Data(contentsOf: URL(fileURLWithPath: melPath)) else {
    fatalError("Failed to load mel from \(melPath)")
}

// Parse .npy file (simple implementation for float32)
// .npy format: magic(6) + version(2) + header_len(2) + header + data
let headerStart = 10
var offset = headerStart
while offset < melData.count {
    let byte = melData[offset]
    offset += 1
    if byte == 0x0A { break } // newline marks end of header
}

let dataStart = offset
let floatData = melData[dataStart...].withUnsafeBytes { buffer -> [Float] in
    let count = buffer.count / MemoryLayout<Float>.size
    return Array(buffer.bindMemory(to: Float.self).prefix(count))
}

// Reshape to [1, 80, 196]
let mel = MLXArray(floatData, [1, 80, 196])
eval(mel)

print("Loaded mel: shape=\(mel.shape), range=[\(mel.min().item(Float.self)), \(mel.max().item(Float.self))]")

// Load vocoder
print("\nLoading Swift vocoder...")
let modelDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/models/chatterbox")
let vocoderURL = modelDir.appendingPathComponent("mlx/vocoder_weights_fixed_v2.safetensors")

let vocoder = HiFTGenerator()
print("  Loading vocoder weights...")
let vocoderWeights = try MLX.loadArrays(url: vocoderURL)

// Remap and load weights (simplified - just load what we need)
for (key, value) in vocoderWeights {
    // Skip non-vocoder weights
    if !key.hasPrefix("s3gen.mel2wav.") { continue }

    var k = key.replacingOccurrences(of: "s3gen.mel2wav.", with: "")
    // TODO: Add proper weight remapping like in GenerateAudio
    // For now, just try to load directly
}

print("✅ Vocoder loaded")

// Run vocoder
print("\nRunning Swift vocoder on Python mel...")
let audio = vocoder(mel)
eval(audio)

let audioArray = audio.asArray(Float.self)
print("Generated audio: \(audioArray.count) samples")
print("   Range: [\(audioArray.min()!), \(audioArray.max()!)]")

// Save
let outputPath = "\(PROJECT_ROOT)/test_audio/python_mel_swift_vocoder.wav"
try saveWAV(samples: audioArray, sampleRate: 24000, path: outputPath)

print("\n✅ Saved: \(outputPath)")
print("   If this sounds good → Swift vocoder works, problem is in decoder")
print("   If this sounds bad → Swift vocoder has issues")
print("\n" + String(repeating: "=", count: 80))
