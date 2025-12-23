import MLX
import MLXRandom
import Foundation
import Nightingale

print(String(repeating: "=", count: 80))
print("VOCODER SHAPE TEST - Using Python Mel")
print(String(repeating: "=", count: 80))

// Load S3Gen with vocoder weights
let s3genPath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/models/chatterbox/s3gen.safetensors")
let weights = try! MLX.loadArrays(url: s3genPath)
let s3gen = S3Gen(flowWeights: weights, vocoderWeights: weights)

// Load Python mel that we know is correct
let pythonMelPath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/test_audio/cross_validate/python_mel.safetensors")
let melData = try! MLX.loadArrays(url: pythonMelPath)
let testMel = melData["mel"]!
eval(testMel)
print("\n🔍 Python Mel loaded:")
print("   Shape: \(testMel.shape)")
print("   Range: [\(testMel.min().item(Float.self)), \(testMel.max().item(Float.self))]")
print("   Mean: \(testMel.mean().item(Float.self)), Std: \(testMel.variance().sqrt().item(Float.self))")

// Enable vocoder debugging
Mel2Wav.debugEnabled = true

print("\n--- Running Vocoder ---")
let audio = s3gen.vocoder(testMel)
eval(audio)
print("\n🔍 Vocoder Output Shape: \(audio.shape)")
eval(audio)
let audioMin = audio.min().item(Float.self)
let audioMax = audio.max().item(Float.self)
let audioMean = audio.mean().item(Float.self)
let audioStd = audio.variance().sqrt().item(Float.self)
print("   Audio range: [\(audioMin), \(audioMax)]")
print("   Audio mean: \(audioMean), std: \(audioStd)")

// Print first 20 samples for comparison
let first20 = Array(audio[0, 0..<min(20, audio.shape[1])].asArray(Float.self))
print("   First 20 samples: \(first20)")

// Save audio for comparison
func writeWAV(audio: [Float], sampleRate: Int, to url: URL) throws {
    var data = Data()
    let numChannels: UInt16 = 1
    let bitsPerSample: UInt16 = 16
    let byteRate = UInt32(sampleRate * Int(numChannels) * Int(bitsPerSample) / 8)
    let blockAlign = UInt16(numChannels * bitsPerSample / 8)
    let dataSize = UInt32(audio.count * Int(bitsPerSample) / 8)
    let fileSize = 36 + dataSize

    data.append(contentsOf: "RIFF".utf8)
    data.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
    data.append(contentsOf: "WAVE".utf8)
    data.append(contentsOf: "fmt ".utf8)
    data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })
    data.append(contentsOf: "data".utf8)
    data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

    for sample in audio {
        let scaled = Int16(max(-1.0, min(1.0, sample)) * 32767.0)
        data.append(contentsOf: withUnsafeBytes(of: scaled.littleEndian) { Array($0) })
    }
    try data.write(to: url)
}

let audioSamples = audio.squeezed().asArray(Float.self)
let outputPath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/test_audio/cross_validate/python_mel_swift_vocoder.wav")
try writeWAV(audio: audioSamples, sampleRate: 24000, to: outputPath)
print("\n✅ Saved Swift vocoder audio to: \(outputPath.path)")

print("\n" + String(repeating: "=", count: 80))
