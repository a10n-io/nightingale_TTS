import Foundation
import MLX
import MLXNN
import Nightingale

print(String(repeating: "=", count: 80))
print("VOCODER COMPARISON TEST: Python vs Swift")
print(String(repeating: "=", count: 80))

let PROJECT_ROOT = "/Users/a10n/Projects/nightingale_TTS"

// Load mel spectrogram
let melPath = URL(fileURLWithPath: "\(PROJECT_ROOT)/test_audio/test_mel.safetensors")
print("\nLoading mel from: \(melPath.path)")

let melData = try! MLX.loadArrays(url: melPath)
let mel = melData["mel"]!  // [1, 80, 248]

print("Mel shape: \(mel.shape)")
print("Mel range: [\(mel.min().item(Float.self)), \(mel.max().item(Float.self))]")
print("Mel mean: \(mel.mean().item(Float.self))")

// Load S3Gen to get vocoder
print("\nLoading S3Gen...")
let modelDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/models/chatterbox")
let s3genPath = modelDir.appendingPathComponent("s3gen.safetensors")
let weights = try! MLX.loadArrays(url: s3genPath)
print("✅ Loaded \(weights.count) weight arrays")

// Initialize S3Gen (which initializes vocoder with weights)
let s3gen = S3Gen(flowWeights: weights, vocoderWeights: weights)
print("✅ S3Gen initialized")

// Run vocoder
print("\nRunning Swift vocoder...")
let audio = s3gen.vocoder(mel)  // [1, T_audio]
eval(audio)

print("Audio shape: \(audio.shape)")
print("Audio range: [\(audio.min().item(Float.self)), \(audio.max().item(Float.self))]")
print("Audio mean: \(audio.mean().item(Float.self))")

// Calculate RMS
let audioSquared = audio * audio
let rms = sqrt(audioSquared.mean()).item(Float.self)
print("Audio RMS: \(rms)")

// Save audio
let audioSamples = audio.squeezed().asArray(Float.self)
print("\n✅ Generated \(audioSamples.count) samples")
print("Duration: \(String(format: "%.2f", Float(audioSamples.count) / 24000.0))s")

// Write WAV file
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

let outputPath = URL(fileURLWithPath: "\(PROJECT_ROOT)/test_audio/swift_vocoder_test.wav")
try writeWAV(audio: audioSamples, sampleRate: 24000, to: outputPath)
print("✅ Saved audio to: \(outputPath.path)")

print("\n" + String(repeating: "=", count: 80))
print("✅ TEST COMPLETE!")
print(String(repeating: "=", count: 80))
