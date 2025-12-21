import Foundation
import MLX
import MLXNN
import Nightingale

// MARK: - Main

print("=" * 60)
print("SWIFT VOCODER TEST")
print("=" * 60)

let PROJECT_ROOT = "/Users/a10n/Projects/nightingale_TTS"
let modelDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/models")
let refDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/E2E/reference_outputs/samantha/expressive_surprise_en")

// Load test mel from Python (BCT format - channels second)
let melBCT = try NPYLoader.load(contentsOf: refDir.appendingPathComponent("test_mel_BCT.npy"))
print("Loaded test mel (BCT): \(melBCT.shape)")
print("  Range: [\(melBCT.min().item(Float.self)), \(melBCT.max().item(Float.self))]")
print("  Bin 0 (low freq) mean: \(melBCT[0, 0, 0...].mean().item(Float.self))")
print("  Bin 79 (high freq) mean: \(melBCT[0, 79, 0...].mean().item(Float.self))")

// Load vocoder weights
// NOTE: For loading directly into Mel2Wav (not through S3Gen), we don't need "vocoder." prefix
func remapVocoderKey(_ key: String) -> String? {
    var k = key
    // Basic renames (no prefix needed when loading directly into Mel2Wav)
    k = k.replacingOccurrences(of: "conv_pre", with: "convPre")
    k = k.replacingOccurrences(of: "conv_post", with: "convPost")
    k = k.replacingOccurrences(of: "activations1", with: "acts1")
    k = k.replacingOccurrences(of: "activations2", with: "acts2")

    if k.contains("f0_predictor.") {
        k = k.replacingOccurrences(of: "f0_predictor.condnet.0.", with: "f0Predictor.convs.0.")
        k = k.replacingOccurrences(of: "f0_predictor.condnet.2.", with: "f0Predictor.convs.1.")
        k = k.replacingOccurrences(of: "f0_predictor.condnet.4.", with: "f0Predictor.convs.2.")
        k = k.replacingOccurrences(of: "f0_predictor.condnet.6.", with: "f0Predictor.convs.3.")
        k = k.replacingOccurrences(of: "f0_predictor.condnet.8.", with: "f0Predictor.convs.4.")
        k = k.replacingOccurrences(of: "f0_predictor.classifier.", with: "f0Predictor.classifier.")
        return k
    }
    if k.contains("m_source.") {
        k = k.replacingOccurrences(of: "m_source.l_linear.", with: "mSource.linear.")
        return k
    }
    if k.contains("source_downs.") {
        k = k.replacingOccurrences(of: "source_downs.", with: "sourceDowns.")
        return k
    }
    if k.contains("source_resblocks.") {
        k = k.replacingOccurrences(of: "source_resblocks.", with: "sourceResBlocks.")
        return k
    }
    return k
}

func remapVocoderWeights(_ weights: [String: MLXArray], transposeConv1d: Bool = false) -> [String: MLXArray] {
    var remapped: [String: MLXArray] = [:]
    for (key, value) in weights {
        if let newKey = remapVocoderKey(key) {
            // Debug: Show remapping for m_source
            if key.contains("m_source") {
                print("DEBUG: Remapped '\(key)' -> '\(newKey)' (shape \(value.shape))")
            }
            let isEmbedding = newKey.contains("Embedding.weight") || newKey.contains("speechEmb.weight")
            let isLinear = newKey.hasSuffix(".weight") && value.ndim == 2 && !isEmbedding
            let isConv1d = newKey.hasSuffix(".weight") && value.ndim == 3
            if isLinear {
                remapped[newKey] = value.T
                if key.contains("m_source") {
                    let transposed = value.T
                    eval(transposed)
                    print("DEBUG: After transpose: shape=\(transposed.shape), values=\(transposed.flattened().asArray(Float.self))")
                }
            } else if isConv1d && transposeConv1d {
                let isConvTranspose = newKey.contains("ups.") && newKey.hasSuffix(".weight")
                remapped[newKey] = isConvTranspose ? value.transposed(1, 2, 0) : value.transposed(0, 2, 1)
            } else {
                remapped[newKey] = value
                if key.contains("m_source") {
                    print("DEBUG: No transpose (bias): values=\(value.flattened().asArray(Float.self))")
                }
            }
        }
    }
    // Check if mSource keys are in remapped
    for key in remapped.keys where key.contains("mSource") {
        print("DEBUG: mSource key in remapped: '\(key)'")
    }
    return remapped
}

print("\nLoading vocoder...")
let vocoderURL = modelDir.appendingPathComponent("mlx/vocoder_weights.safetensors")
let vocoderWeights = try MLX.loadArrays(url: vocoderURL)

// Create vocoder directly
let vocoder = Mel2Wav()
let remappedWeights = remapVocoderWeights(vocoderWeights, transposeConv1d: true)
print("Updating vocoder with \(remappedWeights.count) weights...")
vocoder.update(parameters: ModuleParameters.unflattened(remappedWeights))
eval(vocoder)
print("Vocoder loaded")

// Enable debugging
Mel2Wav.debugEnabled = true

// Check weight shapes before running
print("\nChecking convPre weight shape...")
let convPreWeight = vocoder.convPre.weight
eval(convPreWeight)
print("convPre.weight shape: \(convPreWeight.shape)")
print("  Expected for MLX Conv1d: [out_channels, kernel_size, in_channels]")
print("  If from PyTorch [O, I, K], should be transposed to [O, K, I]")

// Run vocoder on test mel
print("\nRunning Swift vocoder...")
let wav = vocoder(melBCT)
eval(wav)

print("Swift vocoder output: \(wav.shape)")
let wavFlat = wav.squeezed()
eval(wavFlat)
print("  Range: [\(wavFlat.min().item(Float.self)), \(wavFlat.max().item(Float.self))]")

// Analyze frequency content
let wavArray = wavFlat.asArray(Float.self)
print("  Samples: \(wavArray.count)")

// Simple frequency analysis - check energy distribution
// For 24kHz sample rate, bin width = 24000 / N
let n = wavArray.count
var realSum: Float = 0
var imagSum: Float = 0

// Compute simple DFT for low frequency (100 Hz)
let freq: Float = 100.0
let sr: Float = 24000.0
for (i, sample) in wavArray.enumerated() {
    let angle = 2.0 * Float.pi * freq * Float(i) / sr
    realSum += sample * cos(angle)
    imagSum += sample * sin(angle)
}
let energy100Hz = sqrt(realSum * realSum + imagSum * imagSum) / Float(n)

// Compute for high frequency (10000 Hz)
realSum = 0
imagSum = 0
for (i, sample) in wavArray.enumerated() {
    let angle = 2.0 * Float.pi * 10000.0 * Float(i) / sr
    realSum += sample * cos(angle)
    imagSum += sample * sin(angle)
}
let energy10kHz = sqrt(realSum * realSum + imagSum * imagSum) / Float(n)

print("\nFrequency analysis:")
print("  Energy at 100 Hz: \(energy100Hz)")
print("  Energy at 10 kHz: \(energy10kHz)")

if energy100Hz > energy10kHz {
    print("  -> Low frequency has more energy (CORRECT for speech)")
} else {
    print("  -> High frequency has more energy (FREQUENCY INVERSION!)")
}

// Save output for comparison
print("\nSaving output...")
// Simple WAV writing
var data = Data()
let numChannels: UInt16 = 1
let bitsPerSample: UInt16 = 16
let byteRate = UInt32(24000 * Int(numChannels) * Int(bitsPerSample) / 8)
let blockAlign = UInt16(numChannels * bitsPerSample / 8)
let dataSize = UInt32(wavArray.count * Int(bitsPerSample) / 8)
let fileSize = 36 + dataSize

data.append(contentsOf: "RIFF".utf8)
data.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
data.append(contentsOf: "WAVE".utf8)
data.append(contentsOf: "fmt ".utf8)
data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })
data.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
data.append(contentsOf: withUnsafeBytes(of: UInt32(24000).littleEndian) { Array($0) })
data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
data.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })
data.append(contentsOf: "data".utf8)
data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

for sample in wavArray {
    let scaled = Int16(max(-1.0, min(1.0, sample)) * 32767.0)
    data.append(contentsOf: withUnsafeBytes(of: scaled.littleEndian) { Array($0) })
}

let outputPath = refDir.appendingPathComponent("test_vocoder_swift.wav")
try data.write(to: outputPath)
print("Saved: \(outputPath.path)")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

func *(_ s: String, _ n: Int) -> String {
    return String(repeating: s, count: n)
}
