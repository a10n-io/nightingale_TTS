import Foundation
import MLX
import MLXNN
import MLXRandom
import Nightingale
import ArgumentParser

// MARK: - WAV Writer

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

// MARK: - Async Main Runner

func runAsync() async throws {
    // Parse command line args manually
    guard CommandLine.arguments.count >= 3,
          CommandLine.arguments[1] == "--config" else {
        print("Usage: GenerateAudioE2E --config <path-to-config.json>")
        throw NSError(domain: "GenerateAudioE2E", code: 1)
    }

    let configPath = CommandLine.arguments[2]
    let PROJECT_ROOT = "/Users/a10n/Projects/nightingale_TTS"

    // Load config
    let configURL = URL(fileURLWithPath: configPath)
    let configData = try Data(contentsOf: configURL)
    let configDict = try JSONSerialization.jsonObject(with: configData) as! [String: Any]

    let text = configDict["text"] as! String
    let voice = configDict["voice"] as! String
    let language = configDict["language"] as! String
    let outputDir = URL(fileURLWithPath: configDict["output_dir"] as! String)
    let seed = configDict["seed"] as! Int
    let temperature = configDict["temperature"] as! Double
    let cfgWeight = configDict["cfg_weight"] as! Double
    let repetitionPenalty = configDict["repetition_penalty"] as! Double
    let topP = configDict["top_p"] as! Double
    let minP = configDict["min_p"] as! Double

    print(String(repeating: "=", count: 60))
    print("SWIFT PIPELINE (Steps 1-9) using ChatterboxEngine")
    print(String(repeating: "=", count: 60))

    // Set deterministic seed
    MLXRandom.seed(UInt64(seed))

    let modelDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/models/mlx")
    let voicesDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/baked_voices")

    // Initialize ChatterboxEngine
    print("\nLoading ChatterboxEngine...")
    let engine = ChatterboxEngine()

    // Load models
    try await engine.loadModels(modelsURL: modelDir)
    print("  Models loaded")

    // Load voice
    try await engine.loadVoice(voice, voicesURL: voicesDir)
    print("  Voice loaded: \(voice)")

    print("\n  Language: \(language)")
    print("  Text: \"\(text)\"")

    // =====================================================================
    // Step 1: Tokenization (using ChatterboxEngine's BPE tokenizer)
    // =====================================================================
    print("\n[Step 1] Tokenization")
    let textTokens = try await engine.tokenizeText(text)
    eval(textTokens)
    let tokenCount = textTokens.shape[1]
    print("  Tokens: \(tokenCount)")

    // Save Step 1 outputs
    try textTokens.squeezed().save(npy: outputDir.appendingPathComponent("step1_text_tokens.npy"))

    // =====================================================================
    // Steps 2-9: Full generation using ChatterboxEngine
    // =====================================================================
    print("\n[Steps 2-9] Full Generation Pipeline")
    print("  (Note: Using temperature=\(temperature), other params controlled internally)")

    // Generate audio using the engine
    // Note: ChatterboxEngine.generateAudio only exposes temperature parameter
    // Other parameters (cfgWeight, repetitionPenalty, topP, minP) are internal to T3Model
    let audioSamples = try await engine.generateAudio(text, temperature: Float(temperature))
    print("  Audio samples: \(audioSamples.count)")
    print("  Duration: \(Float(audioSamples.count) / 24000.0)s")

    // Save Step 8 output (audio)
    let audioArray = MLXArray(audioSamples).expandedDimensions(axis: 0)
    try audioArray.save(npy: outputDir.appendingPathComponent("step8_audio.npy"))

    // Save S3Gen conditioning (same as baked voice data)
    let s3VoiceDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/baked_voices/\(voice)/npy_original")
    let s3Embedding = try NPYLoader.load(contentsOf: s3VoiceDir.appendingPathComponent("soul_s3_192.npy"))
    try s3Embedding.save(npy: outputDir.appendingPathComponent("step4_s3_embedding.npy"))

    // =====================================================================
    // Step 9: Save WAV
    // =====================================================================
    print("\n[Step 9] Save WAV")
    let wavPath = outputDir.appendingPathComponent("swift_output.wav")
    try writeWAV(audio: audioSamples, sampleRate: 24000, to: wavPath)
    print("  Saved: \(wavPath.path)")

    print("\n" + String(repeating: "=", count: 60))
    print("SWIFT PIPELINE COMPLETE")
    print(String(repeating: "=", count: 60))
}

// MARK: - Main Entry Point

@main
struct GenerateAudioE2E {
    static func main() async {
        do {
            try await runAsync()
        } catch {
            print("Error: \(error)")
            Darwin.exit(1)
        }
    }
}
