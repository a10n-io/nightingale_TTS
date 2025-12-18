import Foundation
import Nightingale
import AVFoundation

// MARK: - Audio Utilities

func saveAsWAV(samples: [Float], sampleRate: Int, to url: URL) {
    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false)!
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
    buffer.frameLength = AVAudioFrameCount(samples.count)

    let channelData = buffer.floatChannelData![0]
    for i in 0..<samples.count {
        channelData[i] = samples[i]
    }

    do {
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
    } catch {
        print("Failed to save WAV: \(error)")
    }
}

func playAudio(samples: [Float], sampleRate: Int) {
    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false)!
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
    buffer.frameLength = AVAudioFrameCount(samples.count)

    let channelData = buffer.floatChannelData![0]
    for i in 0..<samples.count {
        channelData[i] = samples[i]
    }

    let audioEngine = AVAudioEngine()
    let playerNode = AVAudioPlayerNode()
    audioEngine.attach(playerNode)
    audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: format)

    do {
        try audioEngine.start()
        playerNode.scheduleBuffer(buffer, completionHandler: nil)
        playerNode.play()

        // Wait for playback to complete
        let duration = Double(samples.count) / Double(sampleRate)
        Thread.sleep(forTimeInterval: duration + 0.5)

        playerNode.stop()
        audioEngine.stop()
    } catch {
        print("Failed to play audio: \(error)")
    }
}

// MARK: - Main

func runMain() async {
    print(String(repeating: "=", count: 80))
    print("SWIFT AUDIO GENERATION TEST")
    print(String(repeating: "=", count: 80))
    print()

    let modelPath = "/Users/a10n/Projects/nightingale_TTS/models/mlx"
    let voicePath = "/Users/a10n/Projects/nightingale_TTS/baked_voices"
    let outputPath = "/Users/a10n/Projects/nightingale_TTS/test_audio"

    // Text to synthesize
    let text = "Hello, this is a test of the nightingale speech synthesis system."

    print("Text: \"\(text)\"")
    print()

    // Initialize engine
    print("Initializing ChatterboxEngine...")
    let engine = ChatterboxEngine()

    do {
        // Load models
        print("Loading models from: \(modelPath)")
        let modelsURL = URL(fileURLWithPath: modelPath)
        try await engine.loadModels(modelsURL: modelsURL)
        print("Models loaded!")
        print()

        // Load voice
        print("Loading voice: baked_voice_npy")
        let voicesURL = URL(fileURLWithPath: voicePath)
        try await engine.loadVoice("baked_voice_npy", voicesURL: voicesURL)
        print("Voice loaded!")
        print()

        // Generate audio
        print(String(repeating: "=", count: 80))
        print("GENERATING AUDIO")
        print(String(repeating: "=", count: 80))
        print()

        let startTime = Date()
        let audioSamples = try await engine.generateAudio(text, temperature: 0.4)
        let elapsed = Date().timeIntervalSince(startTime)

        print()
        print(String(repeating: "=", count: 80))
        print("GENERATION COMPLETE")
        print(String(repeating: "=", count: 80))
        print()
        print("Generated \(audioSamples.count) samples in \(String(format: "%.2f", elapsed))s")
        print("Duration: \(String(format: "%.2f", Float(audioSamples.count) / 24000.0))s @ 24000 Hz")
        print()

        // Save as WAV
        let outputURL = URL(fileURLWithPath: outputPath).appendingPathComponent("swift_generated.wav")
        saveAsWAV(samples: audioSamples, sampleRate: 24000, to: outputURL)
        print("Saved to: \(outputURL.path)")
        print()

        // Play audio
        print("Playing audio...")
        playAudio(samples: audioSamples, sampleRate: 24000)

    } catch {
        print("ERROR: \(error)")
    }
}

// Entry point - use RunLoop to avoid semaphore deadlock with async/await
var finished = false
Task {
    await runMain()
    finished = true
}

// Pump the run loop until the task completes
while !finished {
    RunLoop.current.run(mode: .default, before: Date(timeIntervalSinceNow: 0.1))
}
