import Foundation
import MLX

print("Starting test...")

let tracePath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/test_audio/python_decoder_trace.safetensors")
print("Loading from: \(tracePath.path)")

do {
    let trace = try MLX.loadArrays(url: tracePath)
    print("✅ Loaded \(trace.count) arrays")
    for (key, value) in trace {
        print("  \(key): \(value.shape)")
    }
} catch {
    print("❌ Error: \(error)")
}

print("Done.")
