import Foundation
import MLX

print("Testing NPY loading...")

let path = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/test_audio/decoder_trace_npy/t.npy")
print("Loading from: \(path.path)")
print("File exists: \(FileManager.default.fileExists(atPath: path.path))")

do {
    let arr = try MLX.loadArray(url: path)
    print("✅ Loaded successfully!")
    print("Shape: \(arr.shape)")
    print("Value: \(arr.item(Float.self))")
} catch {
    print("❌ Error: \(error)")
}
