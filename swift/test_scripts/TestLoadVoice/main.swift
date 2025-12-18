import Foundation
import MLX
import Nightingale

print(String(repeating: "=", count: 80))
print("NIGHTINGALE - TEST PREBAKED VOICE LOADING")
print(String(repeating: "=", count: 80))
print()

// Use absolute path to baked voices
let voicePath = "/Users/a10n/Projects/nightingale/baked_voices/samantha_full"
print("Voice path: \(voicePath)")
print()

// Test loading each prebaked voice file
let files = [
    "soul_t3_256.npy",     // T3 speaker embedding (The Actor)
    "soul_s3_192.npy",     // S3Gen speaker embedding (The Instrument)
    "t3_cond_tokens.npy",  // T3 conditioning tokens
    "prompt_token.npy",    // S3Gen prompt tokens
    "prompt_feat.npy"      // S3Gen prompt mel features
]

var success = true

for filename in files {
    let filepath = voicePath + "/" + filename
    let fileURL = URL(fileURLWithPath: filepath)

    do {
        print("Loading: \(filename)...")

        let data = try NPYLoader.load(contentsOf: fileURL)

        print("✅ Loaded successfully!")
        print("   Shape: \(data.shape)")
        print("   Dtype: \(data.dtype)")
        print("   Size: \(data.size) elements")

        // Show some basic stats
        let flatData = data.reshaped([-1])
        let minVal = flatData.min().item(Float.self)
        let maxVal = flatData.max().item(Float.self)
        let meanVal = flatData.mean().item(Float.self)

        print("   Range: [\(minVal), \(maxVal)]")
        print("   Mean: \(meanVal)")
        print()

    } catch {
        print("❌ Failed to load \(filename)")
        print("   Error: \(error)")
        print()
        success = false
    }
}

print(String(repeating: "=", count: 80))
if success {
    print("✅ ALL VOICE FILES LOADED SUCCESSFULLY")
    print(String(repeating: "=", count: 80))
    print()
    print("The prebaked voice is ready to use!")
    print()
    print("Voice components:")
    print("  soul_t3_256.npy  → T3 speaker embedding (The Actor)")
    print("  soul_s3_192.npy  → S3Gen speaker embedding (The Instrument)")
    print("  t3_cond_tokens.npy → T3 conditioning tokens")
    print("  prompt_token.npy → S3Gen prompt tokens")
    print("  prompt_feat.npy  → S3Gen prompt mel features")
} else {
    print("❌ SOME FILES FAILED TO LOAD")
    print(String(repeating: "=", count: 80))
    exit(1)
}
