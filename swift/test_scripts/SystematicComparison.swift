import Foundation
import MLX
import MLXNN
import MLXRandom
import Nightingale

print("=" * 80)
print("SYSTEMATIC SWIFT MEL GENERATION - TRACE ALL STAGES")
print("=" * 80)

// Load S3Gen
print("\nLoading S3Gen...")
let s3gen = try! S3Gen(weightsURL: URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/models/chatterbox/s3gen.safetensors"))
print("✅ S3Gen loaded")

// Test both voices
for voiceName in ["samantha", "sujano"] {
    print("\n" + String(repeating: "=", count: 80))
    print("VOICE: \(voiceName.uppercased())")
    print(String(repeating: "=", count: 80))
    
    // Load voice
    let voicePath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/baked_voices/\(voiceName)/baked_voice.safetensors")
    let voiceData = try! MLX.loadArrays(url: voicePath)
    let speakerEmb = voiceData["speaker_emb"]!
    let speechEmbMatrix = voiceData["speech_emb_matrix"]!
    let promptToken = voiceData["prompt_token"]!
    let promptFeat = voiceData["prompt_feat"]!
    
    // Load Python tokens for comparison
    let tokenPath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/test_audio/cross_validate/python_speech_tokens.safetensors")
    let tokenData = try! MLX.loadArrays(url: tokenPath)
    let tokens = tokenData["speech_tokens"]!
    
    print("\n1️⃣ SPEECH TOKENS:")
    print("   Shape: \(tokens.shape)")
    print("   First 20: \(Array(tokens[0..<20].asArray(Int.self)))")
    print("   Last 20: \(Array(tokens[(tokens.shape[0]-20)...].asArray(Int.self)))")
    
    // Get encoder output
    print("\n2️⃣ ENCODER OUTPUT:")
    let (encoderOut, _) = s3gen.encoder.callAsFunction(tokens)
    eval(encoderOut)
    print("   Shape: \(encoderOut.shape)")
    print("   Range: [\(encoderOut.min().item(Float.self)), \(encoderOut.max().item(Float.self))]")
    print("   Mean: \(encoderOut.mean().item(Float.self)), Std: \(encoderOut.variance().sqrt().item(Float.self))")
    
    // Save encoder output
    let encPath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/test_audio/swift_\(voiceName)_encoder.safetensors")
    try! MLX.save(arrays: ["encoder_out": encoderOut], url: encPath)
    print("   ✅ Saved to: \(encPath.lastPathComponent)")
    
    // Get full mel
    print("\n3️⃣ FINAL MEL SPECTROGRAM:")
    let (mel, _) = s3gen.getEncoderAndFlowOutput(
        tokens: tokens,
        speakerEmb: speakerEmb,
        speechEmbMatrix: speechEmbMatrix,
        promptToken: promptToken,
        promptFeat: promptFeat
    )
    
    eval(mel)
    print("   Shape: \(mel.shape)")
    print("   Range: [\(mel.min().item(Float.self)), \(mel.max().item(Float.self))]")
    print("   Mean: \(mel.mean().item(Float.self)), Std: \(mel.variance().sqrt().item(Float.self))")
    
    // Analyze prompt vs generated
    let L_pm = promptFeat.shape[1]
    if mel.shape[2] > L_pm {
        let promptMel = mel[0..., 0..., 0..<L_pm]
        let genMel = mel[0..., 0..., L_pm...]
        eval(promptMel); eval(genMel)
        
        print("\n   📊 Prompt region (0-\(L_pm)):")
        print("      Range: [\(promptMel.min().item(Float.self)), \(promptMel.max().item(Float.self))]")
        print("      Mean: \(promptMel.mean().item(Float.self))")
        
        print("\n   📊 Generated region (\(L_pm)+):")
        print("      Range: [\(genMel.min().item(Float.self)), \(genMel.max().item(Float.self))]")
        print("      Mean: \(genMel.mean().item(Float.self))")
        
        if genMel.max().item(Float.self) > 0 {
            print("      ⚠️  WARNING: Generated mel has positive values!")
        }
    }
    
    // Save mel
    let melPath = URL(fileURLWithPath: "/Users/a10n/Projects/nightingale_TTS/test_audio/swift_\(voiceName)_mel.safetensors")
    try! MLX.save(arrays: ["mel": mel], url: melPath)
    print("\n   ✅ Saved to: \(melPath.lastPathComponent)")
    
    // Generate audio
    print("\n4️⃣ VOCODER OUTPUT:")
    let audio = s3gen.vocoder(mel)
    eval(audio)
    print("   Shape: \(audio.shape)")
    print("   Range: [\(audio.min().item(Float.self)), \(audio.max().item(Float.self))]")
    print("   Mean: \(audio.mean().item(Float.self)), Std: \(audio.variance().sqrt().item(Float.self))")
}

print("\n" + String(repeating: "=", count: 80))
print("SWIFT TRACING COMPLETE")
print(String(repeating: "=", count: 80))
print("\nCompare Python vs Swift outputs at each stage to find divergence.")
print(String(repeating: "=", count: 80))
