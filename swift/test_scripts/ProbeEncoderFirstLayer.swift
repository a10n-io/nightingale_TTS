import Foundation
import MLX
import Nightingale

print("=" + String(repeating: "=", count: 79))
print("FIRST LAYER PROBE - Swift Encoder")
print("=" + String(repeating: "=", count: 79))

// Load model
let modelsPath = "../models/chatterbox"
let s3gen = try! S3Gen(weightsFolder: modelsPath)

// Load voice
let voiceDir = "../baked_voices/samantha"
let voiceArrays = try! MLX.loadArrays(url: URL(fileURLWithPath: voiceDir + "/baked_voice.safetensors"))
let speakerEmb = voiceArrays["t3.speaker_emb"]!
let speechEmbMatrix = voiceArrays["gen.embedding"]!
let promptToken = voiceArrays["gen.prompt_token"]!
let promptFeat = voiceArrays["gen.prompt_feat"]!

// Load Python tokens
let tokensPath = "../test_audio/cross_validate/python_speech_tokens.safetensors"
let tokensArrays = try! MLX.loadArrays(url: URL(fileURLWithPath: tokensPath))
let speechTokens = tokensArrays["speech_tokens"]!

print("\n📊 Inputs:")
print("   tokens: \(speechTokens.shape)")
print("   speechEmbMatrix: \(speechEmbMatrix.shape)")

// ==============================================================================
// CHECKPOINT 1: speechEmbMatrix (raw weights)
// ==============================================================================
print("\n" + String(repeating: "=", count: 80))
print("CHECKPOINT 1: speechEmbMatrix (raw weights)")
print(String(repeating: "=", count: 80))
eval(speechEmbMatrix)
print("   Shape: \(speechEmbMatrix.shape)")
let embMin = speechEmbMatrix.min()
let embMax = speechEmbMatrix.max()
let embMean = speechEmbMatrix.mean()
let embStd = speechEmbMatrix.variance().sqrt()
eval(embMin, embMax, embMean, embStd)
print("   Range: [\(embMin.item(Float.self)), \(embMax.item(Float.self))]")
print("   Mean: \(embMean.item(Float.self))")
print("   Std: \(embStd.item(Float.self))")

print("\n   First 10 values [0, 0, 0:10]:")
let first10 = speechEmbMatrix[0, 0, 0..<10]
eval(first10)
let first10Arr = (0..<10).map { first10[$0].item(Float.self) }
print("   \(first10Arr)")

print("\n   Sample from middle [0, 256, 0:10]:")
let middle10 = speechEmbMatrix[0, 256, 0..<10]
eval(middle10)
let middle10Arr = (0..<10).map { middle10[$0].item(Float.self) }
print("   \(middle10Arr)")

// ==============================================================================
// CHECKPOINT 2 & 3: Run encoder and capture intermediate outputs
// ==============================================================================
// We need to modify the encoder to capture outputs after embed and after pos encoding
// For now, let's just run it and manually extract from the flow

// The encoder in Swift is part of S3Gen
// Let me access it directly to hook into it

// Actually, let me just duplicate the encoder forward pass here and capture outputs

print("\n" + String(repeating: "=", count: 80))
print("Running encoder manually to capture intermediate outputs...")
print(String(repeating: "=", count: 80))

// Access the encoder
let encoder = s3gen.encoder

// Prepare token input - we need to do the embedding lookup
// tokens: [T] -> need to gather from speechEmbMatrix

// First, concatenate prompt_token and generated tokens
let tokenFull = concatenated([promptToken.squeezed(axis: 0), speechTokens], axis: 0)
print("\n   Full token sequence: \(tokenFull.shape)")

// Embedding lookup: gather from speechEmbMatrix
// speechEmbMatrix is [1, vocab_size?, dim]
// We need to gather using tokenFull indices
// tokenFull is [T]
// Result should be [T, dim]

let vocabSize = speechEmbMatrix.shape[1]
let embDim = speechEmbMatrix.shape[2]

// Gather embeddings
// speechEmbMatrix[0, tokenFull, :] in numpy notation
let embeddedAfterLookup = take(speechEmbMatrix.squeezed(axis: 0), tokenFull.asType(.int32), axis: 0)
print("   After embedding lookup: \(embeddedAfterLookup.shape)")
eval(embeddedAfterLookup)

let emb2Min = embeddedAfterLookup.min()
let emb2Max = embeddedAfterLookup.max()
let emb2Mean = embeddedAfterLookup.mean()
let emb2Std = embeddedAfterLookup.variance().sqrt()
eval(emb2Min, emb2Max, emb2Mean, emb2Std)

print("\n" + String(repeating: "=", count: 80))
print("CHECKPOINT 2: After embedding lookup")
print(String(repeating: "=", count: 80))
print("   Shape: \(embeddedAfterLookup.shape)")
print("   Range: [\(emb2Min.item(Float.self)), \(emb2Max.item(Float.self))]")
print("   Mean: \(emb2Mean.item(Float.self))")
print("   Std: \(emb2Std.item(Float.self))")

print("\n   First 10 values [0, 0:10]:")
let emb2First10 = embeddedAfterLookup[0, 0..<10]
eval(emb2First10)
let emb2First10Arr = (0..<10).map { emb2First10[$0].item(Float.self) }
print("   \(emb2First10Arr)")

if embeddedAfterLookup.shape[0] > 500 {
    print("\n   Position 500 [500, 0:10]:")
    let emb2Pos500 = embeddedAfterLookup[500, 0..<10]
    eval(emb2Pos500)
    let emb2Pos500Arr = (0..<10).map { emb2Pos500[$0].item(Float.self) }
    print("   \(emb2Pos500Arr)")
}

// ==============================================================================
// CHECKPOINT 3: After positional encoding
// ==============================================================================
// This is trickier - we need to run through the encoder's embed layer
// which includes the linear projection and positional encoding

// For Swift, the encoder forward does:
// 1. embed.linear (linear projection)
// 2. embed.norm (layer norm)
// 3. pos_enc (positional encoding addition)

// Let me run the full encoder and capture the output after the first few operations
// Actually, let me just capture the input to the first encoder layer

// Run the encoder to get the full output, and we'll manually trace through
// the first layer operations later if needed

print("\n" + String(repeating: "=", count: 80))
print("CHECKPOINT 3: After full encoder forward (for reference)")
print(String(repeating: "=", count: 80))

// Run full encoder
var h = embeddedAfterLookup.expandedDimensions(axis: 0)  // Add batch dim [1, T, dim]
print("   Input to encoder (with batch): \(h.shape)")

// For now, let's just report what we have
// To get checkpoint 3 properly, we'd need to modify the encoder code
// But for the first layer probe, checkpoint 2 (after embedding lookup) is most critical

print("\n" + String(repeating: "=", count: 80))
print("Saving checkpoints...")
print(String(repeating: "=", count: 80))

let forensicDir = "../test_audio/forensic"
try? FileManager.default.createDirectory(atPath: forensicDir, withIntermediateDirectories: true)

// Save checkpoints
// Note: checkpoint2 needs batch dimension removed to match Python format
let checkpoint2Squeezed = embeddedAfterLookup.expandedDimensions(axis: 0)

try? MLX.save(
    arrays: [
        "checkpoint1_speech_emb_matrix": speechEmbMatrix,
        "checkpoint2_after_embed": checkpoint2Squeezed,
    ],
    url: URL(fileURLWithPath: forensicDir + "/swift_encoder_first_layer_probe.safetensors")
)

print("\n✅ Saved to: \(forensicDir)/swift_encoder_first_layer_probe.safetensors")
print("\n" + String(repeating: "=", count: 80))
