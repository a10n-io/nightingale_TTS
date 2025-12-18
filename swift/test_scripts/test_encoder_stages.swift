#!/usr/bin/env swift

import Foundation
import MLX
import MLXNN

// Add Sources to module search path
let packageRoot = URL(fileURLWithPath: #file)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .path
let sourcesPath = "\(packageRoot)/Sources"

// Import must be in a separate line after path configuration
#if canImport(Nightingale)
import Nightingale
#endif

print("=" + String(repeating: "=", count: 79))
print("SWIFT ENCODER - STAGE-BY-STAGE ANALYSIS")
print("=" + String(repeating: "=", count: 79))
print()

// Load saved token embeddings from PyTorch
let tokenEmbPath = "/Users/a10n/Projects/nightingale/verification_outputs/step5/token_emb.npy"
guard let tokenEmbData = readNpy(path: tokenEmbPath) else {
    fatalError("Failed to load token_emb.npy")
}

print("Input: token_emb.npy")
print("  Shape: \(tokenEmbData.shape)")
print()

// Load encoder weights
let weightsPath = "/Users/a10n/Projects/nightingale/models/chatterbox_hf.safetensors"
let weightsDict = loadSafetensors(path: weightsPath)

// Create encoder
let encoder = FlowEncoder(hiddenDim: 512, melDim: 80, numHeads: 8, weights: weightsDict)

print()
print("Running encoder stages...")
print()

// Stage 1: Initial embedding
let (hAfterEmbed, posEmb) = encoder.embed(tokenEmbData)
eval(hAfterEmbed, posEmb)
print("Stage 1 - After embed:")
print("  Shape: \(hAfterEmbed.shape)")
print("  posEmb: \(posEmb.shape)")
print("  Mean: \(String(format: "%.6f", Float(hAfterEmbed.mean().item())))")
print("  Std: \(String(format: "%.6f", Float(hAfterEmbed.variance().sqrt().item())))")
print()

// Stage 2: Main encoder blocks (6)
var hCurrent = hAfterEmbed
for (i, encoder) in encoder.encoders.enumerated() {
    hCurrent = encoder(hCurrent, posEmb: posEmb)
}
eval(hCurrent)
print("Stage 2 - After 6 encoder blocks:")
print("  Shape: \(hCurrent.shape)")
print("  Mean: \(String(format: "%.6f", Float(hCurrent.mean().item())))")
print("  Std: \(String(format: "%.6f", Float(hCurrent.variance().sqrt().item())))")
print()

// Stage 3: Pre-lookahead
hCurrent = encoder.preLookaheadLayer(hCurrent)
eval(hCurrent)
print("Stage 3 - After pre_lookahead_layer:")
print("  Shape: \(hCurrent.shape)")
print("  Mean: \(String(format: "%.6f", Float(hCurrent.mean().item())))")
print("  Std: \(String(format: "%.6f", Float(hCurrent.variance().sqrt().item())))")
print()

// Stage 4: 2x upsampling (tiling)
let B = hCurrent.shape[0]
let L = hCurrent.shape[1]
let C = hCurrent.shape[2]
var hUpsampled = hCurrent.expandedDimensions(axis: 2)  // [B, L, 1, C]
hUpsampled = tiled(hUpsampled, repetitions: [1, 1, 2, 1])  // [B, L, 2, C]
hUpsampled = hUpsampled.reshaped([B, L * 2, C])  // [B, L*2, C]
eval(hUpsampled)
print("Stage 4 - After 2x tiling (before up_embed):")
print("  Before upsampling: L=\(L)")
print("  After upsampling: L*2=\(L*2)")
print("  Shape: \(hUpsampled.shape)")
print()

// Stage 5: Up embedding
let (hAfterUpEmbed, posEmbUp) = encoder.upEmbed(hUpsampled)
eval(hAfterUpEmbed, posEmbUp)
print("Stage 5 - After up_embed:")
print("  Shape: \(hAfterUpEmbed.shape)")
print("  posEmbUp: \(posEmbUp.shape)")
print()

// Stage 6: Up encoder blocks (4)
hCurrent = hAfterUpEmbed
for (i, upEncoder) in encoder.upEncoders.enumerated() {
    hCurrent = upEncoder(hCurrent, posEmb: posEmbUp)
}
eval(hCurrent)
print("Stage 6 - After 4 up_encoder blocks:")
print("  Shape: \(hCurrent.shape)")
print()

// Stage 7: Up layer (Conv1d)
hCurrent = encoder.upLayer(hCurrent)
eval(hCurrent)
print("Stage 7 - After up_layer (Conv1d):")
print("  Shape: \(hCurrent.shape)")
print()

// Stage 8: Final norm and projection
hCurrent = encoder.afterNorm(hCurrent)
hCurrent = encoder.encoderProj(hCurrent)
eval(hCurrent)
print("Stage 8 - Final output:")
print("  Shape: \(hCurrent.shape)")
print("  Mean: \(String(format: "%.6f", Float(hCurrent.mean().item())))")
print("  Std: \(String(format: "%.6f", Float(hCurrent.variance().sqrt().item())))")
print()

print("=" + String(repeating: "=", count: 79))
print("COMPARISON WITH PYTORCH")
print("=" + String(repeating: "=", count: 79))
print()
print("PyTorch Stage 4 (after 2x upsample): 972 frames")
print("Swift Stage 4 (after 2x tiling): \(L*2) frames")
print()
if L * 2 != 972 {
    print("❌ MISMATCH: Swift has \(L*2) frames, PyTorch has 972 frames")
    print("   Difference: \(972 - L*2) frames")
    print()
    print("This means the input to upsampling has:")
    print("   PyTorch: 972/2 = 486 frames")
    print("   Swift: \(L) frames")
    print()
    if L != 486 {
        print("❌ ERROR: Input sequence length already differs BEFORE upsampling!")
        print("   We need to debug stages 1-3 (embed, encoders, pre_lookahead)")
    }
} else {
    print("✅ MATCH: Sequence lengths match!")
}
