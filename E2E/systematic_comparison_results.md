# Systematic Python vs Swift Comparison

## Samantha Voice - Stage by Stage

| Stage | Python | Swift | Status |
|-------|--------|-------|--------|
| **Speech Tokens** | | | |
| Shape | [98] | [98] | ✅ MATCH |
| Values | [1732, 2068, ...] | [1732, 2068, ...] | ✅ MATCH (100%) |
| **Encoder Output** | | | |
| Shape | [1, 392, 512] | [1, 696, 512] | ✅ (upsampled 2x vs 4x) |
| Range | TBD | [-763, 581] | ⚠️ FIXED (was [-6.24, 6.65]) |
| Mean | TBD | ~0 | ✅ Reasonable |
| **Generated Mel** | | | |
| Shape | [1, 80, 196] | [1, 80, 196] | ✅ MATCH |
| Range | [-10.38, 0.00] | [-10.44, -0.70] | ⚠️ Swift too dark |
| Mean | -5.81 | -7.27 | ⚠️ Swift 1.46 dB darker |
| Max | **0.00** | **-0.70** | ⚠️ Swift missing bright values |
| **Audio Output** | | | |
| Shape | [1, 94080] | [1, 94076] | ✅ MATCH |
| Range | [-0.075, 0.089] | [-0.033, 0.044] | ⚠️ Swift quieter |
| Quality | **PERFECT** | **MUMBLED** | ❌ Swift degraded |

## Sujano Voice - Stage by Stage

| Stage | Python | Swift | Status |
|-------|--------|-------|--------|
| **Speech Tokens** | | | |
| Shape | [98] | [98] | ✅ MATCH |
| Values | Same as samantha | Same as samantha | ✅ MATCH |
| **Encoder Output** | | | |
| Shape | [1, 392, 512] | [1, 696, 512] | ✅ (upsampled 2x vs 4x) |
| Range | TBD | TBD | ? |
| Mean | TBD | TBD | ? |
| **Generated Mel** | | | |
| Shape | [1, 80, 204] | [1, 80, 204] | ✅ MATCH |
| Range | TBD | [-5.27, **1.78**] | ❌ Swift has POSITIVE values! |
| Mean | TBD | -3.00 | ⚠️ Too bright |
| Max | < 0 (expected) | **+1.78** | ❌ INVALID (should be negative) |
| **Audio Output** | | | |
| Shape | [1, 97920] | [1, 97916] | ✅ MATCH |
| Range | [-0.795, 1.033] | [-0.496, 0.708] | ⚠️ Swift clipped |
| Quality | **PERFECT** | **STATIC/CUTOFF** | ❌ Swift broken |

## Key Findings

### ✅ Fixed Issues
1. **Encoder magnitude**: Identity norm initialization fixed the ~100x suppression
   - Before: Swift encoder [-6.24, 6.65]  
   - After: Swift encoder [-763, 581] ≈ Python [-595, 851]

### ❌ Remaining Issues

#### Samantha (Mumbled)
- **Mel too dark**: Swift -7.27 vs Python -5.81 (1.46 dB difference)
- **Missing bright values**: Swift max=-0.70 vs Python max=0.00
- Likely causes:
  - Decoder ODE solver accumulating darkness
  - finalProj bias or scaling issue
  - Missing activation or normalization

#### Sujano (Static/Cutoff)
- **CRITICAL**: Generated mel has **positive values** (max=+1.78)
  - Log-scale mels must be negative!
  - Indicates severe decoder overflow/saturation
- Prompt region works → generated region fails
- Likely causes:
  - Encoder identity norm too strong for bright voices
  - Decoder produces unbounded outputs
  - Missing output clamping

## Recommended Fixes

### Priority 1: Clamp Mel Output (Sujano)
Add clamping after finalProj to prevent positive mel values:
```swift
mel = minimum(mel, 0.0)  // Ensure log-mels stay negative
```

### Priority 2: Investigate Decoder Darkness (Samantha)
- Compare Python vs Swift decoder intermediate outputs
- Check if ODE velocity field has spatial bias
- Verify finalProj weights and bias are correct

### Priority 3: Encoder Strength Tuning
- Consider scaling encoder output by 0.5-0.8 for bright voices
- Or add voice-specific normalization

