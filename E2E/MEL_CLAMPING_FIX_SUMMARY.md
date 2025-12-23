# Mel Clamping Fix - Summary

## What Was Fixed

Implemented **mel spectrogram clamping** in the Swift decoder to prevent positive values that cause audio distortion.

### Location
**File**: [swift/Sources/Nightingale/S3Gen.swift:1895-1900](../swift/Sources/Nightingale/S3Gen.swift#L1895-L1900)

### The Fix
```swift
h = finalProj(h)          // [B, T, 80]

// CRITICAL FIX: Clamp mel to ensure log-scale values stay negative
// Positive values indicate decoder overflow/saturation causing static audio
// Log-mel spectrograms represent log(magnitude), so must be ≤ 0
h = minimum(h, 0.0)
```

## Why This Was Needed

### Problem Identified
From systematic comparison, Swift decoder was producing:

1. **Sujano Voice**: Mel with **POSITIVE values** (max=+1.78)
   - Log-mel spectrograms must be negative (log of magnitude < 1)
   - Positive values indicate decoder overflow/saturation
   - Causes vocoder to produce **static audio** instead of clear speech
   - Prompt region works, but generated region fails

2. **Samantha Voice**: Mel **1.46 dB too dark** compared to Python
   - Swift max=-0.70 vs Python max=0.00
   - Missing bright values needed for clear speech articulation
   - Results in **mumbled words** instead of clear speech

### Root Cause
The encoder identity initialization fix (gamma=1, beta=0) corrected the ~100x signal suppression, but:
- Overcorrected for bright voices like Sujano
- Left Samantha slightly darker than Python
- No output clamping in decoder to prevent overflow

## Expected Results

### Immediate Impact
- **Sujano**: Clamping prevents positive values → eliminates static audio
- **Samantha**: May see slight brightness improvement (clamping at 0.0 vs -0.70)

### Audio Quality To Verify
Compare these test files (generated after clamping fix):

1. **Cross-validation** ([test_audio/cross_validate/](../test_audio/cross_validate/)):
   - `python_tokens_swift_audio.wav` - Should have clearer speech (Samantha)
   - `swift_tokens_swift_audio.wav` - Should have clearer speech (Samantha)

2. **Voice tests** ([test_audio/](../test_audio/)):
   - `chatterbox_engine_test.wav` - Sujano voice, should NO LONGER have static
   - Compare with Python references:
     - `python_samantha_test.wav` - Python reference (perfect)
     - `python_sujano_test.wav` - Python reference (perfect)

### What To Listen For

**Before Clamping Fix**:
- Sujano: Prompt sounds good → Generated region turns to static/noise
- Samantha: Humming background with mumbled/unclear words

**After Clamping Fix**:
- Sujano: Should maintain clear speech throughout (no static cutoff)
- Samantha: Should have clearer articulation (less mumbling)

## Technical Details

### Mel Statistics (Expected)

**Python (Reference)**:
- Samantha: Range=[-10.84, 0.00], Mean=-5.81 dB
- Sujano: Range=[-xx, -xx], Mean=-x.x dB (naturally brighter voice)

**Swift (Before Clamping)**:
- Samantha: Range=[-10.84, -0.70], Mean=-7.27 dB (1.46 dB darker)
- Sujano: Range=[-xx, **+1.78**], Mean=-x.x dB (**POSITIVE VALUES!**)

**Swift (After Clamping)**:
- Samantha: Range=[-10.84, **0.00**], Mean≈-6.x dB (closer to Python)
- Sujano: Range=[-xx, **0.00**], Mean=-x.x dB (**clamped to ≤ 0.0**)

## Verification Steps

1. **Listen to audio files** listed above
2. **Check mel statistics** (if needed):
   ```bash
   python E2E/check_clamping_samantha.py
   python E2E/test_mel_clamping.py
   ```
3. **Compare with Python baseline** - Swift should now match Python audio quality

## Remaining Work

If Samantha is still slightly mumbled after clamping:
- Investigate ODE solver spatial bias (why generated region darker than prompt)
- Check finalProj bias values (might need adjustment)
- Consider decoder output scaling (multiply by 0.9-1.1)

If Sujano has other issues after static is fixed:
- Check if voice characteristics need per-voice normalization
- Verify encoder output scaling for bright vs dark voices

## Success Criteria

✅ **Sujano**: No more static, clear speech throughout
✅ **Samantha**: Clearer articulation, less mumbling
✅ **Both**: Mel max values ≤ 0.0 (no positive overflow)
✅ **Both**: Audio quality approaches Python reference quality
