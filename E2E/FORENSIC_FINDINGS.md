# Forensic Investigation Findings

## Executive Summary

**ROOT CAUSE IDENTIFIED:** The **encoder** is producing completely different outputs between Python and Swift, which cascades through the entire pipeline causing the 0.91 dB mel difference and mumbled audio.

## Investigation Timeline

### Initial Problem
- Swift audio sounds mumbled
- Previous attempts: +1.5 dB, +2.0 dB, +7.82 dB corrections all failed
- User requested: "systematic forensic approach" for "mathematical precision"

### Forensic Analysis Results

#### Step 1: finalProj Weights Comparison ✅
**Result: PERFECT MATCH**
- Weight diff: 0.00000000
- Bias diff: 0.00000000
- Conclusion: finalProj weight loading is correct

#### Step 2: Final Mel Comparison
**Result: 0.91 dB DIFFERENCE**
- Python mel: mean=-5.79 dB, range=[-10.28, +0.07]
- Swift mel: mean=-4.88 dB, range=[-12.63, +0.66]
- Difference: Swift is 0.91 dB BRIGHTER (not darker as initially thought!)
- Correlation: 0.696 (moderate)

#### Step 3: ODE Output Comparison (Before finalProj)
**Result: LARGE DIVERGENCE**
- Python ODE: mean=0.603, std=0.909
- Swift ODE: mean=0.601, std=0.902
- Abs mean diff: **0.311** (LARGE!)
- Correlation: **0.864** (should be >0.99)
- Max abs diff: 3.574

**Key Insight:** When applying the SAME finalProj weights to both ODE outputs:
- Python mel: -5.85 dB
- Swift mel: -4.88 dB
- Difference: -0.97 dB (matches observed 0.91 dB!)

**Conclusion:** ODE solver is producing structurally different outputs.

#### Step 4: Encoder Output Comparison ⚠️ **ROOT CAUSE**
**Result: MASSIVE DIVERGENCE**
- Python encoder: mean=-0.007, std=0.455, range=[-1.75, +1.85]
- Swift encoder: mean=0.011, std=0.227, range=[-0.91, +0.84]
- Abs mean diff: **0.400** (HUGE - larger than ODE diff!)
- Correlation: **0.009** (virtually NO correlation!)

**Critical Finding:** The encoder difference (0.40) is LARGER than the ODE difference (0.31), proving the encoder is the root cause.

**Observations:**
- Swift encoder has HALF the standard deviation (0.227 vs 0.455)
- Swift encoder has HALF the range (1.75 vs 3.59)
- Essentially NO correlation (0.009) - they're producing completely unrelated outputs!

## Causal Chain

```
Encoder Outputs Differ (correlation 0.009)
         ↓
ODE Solver Gets Wrong Conditioning
         ↓
ODE Outputs Diverge (correlation 0.864)
         ↓
Same finalProj Applied to Different Inputs
         ↓
Mel Differs by 0.91 dB
         ↓
Vocoder Produces Mumbled Audio
```

## What We Know

✅ **Working Correctly:**
1. Vocoder (confirmed: python_mel_swift_vocoder.wav plays perfectly)
2. finalProj weights (diff = 0.0)
3. Token generation (100% match)

❌ **Broken:**
1. **Encoder** - producing completely different outputs (ROOT CAUSE)
2. ODE solver - diverging due to wrong encoder conditioning
3. Final mel - differs by 0.91 dB

## Next Steps to Fix

### Priority 1: Investigate Encoder
1. **Compare encoder weights** between Python and Swift
   - Check if weights are loaded correctly
   - Look for transposition issues
   - Verify all sublayers

2. **Check encoder inputs:**
   - Verify speech_emb_matrix is identical
   - Verify speech tokens are identical (already confirmed)
   - Check if there's any preprocessing difference

3. **Check encoder architecture:**
   - Verify layer normalization implementations
   - Check upsampling implementation
   - Verify attention mechanisms

### Priority 2: Once Encoder is Fixed
- Re-run full pipeline
- Verify ODE outputs match
- Verify mel outputs match
- Confirm audio quality

## Files Saved for Analysis

```
test_audio/forensic/
├── python_mel_raw.safetensors          # Python final mel
├── swift_mel_raw.safetensors           # Swift final mel
├── python_ode_output.safetensors       # Python ODE output (before finalProj)
├── swift_ode_output.safetensors        # Swift ODE output (before finalProj)
├── python_encoder_output.safetensors   # Python encoder output ⚠️ ROOT CAUSE
├── swift_encoder_output.safetensors    # Swift encoder output ⚠️ ROOT CAUSE
├── python_finalproj_weights.safetensors # For weight comparison
```

## Metrics Summary

| Stage | Python | Swift | Abs Diff | Correlation | Status |
|-------|--------|-------|----------|-------------|--------|
| **Encoder** | mean=-0.007, std=0.455 | mean=0.011, std=0.227 | **0.400** | **0.009** | ❌ **ROOT CAUSE** |
| ODE Output | mean=0.603, std=0.909 | mean=0.601, std=0.902 | 0.311 | 0.864 | ❌ Diverging |
| finalProj Weights | - | - | 0.000 | 1.000 | ✅ Perfect |
| Final Mel | mean=-5.79 | mean=-4.88 | 0.91 dB | 0.696 | ❌ Different |

## Conclusion

The systematic forensic approach successfully identified the root cause: **the encoder is producing completely uncorrelated outputs** between Python and Swift. This is not a small numerical difference - the outputs are structurally completely different (correlation 0.009).

Fixing the encoder will cascade through and fix:
- ODE solver outputs
- Final mel outputs
- Audio quality

**All ad-hoc corrections (+1.5, +2.0, +7.82 dB) failed because they were treating the symptom (wrong mel) instead of the disease (broken encoder).**
