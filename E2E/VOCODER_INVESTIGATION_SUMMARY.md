# Vocoder Investigation Summary

**Date:** 2025-12-23
**Status:** IN PROGRESS
**Current Correlation:** 0.817

## Investigation Progress

### ✅ Verified: Input Data Matches
- Python and Swift vocoders receive IDENTICAL input data
- Data is just transposed: Python [B,T,C], Swift [B,C,T]
- Correlation: 1.0 (perfect match)

### ✅ Fixed: Removed Incorrect F0 Scaling
- **Bug Found:** Swift was multiplying F0 by 24000.0
- **Root Cause:** Misunderstanding of f0_predictor output scale
- **Fix Applied:** Removed `f0 = f0 * 24000.0` from [S3Gen.swift:2241](../swift/Sources/Nightingale/S3Gen.swift#L2241)
- **Result:** Correlation remains 0.817 (no change)
- **Conclusion:** F0 scaling was NOT the main issue

### 🔍 Layer-by-Layer Divergence Analysis

| Layer | Python | Swift | Correlation | Status |
|-------|--------|-------|-------------|--------|
| Input Mel | [1, 80, 196] | [1, 80, 196] | 1.000 | ✅ MATCH |
| F0 Prediction | [1, 196] | [1, 196] | 1.000 | ✅ MATCH |
| F0 Upsampled | [1, 94080, 1] | [1, 94080, 1] | 1.000 (pattern) | ⚠️ DIFFERENT METHOD |
| **Source Signal** | [1, 94080, 1] | [1, 94080, 1] | **0.0045** | ❌ **DIVERGED** |
| Conv Pre | [1, 512, 196] | [1, 196, 512] | 1.000 | ✅ MATCH |

**First Divergence Point:** Source Signal Generation (Step 3)

### 🐛 Remaining Issues

#### 1. F0 Upsampling Method Difference
- **Python:** `torch.nn.Upsample(scale_factor=480)` - uses interpolation (default: linear)
- **Swift:** `tiled()` - simple repeat/tile
- **Impact:** Pattern correlation is 1.0, but values differ in scale
- **Investigation Needed:** Check if this affects source generation

#### 2. Source Signal Generation (0.0045 correlation!)
**Python Source:**
```
Mean: 0.00591
Std: 0.00012
Range: [0.00540, 0.00638]
First 5: [0.00589, 0.00604, 0.00589, 0.00586, 0.00607]
```

**Swift Source:**
```
Mean: 0.00591
Std: 0.00025
Range: [0.00516, 0.00659]
First 5: [0.00568, 0.00639, 0.00634, 0.00567, 0.00539]
```

**Observation:**
- Means are similar (~0.0059)
- Swift has 2x higher std deviation
- Completely different value patterns (0.0045 correlation)

**Possible Causes:**
1. SourceModuleHnNSF / SineGen weight loading issues
2. Random phase initialization differences
3. Implementation differences in sine generation
4. F0 upsampling method affecting source

#### 3. Vocoder End-to-End Output
- **Correlation:** 0.817
- **SNR:** 4.37 dB
- **Status:** Not production-ready

## Next Steps

### Priority 1: Investigate Source Signal Divergence
1. ✅ Verify F0 values match before source generation
2. ❌ Check SourceModuleHnNSF/SineGen implementation
3. ❌ Compare mSource weights between Python and Swift
4. ❌ Check random phase initialization
5. ❌ Test if upsampling method (interpolate vs tile) affects source

### Priority 2: Systematic Debugging
1. Save intermediate source generation steps
2. Compare sine wave generation
3. Check noise addition
4. Verify linear layer in source module

## Files Modified

- [S3Gen.swift:2231-2237](../swift/Sources/Nightingale/S3Gen.swift#L2231-L2237) - Removed incorrect F0 scaling
- [TraceVocoderLayers.swift](../swift/test_scripts/TraceVocoderLayers.swift) - Updated trace script

## Test Commands

```bash
# Python vocoder trace
python/venv/bin/python E2E/trace_python_vocoder_layers.py

# Swift vocoder trace
cd swift && swift run TraceVocoderLayers && cd ..

# Compare layers
python/venv/bin/python E2E/compare_vocoder_layers.py

# Cross-validation
python/venv/bin/python E2E/cross_validate_vocoder.py
cd swift && swift run SaveVocoderCrossValidation && cd ..
python/venv/bin/python E2E/compare_vocoder_cross_validation.py
```

## Key Insight

The vocoder divergence is NOT from input data or F0 prediction - those match perfectly (1.0 correlation). The divergence happens at **Source Signal Generation**, where Swift and Python produce completely different source signals (0.0045 correlation) despite receiving similar F0 inputs. This suggests an implementation bug in Swift's SourceModuleHnNSF or SineGen, possibly related to:
- Weight loading
- Random initialization
- Sine wave generation algorithm
- F0 upsampling interpolation vs tiling

The fact that Conv Pre still matches (1.0 correlation) confirms the mel processing path is correct - only the source path is broken.
