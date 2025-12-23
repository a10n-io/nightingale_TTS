# Vocoder Debugging - KEY FINDINGS

**Date:** 2025-12-23
**Status:** Bug Located - Not in mSource, but in source processing/fusion or main path

## Critical Discoveries

### ✅ mSource.linear Weights ARE Loading Correctly

**Python weights:**
```
Shape: [1, 9]
Mean (absolute): 0.001047
Std: 0.001262
First 5: [-0.001179, -0.000267, -0.000394, 0.001139, 0.001359]
```

**Swift weights:**
```
Shape: [9, 1] (transposed)
Mean (absolute): 0.001047  ✅ EXACT MATCH
Std: 0.001190
First 5: [-0.001179, -0.000267, -0.000394, 0.001139, 0.001359]  ✅ EXACT MATCH
```

**Conclusion:** Weights load perfectly via Module.update()

---

### ✅ Source Signal 0.0045 Correlation is NORMAL

**Tested Python vocoder with same input, 3 runs:**
```
Source correlations (random phases):
  Run 0 vs Run 1: 0.000961
  Run 0 vs Run 2: 0.001667
  Run 1 vs Run 2: -0.004384

Mean/Std (consistent):
  All runs: mean=0.005906, std=0.000119
```

**Swift vs Python source correlation:** 0.0045

**Conclusion:** Source signals use random phase initialization, producing ~0.001-0.005 correlation between runs. This is EXPECTED behavior for neural vocoders. The Swift implementation is working correctly here.

---

### 🚨 CRITICAL: Python Vocoder is 99.98% Deterministic

**Tested Python VOCODER OUTPUT with same input, 3 runs:**
```
Vocoder output correlations:
  Run 0 vs Run 1: 0.999836  ← 99.98% match!
  Run 0 vs Run 2: 0.999791  ← 99.98% match!
```

**Key Insight:** Despite source having random phases (0.001 correlation), the FINAL VOCODER OUTPUT is essentially deterministic (0.9998 correlation)!

This means:
1. Random source phases don't significantly affect final audio
2. The source contribution is statistically consistent
3. The vocoder should produce nearly identical outputs given the same mel input

---

### ❌ REAL BUG: Swift Vocoder Only 0.82 Correlation

**Swift vs Python vocoder output:** 0.8171 correlation

**Expected:** >0.999 correlation (like Python vs Python)
**Actual:** 0.817 correlation
**Gap:** Massive 18% correlation loss

**Conclusion:** There IS a real bug in Swift's vocoder implementation, but it's NOT in mSource.linear or source generation.

---

## Bug Location Analysis

### ✅ VERIFIED CORRECT:
1. **Input mel** - 1.0 correlation
2. **F0 prediction** - 1.0 correlation
3. **mSource.linear weights** - Exact match
4. **Source signal generation** - Random phases are correct behavior
5. **Conv pre** - 1.0 correlation

### ❌ BUG MUST BE IN:
1. **Source STFT** - Converting source signal to frequency domain
2. **Source fusion layers:**
   - `source_downs[0-2]` - Conv1d layers
   - `source_resblocks[0-2]` - ResBlock layers
3. **Main vocoder path:**
   - `ups[0-2]` - ConvTransposed1d upsampling
   - `resblocks[0-8]` - ResBlock layers (3 per upsampling stage)
   - `conv_post` - Final 1D convolution
4. **iSTFT reconstruction** (if used)

---

## Next Steps

### Priority 1: Trace Source Processing
1. Save source STFT output (sSTFT) from both Python and Swift
2. Compare source_downs outputs at each stage
3. Compare source_resblocks outputs
4. Verify where source gets fused with main path

### Priority 2: Trace Main Vocoder Path
1. Save outputs after each ups[i] layer
2. Save outputs after each resblock
3. Compare conv_post output
4. Verify final audio reconstruction

### Priority 3: Check for Known Issues
1. Transpose bugs in Conv1d/ConvTransposed1d layers
2. Weight loading issues in source_downs/source_resblocks
3. STFT window/parameters mismatch
4. Activation function differences (LeakyReLU slope)

---

## Test Commands

```bash
# Check Python mSource weights
python E2E/check_python_msource_weights.py

# Test Python vocoder determinism
python E2E/test_source_deterministic.py

# Run Swift with weight debugging
cd swift && swift run SaveVocoderCrossValidation

# Compare vocoder outputs
python E2E/compare_vocoder_cross_validation.py
```

---

## Key Files

- [S3Gen.swift:2064-2085](../swift/Sources/Nightingale/S3Gen.swift#L2064-L2085) - SourceModuleHnNSF
- [S3Gen.swift:2008-2062](../swift/Sources/Nightingale/S3Gen.swift#L2008-L2062) - SineGen
- [S3Gen.swift:2087-2330](../swift/Sources/Nightingale/S3Gen.swift#L2087-L2330) - Mel2Wav vocoder
- [hifigan.py:234-283](../python/chatterbox/src/chatterbox/models/s3gen/hifigan.py#L234-L283) - Python SourceModuleHnNSF
- [hifigan.py:286-475](../python/chatterbox/src/chatterbox/models/s3gen/hifigan.py#L286-L475) - Python Mel2WavWrapper

---

## Hypothesis

The bug is likely in the **source fusion** code where the source STFT gets downsampled and combined with the main mel path via ResBlocks. This is a complex part of the vocoder with multiple Conv1d layers that could have transpose/weight loading issues similar to what we fixed in the encoder.

The source path is processed independently and then **added** to the main path at each upsampling stage. If this fusion is broken, the vocoder would still work but with degraded quality (explaining the 0.82 correlation instead of total failure).
