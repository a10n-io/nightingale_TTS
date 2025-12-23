# Vocoder Correlation Issue - CRITICAL

## Current Status: 0.19 Correlation (SEVERE PROBLEM)

**Date:** 2025-12-23
**Status:** CRITICAL - REQUIRES IMMEDIATE INVESTIGATION
**Priority:** HIGH

## Summary

The vocoder achieves only **0.1948 correlation** between Python and Swift implementations. This is **critically poor** and indicates a fundamental implementation difference in the vocoder network.

## Key Findings

### Vocoder Output Comparison
```
Python vocoder audio:
  Mean: -0.00005103
  Std: 0.01811083
  Range: [-0.073, 0.095]
  Duration: 3.92 seconds

Swift vocoder audio:
  Mean: -0.00005272
  Std: 0.01846640
  Range: [-0.076, 0.105]
  Duration: 3.92 seconds

CORRELATION: 0.1948 (CRITICALLY LOW)
SNR: -2.15 dB (NEGATIVE - noise exceeds signal!)

Element-wise differences:
  Mean absolute diff: 0.0145
  Max absolute diff: 0.1576
  Median absolute diff: 0.0069
```

### First 10 Audio Samples
```
Python: [0.000380, -0.000062, 0.000475, 0.000020, 0.000232, ...]
Swift:  [0.000259,  0.000151, 0.000403, -0.000044, -0.000069, ...]
```

Even the first samples diverge significantly, indicating an immediate difference in the vocoder computation.

## Input Comparison

### Decoder Mel Input to Vocoder
The decoder mel spectrograms feeding into the vocoder are slightly different (0.98 correlation), but this should NOT cause a 0.19 vocoder correlation:

```
Python decoder mel:
  Mean: -5.922373 dB
  Std: 2.263767 dB
  Range: [-10.342, 0.006] dB

Swift decoder mel:
  Mean: -5.883604 dB
  Std: 2.259108 dB
  Range: [-10.439, 0.142] dB
```

**Decoder correlation: 0.98** → **Vocoder correlation: 0.19**

This dramatic drop suggests the vocoder itself has implementation issues, not just input differences.

## Possible Causes

### 1. Weight Loading Issues
- Vocoder weights may not be loading correctly
- Transpose issues in convolutional layers
- Missing or incorrect weight keys

### 2. Network Architecture Mismatch
- Layer ordering different
- Skip connections misconfigured
- Upsampling differences

### 3. F0 Predictor Issues
The vocoder includes an F0 (pitch) predictor:
```python
# Python: hifigan.py:452
f0 = self.f0_predictor(speech_feat)
s = self.f0_upsamp(f0[:, None]).transpose(1, 2)
s, _, _ = self.m_source(s)
```

Potential issues:
- F0 predictor weights not matching
- F0 upsampling different
- Source generator (m_source) implementation differences

### 4. Activation Functions
- LeakyReLU slope differences
- Tanh implementation differences

### 5. Normalization Issues
- Batch norm / instance norm differences
- Running stats not matching

## Investigation Steps

### Priority 1: Check Vocoder Weight Loading
```bash
# Compare vocoder weights between Python and Swift
# Look for "mel2wav.*" or "vocoder.*" keys
```

### Priority 2: Trace F0 Predictor
- Save F0 predictions from both implementations
- Compare F0 values
- Check F0 upsampling

### Priority 3: Check Source Generator
- Compare source signal (s) between implementations
- Verify m_source weights

### Priority 4: Step-by-Step Layer Comparison
- Instrument every vocoder layer
- Save intermediate activations
- Find where divergence begins

## Verification Commands

```bash
# Generate Python vocoder output
python/venv/bin/python E2E/save_python_vocoder_output.py

# Generate Swift vocoder output
cd swift && swift run SaveVocoderOutput && cd ..

# Compare
python/venv/bin/python E2E/compare_vocoder_outputs.py
```

## Impact

**Audio Quality:** With 0.19 correlation and -2.15 dB SNR, the Swift vocoder is producing fundamentally different audio than Python. This is **NOT production-ready**.

**Expected vs Actual:**
- Expected: >0.99 correlation (near-perfect match)
- Actual: 0.19 correlation (essentially random)

## Files

### Python
- `E2E/save_python_vocoder_output.py` - Generate Python vocoder audio
- `python/chatterbox/src/chatterbox/models/s3gen/hifigan.py:446` - Vocoder forward()

### Swift
- `swift/test_scripts/SaveVocoderOutput.swift` - Generate Swift vocoder audio
- `swift/Sources/Nightingale/S3Gen.swift` - Vocoder implementation (search for "Mel2Wav")

### Comparison
- `E2E/compare_vocoder_outputs.py` - Compare vocoder outputs

## Next Steps

1. **URGENT:** Check if vocoder weights are loading at all
2. Instrument F0 predictor and compare predictions
3. Save intermediate vocoder layer outputs
4. Compare weight values directly
5. Check for transpose/reshape issues in convolutions

## References

- HiFi-GAN paper: https://arxiv.org/abs/2010.05646
- F0 (pitch) prediction in vocoders
- Source-filter model for speech synthesis
