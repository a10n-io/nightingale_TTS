# Forensic Investigation Plan - Achieving Mathematical Precision

## Problem Statement

Swift mel output does not match Python, causing:
- Samantha: Mumbled speech (mel too dark by ~1.5-2 dB)
- Sujano: Static/distorted audio (mel has positive values)

**Critical Issue**: Ad-hoc corrections (+1.5 dB, +2.0 dB, +7.82 dB) have all failed because:
1. We're guessing the correction without understanding root cause
2. We don't have direct mel-to-mel comparison
3. Audio-based inference is unreliable (different scales, vocoder effects)

## What We Know

### From Systematic Comparison (E2E/systematic_comparison_results.md):
- **Raw Swift decoder mel** (before any corrections):
  - Range: [-10.44, -0.70]
  - Mean: -7.27
- **Python decoder mel**:
  - Range: [-10.38, 0.00]
  - Mean: -5.81
- **Difference**: Swift is 1.46 dB darker (mean), 0.70 dB darker (max)

### What's Been Fixed:
1. ✅ Encoder suppression (~100x) - fixed via identity norm initialization
2. ✅ Token generation - 100% match with Python

### What's Still Wrong:
1. ❌ Decoder produces mel that's 1.46 dB darker than Python
2. ❌ Applied corrections (+1.5, +2.0, +7.82 dB) all produce audio that's too loud
3. ❌ Don't understand WHY Swift is darker in the first place

## Root Cause Investigation Needed

### Step 1: Save Exact Mel Values from Both Systems

**Python:**
```python
# In decoder forward(), right after finalProj:
mel = self.final_proj(x)
save_file({"mel": mel.cpu()}, "python_decoder_mel_exact.safetensors")
```

**Swift:**
```swift
// In S3Gen.swift, right after finalProj (line 1895):
h = finalProj(h)
try! MLX.save(arrays: ["mel": h], url: outputPath)
```

**Compare numerically:**
```python
diff = python_mel - swift_mel
print(f"Mean diff: {diff.mean()}")
print(f"Max diff: {diff.max()}")
print(f"Channel-by-channel: {diff.mean(dim=(0,2))}")  # Per mel bin
```

### Step 2: Trace Decoder Intermediate Outputs

Compare at EVERY stage:
1. **Encoder output** → should match (already verified)
2. **ODE initial state (x0)** → compare prompt_feat vs random init
3. **ODE steps (10 steps)** → compare velocity field at each step
4. **ODE final state (xt)** → compare before finalProj
5. **finalProj output** → compare weights, bias, output

### Step 3: Check finalProj Weights

```python
# Python
proj_weight = model.s3gen.flow.decoder.final_proj.weight
proj_bias = model.s3gen.flow.decoder.final_proj.bias
```

```swift
// Swift
let projWeight = decoder.finalProj.weight
let projBias = decoder.finalProj.bias
```

Compare:
- Shape
- Range [min, max]
- Mean
- First/last few values

### Step 4: Check ODE Solver Parity

The decoder uses an ODE solver. Differences could arise from:
- Different integration methods (Euler vs adaptive)
- Different timesteps
- Numerical precision (Float32 vs Float64)
- CFG (classifier-free guidance) implementation

**Verification:**
1. Print velocity field at each ODE step
2. Print state (x) at each ODE step
3. Compare step-by-step

### Step 5: Check for Hidden Scaling Factors

Look for any place where Python applies scaling that Swift doesn't:
- Layer norms (gamma, beta)
- Activation functions (GELU, SiLU parameters)
- Attention scaling (1/sqrt(d))
- Embedding scaling

## Expected Output Format

For each stage, document:
```
Stage: [Name]
Python: [shape], range=[min, max], mean=[mean], std=[std]
Swift:  [shape], range=[min, max], mean=[mean], std=[std]
Diff:   abs_mean=[X], abs_max=[Y], relative=[Z%]
Status: ✅ MATCH / ⚠️ SMALL DIFF / ❌ LARGE DIFF
```

## Success Criteria

1. **Numerical precision**: Swift mel matches Python to within 0.01 dB (mean)
2. **No ad-hoc corrections**: Fix root cause, not symptoms
3. **Reproducible**: Same inputs always produce same outputs
4. **Documented**: Full trace showing where any remaining differences come from

## Tools Needed

1. **Python trace script**: Save all intermediate outputs
2. **Swift trace script**: Save all intermediate outputs
3. **Comparison script**: Load both and compare numerically
4. **Visualization**: Plot differences to see patterns

## Current Status

- ❌ No direct mel-to-mel comparison available
- ❌ Root cause unknown
- ❌ Applied 3 different corrections, all wrong
- ✅ Systematic comparison document exists
- ✅ Know the target values (-5.81 mean, 0.00 max)

## Next Immediate Steps

1. Revert Swift to RAW output (remove +2.0 dB correction)
2. Implement mel-saving in both Python and Swift
3. Run both with SAME inputs, save mels
4. Compare mels numerically
5. Identify exact transformation needed
6. Trace backwards to find WHY that transformation is needed
7. Fix root cause

---

**Key Insight**: We've been treating symptoms (dark mel) with corrections, but we haven't found the disease (why it's dark). We need to find and fix the root cause for mathematical precision.
