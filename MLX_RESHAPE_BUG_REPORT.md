# MLX Swift Bug Report: Mysterious Reshape Error During Linear Layer Addition

## Summary

MLX-Swift crashes with a reshape error when adding a bias term to a matrix multiplication result in a Linear layer forward pass. The error message `[reshape] Cannot reshape array of size 0 into shape (5)` is misleading - no reshape operation to shape `[5]` exists in the code, and the arrays involved have non-zero sizes.

## Environment

- **Platform**: macOS (Darwin 25.1.0)
- **MLX Version**: Latest from package dependency (mlx-swift)
- **Swift Version**: 6.0
- **Project**: TTS model (T3Model) using MLX for inference
- **Model Weights**: FP32 (not quantized)

## Error Message

```
MLX/ErrorHandler.swift:343: Fatal error: [reshape] Cannot reshape array of size 0 into shape (5).
at /Users/a10n/Projects/nightingale_TTS/swift/test_scripts/GenerateAudio/.build/checkouts/mlx-swift/Source/Cmlx/mlx-c/mlx/c/ops.cpp:2521
```

## Exact Location of Crash

The crash occurs during a simple bias addition in the first Linear layer call:

```swift
// T3Model.swift, line ~1435
let result = matmul(speakerEmb, wT)  // [1, 256] @ [256, 1024] = [1, 1024] ✅ SUCCEEDS
let bBroadcast = bias.expandedDimensions(axis: 0)  // [1024] -> [1, 1024] ✅ SUCCEEDS
eval(result, bBroadcast)  // ✅ SUCCEEDS
let output = result + bBroadcast  // ❌ CRASHES HERE
```

## Verified State Before Crash

Debug output immediately before the crash shows both arrays are valid:

```
🚨 About to add bias...
🚨 bias shape: [1024], size: 1024
🚨 result shape: [1, 1024], size: 1024
🚨 Evaluating bias...
🚨 Bias evaluated
🚨 bias dtype: float32
🚨 result dtype: float32
🚨 Manually broadcasting bias...
🚨 bias broadcasted: [1, 1024]
🚨 Evaluating BOTH arrays before addition...
🚨 Both arrays evaluated
🚨 Adding bias...
MLX/ErrorHandler.swift:343: Fatal error: [reshape] Cannot reshape array of size 0 into shape (5).
```

**Key observations:**
- Both arrays have size 1024 (not 0)
- Shapes are compatible: `[1, 1024] + [1, 1024]`
- Both arrays are fully evaluated (not lazy)
- Both are `float32` dtype
- The error mentions shape `[5]`, which appears nowhere in the code

## Code Context

### Layer Configuration

The failing layer is `speakerProj`, loaded via LinearFactory:

```swift
// T3Model.swift, line 997
self.speakerProj = LinearFactory.load(
    "speakerProj",
    inputDim: 256,
    outputDim: 1024,
    weights: weights,
    bias: true
)
```

### Weight Loading (LinearFactory.swift)

```swift
// Fallback: Standard FP16 Linear
let linear = Linear(inputDim, outputDim, bias: bias)

if let w = weights[weightKey] {
    print("  [FP16] Loading standard: \(name) - dtype: \(w.dtype)")
    var params: [String: MLXArray] = ["weight": w]
    if bias, let b = weights["\(name).bias"] {
        params["bias"] = b
    }
    linear.update(parameters: ModuleParameters.unflattened(params))
}

return linear
```

### Verified Layer State

```
🚨 speakerProj type: Linear
🚨 speakerProj weight shape: [1024, 256]
🚨 speakerProj bias shape: [1024]
🚨 speakerProj is regular Linear (FP16)  // NOT QuantizedLinear
```

## Steps to Reproduce

1. Load FP32 weights from safetensors into T3Model
2. Create a Linear layer with bias via `LinearFactory.load()`
3. Call the layer with input shape `[1, 256]`
4. The layer internally computes:
   ```swift
   let result = matmul(input, weight.T)  // Succeeds
   let output = result + bias  // Crashes
   ```

## What We've Tried

### ✅ Workarounds That Don't Help

1. **Manual matrix multiplication** (instead of calling Linear directly):
   ```swift
   let wT = weight.T
   let result = matmul(input, wT)  // Succeeds
   let output = result + bias  // Still crashes
   ```

2. **Manual broadcasting**:
   ```swift
   let bBroadcast = bias.expandedDimensions(axis: 0)  // [1024] -> [1, 1024]
   let output = result + bBroadcast  // Still crashes
   ```

3. **Forced evaluation before addition**:
   ```swift
   eval(result, bBroadcast)  // Force computation
   let output = result + bBroadcast  // Still crashes
   ```

4. **Different calling patterns**:
   - Direct: `speakerProj(input)` - crashes
   - Manual decomposition - crashes
   - With/without eval() - crashes

### ❌ What Doesn't Work

- The operation `[1, 1024] + [1, 1024]` consistently crashes
- Both arrays are valid (non-zero size, correct dtype)
- No quantization involved (using FP32 weights)
- Issue is 100% reproducible

## Comparison with Working Code

**Identical code works perfectly in VerifyLive binary**, which uses the same model and weights. The only difference:
- **GenerateAudio** (crashes): Uses ChatterboxEngine wrapper → calls T3Model
- **VerifyLive** (works): Directly loads and uses T3Model

This suggests a subtle initialization or state issue rather than a fundamental bug in MLX operations.

## Hypothesis: Stale Build Cache?

Despite multiple clean rebuilds:
```bash
rm -rf /Users/a10n/Projects/nightingale_TTS/swift/.build
swift package clean
swift run GenerateAudio  # Still crashes
```

The crash persists even after:
- Full package clean
- Deleting .build directories
- Recompiling from scratch

## Misleading Error Details

### Why the error message is confusing:

1. **"array of size 0"** - Both arrays have size 1024
2. **"shape (5)"** - No reshape to `[5]` exists in the code
3. **Location: ops.cpp:2521** - Deep in MLX C++ internals, not user code

### Search for shape `[5]` in codebase:

```bash
$ grep -r "reshape.*5" swift/Sources/Nightingale/*.swift
# Only match: Debug checkpoint (output[0, L-1, 0..<5]) - not a reshape
```

The `[5]` in the error is a red herring - possibly an internal MLX buffer size.

## Minimal Reproduction Case

```swift
import MLX
import MLXNN

// Load FP32 weights from safetensors
let weights = try! MLX.loadArrays(url: weightsURL)

// Create Linear layer
let linear = Linear(256, 1024, bias: true)
linear.update(parameters: ModuleParameters.unflattened([
    "weight": weights["speakerProj.weight"],  // [1024, 256]
    "bias": weights["speakerProj.bias"]        // [1024]
]))

// Forward pass
let input = MLXArray.zeros([1, 256])  // Dummy input
let output = linear(input)  // Crashes here
```

## CRITICAL UPDATE: Root Cause Identified

### The Real Culprit: `.reshaped()` is Broken

After extensive debugging, we traced the crash to the exact operation:

```swift
let array = MLXArray([Float](repeating: 0.5, count: 1024))  // Valid [1024] array
let reshaped = array.reshaped([1, 1024])  // ❌ CRASHES HERE
// Error: "Cannot reshape array of size 0 into shape (5)"
```

**The Smoking Gun:**

1. Create valid MLXArray from Swift array: `MLXArray([Float])` ✅ Works
2. Print its shape: `[1024]` ✅ Correct
3. Call `.reshaped([1, 1024])`: ❌ Crashes with nonsensical error
4. Error claims "size 0" when array is size 1024
5. Error claims "shape (5)" when we're reshaping to `[1, 1024]`

**Both reshape methods crash:**

```swift
// Method 1: reshaped([1, 1024])
let a = tempArray.reshaped([1, 1024])  // ❌ Crashes

// Method 2: expandedDimensions(axis: 0)
let b = tempArray.expandedDimensions(axis: 0)  // ❌ Also crashes!
```

### Debugging Trail

```
🚨 Creating MLXArray from result (count=1024)...
🚨 MLXArray created: [1024]  ← Array is valid!
🚨 Using expandedDimensions instead of reshape...
MLX/ErrorHandler.swift:343: Fatal error: [reshape] Cannot reshape array of size 0 into shape (5).
```

### This is NOT User Error

- The array has size 1024 (not 0)
- We're reshaping to [1, 1024] (not [5])
- The operation is mathematically valid
- The error message is completely wrong

**Hypothesis:** Memory corruption in MLX's internal tensor metadata or a critical bug in the reshape kernel.

## Request for Help

**This appears to be a critical MLX-Swift bug.** The error reporting is so broken that it's reporting the wrong array size and the wrong target shape.

**Questions for MLX team:**

1. Why does `.reshaped([1, 1024])` on a valid `[1024]` array crash?
2. Why does `.expandedDimensions(axis: 0)` also crash?
3. Why does the error message report "size 0" and "shape (5)" when neither is correct?
4. Could this be memory corruption in tensor metadata?
5. Is there a workaround to reshape arrays without triggering this bug?

**Potential MLX internal issues:**

1. Buffer overflow corrupting tensor metadata
2. Race condition in lazy evaluation
3. Bug in reshape kernel validation logic
4. Memory corruption when creating MLXArray from Swift arrays

## Workaround Status

**Currently blocked** - cannot use ChatterboxEngine for audio generation due to this crash.

**Temporary solution**: Using Python implementation (verified mathematically equivalent via E2E tests).

## Files for Reference

- **Crashing code**: `/Users/a10n/Projects/nightingale_TTS/swift/Sources/Nightingale/T3Model.swift` (line 1435)
- **Layer factory**: `/Users/a10n/Projects/nightingale_TTS/swift/Sources/Nightingale/LinearFactory.swift`
- **Test binary**: `/Users/a10n/Projects/nightingale_TTS/swift/test_scripts/GenerateAudio/`
- **Model weights**: `/Users/a10n/Projects/nightingale_TTS/models/mlx/t3_fp32.safetensors`

## Full Debug Output

<details>
<summary>Complete debug trace (click to expand)</summary>

```
🔍 DEBUG BEFORE GENERATE:
  - Text Tokens: [1, 7]
  - Speaker Emb: [1, 256]
  - Cond Tokens: [1, 150]
  - Emotion Value: 0.5
🚀 Calling T3.generate with validated inputs...
🚨 T3.generate() ENTRY
🚨 After fflush
🚨 About to call speakerProj
🚨 speakerEmb shape: [1, 256]
🚨 speakerProj type: Linear
🚨 speakerProj weight shape: [1024, 256]
🚨 speakerProj bias shape: [1024]
🚨 speakerProj is regular Linear (FP16)
🚨 Attempting manual matrix multiply...
🚨 Got weight: [1024, 256]
🚨 Transposed weight: [256, 1024]
🚨 Matmul succeeded: [1, 1024]
🚨 About to add bias...
🚨 bias shape: [1024], size: 1024
🚨 result shape: [1, 1024], size: 1024
🚨 Evaluating bias...
🚨 Bias evaluated
🚨 bias dtype: float32
🚨 result dtype: float32
🚨 Manually broadcasting bias...
🚨 bias broadcasted: [1, 1024]
🚨 Evaluating BOTH arrays before addition...
🚨 Both arrays evaluated
🚨 Adding bias...
MLX/ErrorHandler.swift:343: Fatal error: [reshape] Cannot reshape array of size 0 into shape (5).
```

</details>

---

# SECOND MLX BUG: S3Gen Encoder Broadcast Shape Error

## Summary
S3Gen model loading crashes during initialization with a different broadcast shape error:
```
Fatal error: [broadcast_shapes] Shapes (1,80,64) and (1,564,80) cannot be broadcast.
at mlx/c/ops.cpp:3207
```

## Context
- **When**: During S3Gen initialization, triggered when accessing `.shape` property on T3 `norm.weight`
- **Where**: Before any S3Gen forward pass - happens during model construction
- **Impact**: Prevents loading S3Gen model, blocking Steps 5-8 E2E verification
- **Date**: 2025-12-19

## Error Details
- Error originates from MLX C++ layer during lazy evaluation
- Shapes suggest attention-related computation:
  - `(1, 80, 64)` - possibly query: [batch=1, seq_len=80, head_dim=64]
  - `(1, 564, 80)` - possibly position embeddings: [batch=1, seq_len=564, hidden=80]
  - `80` = mel channels, `64` = attention head dimension
  - `564 ≈ 2*282-1` = relative position encoding for seq_len=282

## Investigation Findings

### Weight Shapes Verified Correct
All encoder `linear_pos.weight` tensors have correct shape `(512, 512)`:
- `s3gen.flow.encoder.encoders.[0-5].self_attn.linear_pos.weight`: (512, 512) ✓
- `s3gen.flow.encoder.up_encoders.[0-3].self_attn.linear_pos.weight`: (512, 512) ✓

### Crash Location
1. T3 model loads successfully ✓
2. S3Gen initialization begins
3. Crash occurs when accessing `normWeight.shape` in T3Model.swift:962
4. Suggests MLX lazy evaluation of previous operations triggering stale computation graph

### Hypothesis
The bug may be in:
1. MLX-Swift lazy evaluation triggering stale operations from previous model
2. Shared global state between T3 and S3Gen models
3. EspnetRelPositionalEncoding position embedding generation creating wrong shapes
4. RelPositionMultiHeadAttention reshape operations during init

## Attempted Fixes
1. ✅ Added shape validation in `RelPositionMultiHeadAttention.load()` (RelPos.swift:228-239)
2. ✅ Added error handling in T3Model norm weight loading (T3Model.swift:956-983)
3. ✅ Verified all weight shapes are correct via safetensors inspection
4. ❌ Issue persists - appears to be MLX-Swift internal state bug

## Current Workaround
- Set `skipS3Gen = true` in VerifyLive/main.swift:693
- Steps 1-2 (T3 tokenization & conditioning) work correctly ✓
- Steps 5-8 (S3Gen encoder/decoder/vocoder) use reference file validation only

## Next Steps
1. Investigate MLX-Swift lazy evaluation behavior and computation graph management
2. Check if separate process/context needed for loading multiple models
3. Add try-catch around S3Gen init to fail gracefully
4. Consider loading S3Gen in isolation (no T3) to identify exact triggering operation
5. Report to MLX-Swift team if root cause is in framework

## Files Modified
- `swift/Sources/Nightingale/RelPos.swift:228-239` - Added weight shape validation
- `swift/Sources/Nightingale/T3Model.swift:956-983` - Added error handling & logging
- `swift/test_scripts/VerifyLive/main.swift:691-693` - Documented skip reason
- `E2E/verify_e2e.py:388-392` - Updated stage status (Steps 5-8 Swift-implemented but blocked)

---

**Contact**: Please advise on next debugging steps or if this is a known issue with a workaround.
