# MLX Broadcast Shape Error Investigation

**Error:** `[broadcast_shapes] Shapes (1,80,64) and (1,564,80) cannot be broadcast`

**Status:** ACTIVE INVESTIGATION - Root cause not yet identified

**Date:** 2025-12-19

---

## Error Summary

### The Crash
- **Location:** Occurs when printing Step 7a header in VerifyLive test
- **Timing:** After Step 5 completes, Step 6 is disabled, when transitioning to Step 7a
- **Trigger:** Lazy evaluation of deferred MLX operations (triggered by string operation or scope exit)
- **Error Message:**
  ```
  MLX/ErrorHandler.swift:343: Fatal error: [broadcast_shapes]
  Shapes (1,80,64) and (1,564,80) cannot be broadcast.
  ```

### Shape Analysis

#### The Cursed Shapes
1. **Shape (1, 80, 64)**
   - 80 = `mel_channels` (vocoder feature dimension)
   - 64 = `attention_head_dim` (dModel / numHeads = 512 / 8)
   - **Problem:** 80 is in the TIME/SEQUENCE dimension slot where sequence length should be
   - This suggests a tensor with mel_channels in the wrong axis

2. **Shape (1, 564, 80)**
   - 564 = 2 × 282 - 1 (Relative Positional Encoding table length)
   - 282 = sequence length (250 prompt tokens + 32 speech tokens)
   - 80 = `mel_channels`
   - **This is the "smoking gun":** PE table projected to 80 instead of 64

#### Correct Dimensions
- `dModel = 512` (encoder hidden dimension)
- `numHeads = 8`
- `headDim = 64` (dModel / numHeads = 512 / 8)
- `melChannels = 80` (vocoder feature dimension)
- `hiddenDim = 256` (decoder hidden dimension)
- `seqLen = 282` (prompt_tokens[250] + speech_tokens[32])

---

## Investigation Timeline

### Phase 1: Initial Hypothesis - MLX Framework Bug
**Result:** ❌ REJECTED

- Initially suspected MLX lazy evaluation bug
- User correctly identified this as a legitimate dimension mismatch in user code
- The broadcast error is real - not a framework issue

### Phase 2: Isolated Crash Location
**Result:** ✅ CONFIRMED - Deferred operation from Step 5 or model loading

**Evidence:**
1. Completely disabled Step 6 → crash still occurs
2. All Step 5 output tensors have correct shapes:
   - `fullTokens: [1, 282]` ✓
   - `tokenEmb: [1, 282, 512]` ✓
   - `spkEmbProj: [1, 80]` ✓
3. Crash happens when printing Step 7a header (String operation)
4. String operations can't create tensor broadcast errors
5. **Conclusion:** The deferred computation was created earlier, evaluation triggered later

### Phase 3: Encoder Initialization Analysis
**Result:** ✅ VERIFIED CORRECT

**Checked:**
- All `EspnetRelPositionalEncoding` initializations use `dModel=512` ✓
- All `RelPositionMultiHeadAttention` initializations:
  - `dModel=512` ✓
  - `numHeads=8` ✓
  - `dHead=64` ✓
- No suspicious `dModel=80` or `dModel=640` values found

**Debug Output Confirmed:**
```
🔍 RelPosAttn.init: dModel=512, numHeads=8, dHead=64
```
(Repeated for all 10 encoder blocks: 6 main + 4 up-encoders)

### Phase 4: Safetensors Weight Verification
**Result:** ✅ VERIFIED CORRECT

**All positional bias weights in safetensors:**
```
s3gen.flow.encoder.encoders.*.self_attn.pos_bias_u: (8, 64) ✓
s3gen.flow.encoder.encoders.*.self_attn.pos_bias_v: (8, 64) ✓
s3gen.flow.encoder.up_encoders.*.self_attn.pos_bias_u: (8, 64) ✓
s3gen.flow.encoder.up_encoders.*.self_attn.pos_bias_v: (8, 64) ✓
```

**All positional projection weights:**
```
s3gen.flow.encoder.*.self_attn.linear_pos.weight: (512, 512) ✓
```

---

## What We Ruled Out

### ❌ Not the Encoder
- All FlowEncoder conformer blocks correctly initialized
- All RelPositionMultiHeadAttention modules have correct dModel/numHeads/dHead
- All loaded weights have correct shapes
- PE tables (`posEnc.pe`, `upPosEnc.pe`) successfully evaluated without errors

### ❌ Not the Decoder (Probably)
- All decoder attention parameters successfully evaluated
- Decoder uses standard MultiHeadAttention (not RelPositionMultiHeadAttention)
- No positional encoding in decoder (uses time embeddings instead)

### ❌ Not Step 5 Outputs
- All three Step 5 output tensors have correct shapes
- Individual evaluation succeeds
- Pairwise evaluation succeeds
- Only triple evaluation fails (but tensors are already evaluated individually)

### ❌ Not the Weights File
- All safetensors weights have correct dimensions
- No (8, 80) or (564, 80) shapes found in weights
- PE weights are correctly (1, 9999, 512)
- Bias weights are correctly (8, 64)

---

## Current Hypothesis

### The Hidden Broadcast Operation

**Theory:** Somewhere in the computation graph, there's a deferred operation that:

1. Takes `spkEmbProj [1, 80]`
2. Reshapes/expands it incorrectly to `[1, 80, *]` where * should be sequence length
3. This creates the `(1, 80, 64)` intermediate tensor
4. Later tries to broadcast with PE table `(1, 564, 80)`
5. **Result:** Broadcast error because 80 (dim 1) ≠ 564 (dim 1) AND 64 (dim 2) ≠ 80 (dim 2)

### Potential Culprits

1. **Speaker Embedding Processing**
   - `spkEmbProj` is [1, 80] - this is the ONLY tensor with dim=80 in Step 5
   - Something might be expanding it incorrectly
   - Could be happening during encoder forward pass (not yet executed)

2. **Lazy Projection in PE Forward Pass**
   - PE table is loaded as [1, 9999, 512]
   - Gets sliced to [1, 564, 512] for current sequence
   - `linearPos(posEmb)` projects to [1, 564, 512]
   - Then reshaped to [1, 564, numHeads, dHead] = [1, 564, 8, 64]
   - **But what if linearPos output is [1, 564, 80] instead?**

3. **Weight Loading Bug**
   - Even though initialization is correct (dModel=512)
   - What if `linearPos.weight` gets overwritten with wrong-shaped weights?
   - Could be from weight remapping/quantization code

---

## Dimension Math Analysis

### The Expected Flow (Correct)

```
posEmb:        [1, 564, 512]   # PE table slice
linearPos:     512 → 512       # Projection
pProj:         [1, 564, 512]   # Output
reshape:       [1, 564, 8, 64] # Split into heads
transpose:     [1, 8, 564, 64] # BHLD format
```

### The Buggy Flow (Hypothesis)

```
posEmb:        [1, 564, 512]   # PE table slice
linearPos:     512 → 80  (???) # WRONG! Should be 512→512
pProj:         [1, 564, 80]    # CURSED OUTPUT
reshape:       [1, 564, ?, ?]  # Try to split 80 into (numHeads, dHead)
                                # If numHeads=1: [1, 564, 1, 80] → transpose → [1, 1, 564, 80]
                                # But need to broadcast with something expecting [*, *, *, 64]
                                # This gives us the (1, 564, 80) we see in the error!
```

**The Question:** How does `linearPos` end up projecting 512 → 80 instead of 512 → 512?

---

## Code Locations

### Key Files
1. [RelPos.swift:118-150](swift/Sources/Nightingale/RelPos.swift) - `RelPositionMultiHeadAttention.init()`
2. [RelPos.swift:164-228](swift/Sources/Nightingale/RelPos.swift) - `RelPositionMultiHeadAttention.callAsFunction()`
3. [RelPos.swift:248-315](swift/Sources/Nightingale/RelPos.swift) - `RelPositionMultiHeadAttention.load(weights:prefix:)`
4. [FlowEncoder.swift:164-204](swift/Sources/Nightingale/FlowEncoder.swift) - `ConformerEncoderBlock`
5. [S3Gen.swift:376-390](swift/Sources/Nightingale/S3Gen.swift) - `ConformerBlock`

### Critical Code Path

```swift
// RelPos.swift:182-194
let pProj = linearPos(posEmb)  // [nBatchPos, posSeqLen, dModel]

// Expected: dModel=512, so pProj should be [1, 564, 512]
// But error suggests: pProj is [1, 564, 80] ???

let expectedDim = numHeads * dHead  // 8 * 64 = 512
if pProj.shape[2] != expectedDim {
    fatalError("linearPos output dim mismatch!")  // This should trigger if bug exists
}

let p = pProj.reshaped([nBatchPos, posSeqLen, numHeads, dHead])
// If pProj is [1, 564, 80], this reshape will fail or produce [1, 564, ?, ?]
```

---

## Next Steps

### 1. Add Debug Logging to linearPos Projection

Add to `RelPositionMultiHeadAttention.callAsFunction()` at line 181:

```swift
let pProj = linearPos(posEmb)
print("🔍 linearPos DEBUG:")
print("  linearPos.weight.shape: \(linearPos.weight.shape)")
print("  posEmb.shape: \(posEmb.shape)")
print("  pProj.shape: \(pProj.shape)")
print("  Expected pProj dim: \(expectedDim) (numHeads=\(numHeads) * dHead=\(dHead))")
eval(pProj)
fflush(stdout)
```

### 2. Verify linearPos Weight Loading

Add to `RelPositionMultiHeadAttention.load()` at line 278:

```swift
if let w = weights["\(prefix).linear_pos.weight"] {
    print("🔍 Loading linear_pos.weight:")
    print("  Key: \(prefix).linear_pos.weight")
    print("  Weight shape: \(w.shape)")
    print("  Expected: [\(dModel), \(dModel)] = [512, 512]")
    eval(w)
    fflush(stdout)

    if w.shape[0] != dModel || w.shape[1] != dModel {
        fatalError("linear_pos.weight has WRONG SHAPE!")
    }

    linearPos.update(parameters: ModuleParameters.unflattened(["weight": w]))
}
```

### 3. Check for Weight Remapping Issues

Search for code that might remap or override `linear_pos.weight`:
- Look in `ChatterboxEngine.swift` weight loading
- Check `LinearFactory` for quantization issues
- Verify weight key prefixes match correctly

### 4. Trace spkEmbProj Usage

Even though spkEmbProj [1, 80] looks innocent, trace where it's used:
- Is it passed to encoder?
- Could it be getting concatenated with encoder outputs?
- Check if there's any operation mixing speaker embeddings with positional encodings

### 5. Run with Forced Evaluation

Modify main.swift to force evaluation immediately after encoder initialization:

```swift
// After encoder is created and weights loaded
for enc in s3gen!.encoder.encoders {
    let testInput = MLXArray.zeros([1, 10, 512])
    let testPE = MLXArray.zeros([1, 19, 512])  // 2*10-1
    let _ = enc(testInput, posEmb: testPE)
    eval(enc.attention.linearPos.weight)
    print("Encoder attention linearPos.weight.shape: \(enc.attention.linearPos.weight.shape)")
}
```

---

## Technical Notes

### MLX Lazy Evaluation Behavior

- Operations build a computation graph
- Graph evaluation is deferred until:
  1. Explicit `eval()` call
  2. `.shape` property access
  3. Conversion to native types (`.item()`, `.asArray()`)
  4. Scope exit / ARC cleanup (sometimes)
- **Critical:** The error location is NOT where the bug is
- The bug is where the incompatible operation was CREATED

### Broadcast Rules

MLX broadcasting follows NumPy rules:
- Shapes are compared element-wise from right to left
- Dimensions must either:
  1. Be equal, OR
  2. One of them is 1

**Why (1,80,64) vs (1,564,80) fails:**
```
Shape 1:  [  1,  80, 64]
Shape 2:  [  1, 564, 80]
          ---  ---  ---
Dim 0:     1 = 1   ✓
Dim 1:    80 ≠ 564  ❌  (neither is 1)
Dim 2:    64 ≠ 80   ❌  (neither is 1)
```

---

## Relevant Configuration

```swift
// S3GenConfig
hiddenDim: 256        // Decoder hidden dimension
melChannels: 80       // Vocoder feature dimension
numHeads: 8           // Attention heads (decoder)
headDim: 64           // Attention head dimension (decoder: 256/4=64)
inputDim: 512         // Encoder dimension

// FlowEncoder
hiddenDim: 512        // Encoder hidden dimension
melDim: 80            // Output projection dimension
numHeads: 8           // Attention heads
dHead: 512/8 = 64     // Attention head dimension

// Sequence
promptTokens: 250
speechTokens: 32
totalTokens: 282
peTableLen: 2*282-1 = 564
```

---

## Key Insight from User

**CRITICAL:** The bug is NOT in the raw `pe.weights` storage (which is correctly [1, 9999, 512]).
The bug is in a **PROJECTION** that takes those weights and projects them to the wrong dimension.

**User's Diagnosis:**
1. PE module takes raw weights (size 512)
2. Projects them to an **output dimension**
3. **BUG:** Output dimension is 80 (melChannels) instead of 64 (headDim)
4. This creates a deferred graph operation: "Take 512 weights, project to 80, wait"
5. When the graph evaluates, it tries to broadcast (1, 564, 80) with attention tensors expecting (*, *, 64)

**The Deferred Operation "Ghost":**
- **Init:** Module created with wrong output dimension (80 instead of 64)
- **Forward Pass:** Graph records "Project weights to 80, add to Attention Matrix"
- **Variable Release:** When fullTokens/tokenEmb are released or printed, graph resolves
- **Crash:** Sees 80 (PE) vs 64 (Attn) and explodes with broadcast error

## Current Investigation Status

### ✅ Verified Correct
1. EspnetRelPositionalEncoding initialization: `dModel=512` ✓
2. RelPositionMultiHeadAttention initialization: `dModel=512, numHeads=8, dHead=64` ✓
3. All safetensors positional weights: `(8, 64)` ✓
4. linearPos initialization: `Linear(512, 512)` ✓

### ❓ Still Investigating
1. **Where is the 80→64 projection happening?**
   - EspnetRelPositionalEncoding has NO projection layer
   - linearPos is 512→512, not involving 80 or 64
   - Must be somewhere else in the code

2. **Is there a weight loading bug?**
   - Could wrong weights be loaded into linearPos?
   - Could there be a dimension mismatch during loading that creates deferred broadcast?

3. **Is there another PE module we missed?**
   - Could there be a different positional encoding implementation?
   - Could the decoder have its own PE (even though it shouldn't)?

## Phase 5: Decoder Speaker Conditioning Investigation (2025-12-19 Latest)
**Result:** ✅ VERIFIED CORRECT - Decoder speaker conditioning uses correct dimensions

**User's Latest Theory:**
- Shape (1, 564, 80) is the CORRECT decoder input (noisy mel) - NOT the bug
- Shape (1, 80, 64) is the intruder from incorrect speaker conditioning
- Suspected: element-wise multiplication (*) instead of matmul in speaker embedding

**Investigation Results:**

### ✅ Verified Speaker Embedding Flow
1. **Input:** `soul_s3_192.npy` shape [1, 192] ✓
2. **Projection:** `spkEmbedAffine = Linear(192, 80)` ✓
   - Python weight: `flow.spk_embed_affine_layer.weight` [80, 192]
   - Python weight: `flow.spk_embed_affine_layer.bias` [80]
   - Swift initialization: `Linear(192, config.melChannels)` where melChannels=80
3. **Output:** `spkCond` shape [1, 80] ✓
4. **Usage in Decoder (S3Gen.swift:1159):**
   ```swift
   let spkExpanded = tiled(speakerEmb.expandedDimensions(axis: 2), repetitions: [1, 1, L])
   // [1, 80] → [1, 80, 1] → [1, 80, L] where L=564
   var h = concatenated([x, mu, spkExpanded, cond], axis: 1) // [B, 320, T]
   ```
   - Concatenation along axis 1, NOT multiplication ✓

### ✅ Verified Decoder Attention Dimensions
1. **Config:**
   - hiddenDim: 256 (decoder feature dimension)
   - numHeads: 8
   - headDim: 64
   - innerDim = 8 × 64 = 512

2. **MultiHeadAttention initialization (S3Gen.swift:1004):**
   ```swift
   self.attention = MultiHeadAttention(dims: dim, numHeads: numHeads, headDim: headDim, ...)
   // dims=256, numHeads=8, headDim=64 → innerDim=512
   ```

3. **Attention projections:**
   - Q/K/V: `Linear(256, 512)` ✓
   - Output: `Linear(512, 256)` ✓

4. **Python weights verification:**
   - `flow.decoder.estimator.*.attn1.to_q.weight`: [512, 256] ✓
   - `flow.decoder.estimator.*.attn1.to_k.weight`: [512, 256] ✓
   - `flow.decoder.estimator.*.attn1.to_v.weight`: [512, 256] ✓
   - `flow.decoder.estimator.*.attn1.to_out.0.weight`: [256, 512] ✓

### ❌ NOT FOUND
1. **No [80, 64] or [64, 80] weights exist in the model**
2. **No element-wise multiplication between speaker embedding and 64-dim tensor**
3. **No reshape operations creating [1, 80, 64]**
4. **All decoder forward pass operations use correct dimensions**

---

## Summary

**What we know:**
- Broadcast error (1,80,64) vs (1,564,80) is real, not MLX bug
- Crash triggered by lazy evaluation after Step 5 completes
- All ENCODER initialization and weights are correct
- All DECODER initialization and weights are correct
- All SPEAKER EMBEDDING flow is correct
- Bug is NOT in encoder PE linearPos (verified [512, 512] after loading)
- Bug is NOT in decoder speaker conditioning (verified correct concatenation)

**What we STILL need to find:**
- The exact line of code that creates the (1, 80, 64) tensor
- Why this tensor is created at all - no weights or operations should produce it
- Whether this is a framework behavior difference between MLX and PyTorch

**Current Status: DEADLOCKED**
- Encoder PE: Thoroughly verified ✓
- Decoder attention: Thoroughly verified ✓
- Speaker embedding: Thoroughly verified ✓
- Weight loading: All checked weights correct ✓

**Possible remaining sources:**
1. Hidden operation in MLX-Swift framework that doesn't exist in Python
2. Weight remapping bug in a section we haven't checked yet
3. Incorrect initialization somewhere outside encoder/decoder
4. Bug in a utility function (reshape, transpose, slice, etc.)
5. Interaction between multiple tensors creating unexpected broadcast
