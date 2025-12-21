# Swift TTS Port - Next Steps for Full Parity

## 🎯 Current Status

### ✅ Successfully Ported & Verified
- All model weights loading correctly (encoder, decoder, vocoder)
- T3 text→speech token generation working
- Full pipeline executing end-to-end
- No crashes or runtime errors
- ODE solver implementing correct formula
- Tensor shapes and formats correct throughout

### ❌ Remaining Issue
**Decoder ODE produces invalid mel spectrograms**
- Output: 53% positive / 47% negative (should be 100% negative log mel)
- Audio: 82.4% high-frequency (should be <20%)
- ODE diverges instead of converges

## 🔍 Root Cause Hypothesis

Since all weights, formulas, and shapes are correct, the issue is in **numerical computation details**:

**Most Likely**: Subtle bug in attention mechanism or layer normalization
- Attention mask format/application
- Q/K/V projection or scaling
- Softmax numerical stability
- Skip connection handling

**Also Possible**:
- FP16 precision accumulation errors
- MLX vs PyTorch operator differences
- LayerNorm eps values mismatch

## 📋 Definitive Debugging Plan

### Step 1: Generate Python Reference (30 min)

```bash
cd E2E
python3 verify_e2e_full.py --text "Test phrase" --voice samantha --lang en
```

This saves intermediate tensors:
- `step3_speech_tokens.npy` - T3 output
- `step6_mu.npy` - Encoder output
- `step7_step1_x_after.npy` through `step7_step10_x_after.npy` - Each ODE step
- `step7_final_mel.npy` - Decoder output
- `step8_audio.npy` - Vocoder output

### Step 2: Compare ODE Steps (20 min)

```python
import numpy as np
from safetensors import safe_open

# Load Swift ODE step outputs (need to save these from Swift)
swift_step1 = np.load('E2E/swift_step1_x_after.npy')
python_step1 = np.load('E2E/reference_outputs/.../step7_step1_x_after.npy')

# Compare
diff = np.abs(swift_step1 - python_step1)
print(f"Max diff: {diff.max()}")
print(f"Mean diff: {diff.mean()}")
print(f"Correlation: {np.corrcoef(swift_step1.flat, python_step1.flat)[0,1]}")
```

**If Step 1 matches**: Bug is in accumulation across steps
**If Step 1 differs**: Bug is in decoder forward pass

### Step 3: Layer-by-Layer Comparison (Variable time)

Add comprehensive tracing to Swift decoder:

```swift
// In FlowMatchingDecoder.callAsFunction, after each block:
if debug {
    eval(h)
    print("After downBlocks[\(i)]: range=[\(h.min()), \(h.max())], mean=\(h.mean())")
    // Save to file for comparison
    try? MLX.save(arrays: ["downBlock\(i)_out": h], url: debugURL)
}
```

Compare with Python:
```python
# Add to Python decoder.py
print(f"After down_blocks[{i}]: {x.min():.4f} to {x.max():.4f}, mean={x.mean():.4f}")
```

Find first layer where outputs diverge → that's where the bug is.

### Step 4: Fix the Bug (Variable time)

Common fixes:
- Attention mask: Ensure 0=attend, -inf=masked
- LayerNorm: Match eps values (encoder uses 1e-12, decoder might differ)
- Skip connections: Verify concatenation vs addition
- Scaling factors: Check attention scale, layer scale, residual scale

## 🚀 Quick Wins to Try First

### A. Check LayerNorm Epsilon

```swift
// In S3Gen.swift, search for LayerNorm initialization
// Encoder uses eps=1e-12, verify decoder does too
```

### B. Verify Attention Scaling

```swift
// In FlowTransformerBlock attention:
let scale = 1.0 / sqrt(Float(headDim))  // Should match Python
```

### C. Check Final Projection

```swift
// Decoder should NOT apply activation/clamp to final output
let out = finalProj(h)  // Should be raw values, no tanh/sigmoid
```

## 💡 Alternative: Use Python Encoder Output

To isolate decoder vs encoder:

```python
# Python: Save encoder output for Swift's speech tokens
python_mu = encoder(swift_speech_tokens)
np.save('swift_speech_tokens_python_mu.npy', python_mu.numpy())
```

```swift
// Swift: Load Python's encoder output
let pythonMu = try NPYLoader.load(...)
// Run decoder with Python's mu
let mel = decoder(pythonMu, ...)
```

If audio is still wrong → decoder bug
If audio is correct → encoder bug

## 📊 Success Criteria

1. **Decoder output**: 100% negative values, range ≈[-10, -2]
2. **Audio spectrum**: >70% energy below 2kHz
3. **Audio length**: Matches Python ±5%
4. **Perceptual quality**: Recognizable speech, correct voice

## 🛠️ Tools Already Created

- `E2E/compare_encoder_outputs.py` - Analyzes mel output format
- `E2E/DEBUGGING_STATUS.md` - Comprehensive verification checklist
- Debug flags in Swift code for tracing ODE steps

## 📞 Getting Help

If stuck after systematic debugging:
1. Share ODE step outputs (both Swift and Python)
2. Share first diverging layer outputs
3. Verify MLX version matches development version
4. Check if issue reproduces with FP32 (rules out precision)

---

**Bottom Line**: All infrastructure is correct. The bug is a subtle numerical issue in one specific layer. Systematic layer-by-layer comparison will find it within a few hours.
