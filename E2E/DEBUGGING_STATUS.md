# Swift TTS Port - Debugging Status

## ✅ Verified Correct

1. **Weight Loading**
   - inputEmbedding.weight: [6561, 512] ✓
   - encoder.encoders[0].feedForward.w1: values match Python ✓
   - encoderProj.weight: mean=0.000109, range=[-0.131, 0.128] ✓
   - decoder.downBlocks[0].resnet.block1.norm: mean=0.535 ✓
   - All Conv1d weights in MLX format [out, kernel, in] ✓
   - Vocoder weights loading correctly ✓

2. **ODE Integration Formula**
   - `xt = xt + v * dt` matches Python ✓
   - Cosine timestep scheduling correct ✓
   - CFG formula: `(1 + cfg) * vCond - cfg * vUncond` ✓

3. **Decoder Input Preparation**
   - Concatenation order: `[x, mu, spks, cond]` → [B, 320, T] ✓
   - All inputs in channels-first format [B, C, T] ✓

## ❌ Current Issue

**Symptom**: Decoder output is 53% positive / 47% negative when it should be 100% negative (log mel)

**ODE Behavior**:
- Initial: [-4.45, 4.60]
- Final: [-5.07, 5.14]
- **ODE is DIVERGING instead of converging!**

**Audio Output**:
- 82.4% high-frequency energy (should be <20%)
- Length: 2.72s (Python: 4.68s)
- Completely wrong voice quality

## 🔍 Likely Root Causes

Since all weights and formulas are correct, the issue is likely in:

1. **Attention Mask Computation**
   - Python uses `mask_to_bias` to convert bool to 0/-inf
   - Need to verify Swift's mask handling is identical

2. **Layer Normalization**
   - Epsilon values (Python uses eps=1e-12 for encoder, might differ for decoder)
   - Pre-norm vs post-norm order

3. **Numerical Precision**
   - FP16 vs FP32 differences accumulating across layers
   - MLX operator differences vs PyTorch

4. **Decoder Layer Implementation**
   - CausalResNetBlock forward pass
   - BasicTransformerBlock attention mechanism
   - Skip connection handling in upBlocks

## 📋 Next Steps

### Immediate Action (Most Efficient)

Generate Python reference outputs for debugging:

```python
# Run Python E2E with SAME inputs as Swift
python E2E/verify_e2e_full.py

# This will save intermediate tensors at each step:
# - step1_text_tokens.npy
# - step2_t3_conditioning.npy
# - step3_speech_tokens.npy
# - step6_encoder_out.npy (mu)
# - step7_step1_x_after.npy, step7_step2_x_after.npy, etc.
# - step7_final_mel.npy
# - step8_audio.npy
```

Then compare Swift vs Python layer-by-layer to find exact divergence point.

### Systematic Debugging

1. **Save Swift ODE intermediate states**
   - Already have: swift_generated_mel_raw.safetensors
   - Add: Save x_after for each ODE step

2. **Compare ODE Step 1**
   - If step 1 matches → bug is in accumulation
   - If step 1 differs → bug is in decoder forward pass

3. **Drill down to specific layer**
   - Add debug output for each downBlock, midBlock, upBlock
   - Compare activations layer-by-layer

### Alternative: Check Common Bugs

1. **Attention mask format**
   ```swift
   // Verify makeAttentionMask produces correct format
   // Should be: 0 for attended, -inf for masked
   ```

2. **Skip connections**
   ```swift
   // Verify skip connection dimensions match
   // Check concatenation vs addition
   ```

3. **Final projection**
   ```swift
   // Check convOut / finalProj produces 80 channels
   // Verify output is NOT clamped/clipped
   ```

## 💡 Key Insight

The ODE divergence is unusual. In proper flow matching:
- t=0: Start at noise ~N(0,1)
- t=1: Converge to target (mu-conditioned mel)
- Velocity dphi/dt should guide toward target

But Swift's velocity values ([-1.85, 1.74]) are pushing AWAY from reasonable mel range.
This suggests the decoder is either:
1. Producing garbage outputs due to layer bug
2. Not using mu conditioning correctly
3. Has wrong signs somewhere in the computation

## 🎯 Recommended Focus

**Most likely culprit**: Attention mechanism in BasicTransformerBlock

Reasons:
- Attention is the most complex component
- Small bugs in Q/K/V computation cause large downstream effects
- Mask handling is error-prone
- Shape mismatches easily occur

**Action**: Add comprehensive tracing to FlowTransformerBlock and compare with Python's BasicTransformerBlock on SAME inputs.
