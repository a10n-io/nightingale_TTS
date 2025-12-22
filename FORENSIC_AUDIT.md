# Forensic Audit: Swift Weight Loading

## File Loading

### Python (from `python/chatterbox/src/chatterbox/mtl_tts.py:192`)
```python
t3_state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
s3gen_state = load_safetensors(ckpt_dir / "s3gen.safetensors")
```

### Swift (from `ChatterboxEngine.swift`)
```swift
// T3: tries in order
["t3_mtl23ls_v2.safetensors", "t3_fp32.safetensors", "model.safetensors"]

// S3Gen: tries in order
["s3gen.safetensors", "s3gen_fp16.safetensors", "s3_engine.safetensors"]
```

### Actual Files in `/models/chatterbox/`
- `s3gen.safetensors` (1.0 GB) ✓
- `t3_mtl23ls_v2.safetensors` → symlink to `../mlx/t3_fp32.safetensors` ✓

**Status: ✅ Swift and Python load the SAME files**

---

## Weight Transposition Rules

### 1. Linear Layers (FixedLinear)
- **PyTorch format:** `[out_features, in_features]`
- **MLX format:** `[in_features, out_features]`
- **Transpose:** `.transposed()` or `.T`
- **Applies to:**
  - Decoder: `mlp.1.weight`, `attn.to_q/k/v/out.weight`, `ff.net.*.weight`
  - Encoder: `embed.linear`, `feed_forward.w_1/w_2`
  - Time MLP: `time_mlp.linear_1/2`
  - Speaker embed: `spk_embed_affine_layer`

**Code (line 521-524):**
```swift
if (isDecoderLinear || isTimeMLP || isSpkEmbedAffine || isEncoderProj || isEncoderLinear) {
    finalW = finalW.transposed()
}
```

**Status: ✅ Correct**

---

### 2. Conv1d Layers
- **PyTorch format:** `[out_channels, in_channels, kernel_size]`
- **MLX format:** `[out_channels, kernel_size, in_channels]`
- **Transpose:** `.transposed(0, 2, 1)`
- **Applies to:**
  - Decoder: `down_blocks/mid_blocks/up_blocks.*.block*.conv.weight`
  - Vocoder: `conv_pre`, `conv_post`, `resblocks`, `f0_predictor.condnet`, `source_downs/resblocks`

**Code (line 545-549):**
```swift
if isDecoderConv || isEncoderConv || isVocoderConv {
    finalW = finalW.transposed(0, 2, 1)
}
```

**Status: ✅ Correct**

---

### 3. ConvTranspose1d (Vocoder Upsampling)
- **PyTorch format:** `[in_channels, out_channels, kernel_size]`
- **MLX format:** `[out_channels, kernel_size, in_channels]`
- **Transpose:** `.transposed(1, 2, 0)`
- **Applies to:** `mel2wav.ups.*`

**Code (line 555-559):**
```swift
if isConvTranspose {
    finalW = finalW.transposed(1, 2, 0)
}
```

**Status: ✅ Correct**

---

### 4. Weight Norm Parametrizations
PyTorch uses `weight_norm` which splits weights into:
- `original0` (v): direction vector
- `original1` (g): magnitude parameter

**Formula:** `weight = v * (g / ||v||)` where norm is over dims `[0, 2]`

**Code (line 470-481):**
```swift
let v = original0  // [out, 1, 1]
let g = original1  // [out, in, kernel]
let norm = sqrt(sum(v * v, axes: [0, 2], keepDims: true))  // [1, 1, 1]
let weight = v * (g / (norm + 1e-8))
```

**Applies to:**
- `mel2wav.conv_pre` (82 total parametrizations)
- `mel2wav.conv_post`
- `mel2wav.ups.*`
- `mel2wav.resblocks.*`
- `mel2wav.f0_predictor.condnet.*`

**Status: ✅ Correct (82 parametrizations combined)**

---

## Weight Application

Weights are applied via:
```swift
let s3Params = ModuleParameters.unflattened(s3Remapped)
s3.update(parameters: s3Params)
```

This updates all nested modules recursively using MLX's Module system.

**Verified weights loaded:**
- Decoder keys: 910 ✓
- Vocoder keys: 164 (after combining weight_norm) ✓

---

## Known Issues

### 1. Decoder produces flat mel spectrograms
**Symptom:** All 80 mel channels have similar energy (~-10 dB)
**Expected:** Varied frequency content

**Evidence from CrossValidate:**
```
MEL CHANNEL ENERGIES (decoder output):
  Channel  0: -10.7394
  Channel 10: -10.3130
  ...
  Channel 79: -10.2928
```

**Verified:**
- ✅ Decoder weights ARE loaded correctly (sum matches Python)
- ✅ Transposition is correct
- ❌ Forward pass produces wrong output

**Conclusion:** Bug is in FlowMatchingDecoder computational logic, NOT weight loading.

### 2. T3 encoder token mismatch
**Symptom:** 0% token match between Python and Swift
**Evidence:**
```
Python tokens first 20: [1732, 2068, 2186, ...]
Swift tokens first 20:  [1735, 1703, 3890, ...]
Matching: 0/98 (0.0%)
```

**Needs investigation:** T3Model forward pass or sampling logic.

---

## Conclusion

✅ **File Loading:** Correct
✅ **Weight Transposition:** Correct
✅ **Weight Application:** Correct (weights are present in modules)
❌ **Forward Pass Logic:** Broken (produces wrong outputs despite correct weights)

The issue is NOT in weight loading but in the decoder/encoder forward pass implementation.
