# Nightingale TTS Verification Status

**Date:** 2025-12-18

## Overview

This document tracks the verification status of the Nightingale TTS pipeline across Python (reference) and Swift (port) implementations.

## Pipeline Architecture

```
Text Input
    ↓
┌─────────────────────────────────────┐
│  T3 Model (The Brain)               │
│  - Text tokenization                │
│  - Speaker conditioning             │
│  - Transformer (30 layers)          │
│  - Speech token generation          │
└─────────────────────────────────────┘
    ↓
Speech Tokens [0-6561]
    ↓
┌─────────────────────────────────────┐
│  S3Gen Model (The Mouth)            │
│  - Token embedding                  │
│  - Upsample encoder (6 conformers)  │
│  - Flow matching decoder (ODE)      │
│  - Mel2Wav vocoder (HiFiGAN-style)  │
└─────────────────────────────────────┘
    ↓
Audio Output (24kHz WAV)
```

## Python Implementation

### Status: FULLY FUNCTIONAL

| Component | Status | Notes |
|-----------|--------|-------|
| Model Loading | ✅ Working | from_local() with MPS support |
| Voice Baking | ✅ Working | Full precision from ref_audio.wav |
| T3 Generation | ✅ Working | Text → speech tokens |
| S3Gen Vocoding | ✅ Working | Tokens → audio |
| Multilingual | ✅ Working | 23+ languages supported |

### Test Command
```bash
cd /Users/a10n/Projects/nightingale_TTS/python
source venv/bin/activate
python test_baked_voice.py
```

### Output
- `test_audio/python_test_english_YYYYMMDD_HHMMSS.wav`
- `test_audio/python_test_dutch_YYYYMMDD_HHMMSS.wav`

## Swift Implementation

### Status: PARTIALLY FUNCTIONAL (T3 works, S3Gen has bug)

| Component | Status | Notes |
|-----------|--------|-------|
| Model Loading | ✅ Working | MLX safetensors format |
| Voice Loading (NPY) | ✅ Working | Converted from .pt to .npy |
| T3 Generation | ✅ Working | Text → speech tokens verified |
| S3Gen Encoder | ✅ Working | Tokens → mel features |
| S3Gen Decoder | ❌ Bug | Conv shape mismatch in ODE loop |
| Mel2Wav Vocoder | ⚠️ Untested | Blocked by decoder bug |

### Known Bug: S3Gen Decoder

**Error:**
```
[conv] Expect the input channels in the input and weight array to match
but got shapes - input: (1,140,80) and weight: (512,80,3)
```

**Analysis:**
- The error occurs during the ODE loop in `FlowMatchingDecoder`
- Debug shows `h.shape = [2, 320, 1038]` entering `down.resnet`
- But crash has completely different tensor `(1, 140, 80)`
- This is a **lazy evaluation bug** where MLX evaluates wrong tensor
- The shape `(1, 140, 80)` matches what `generatedMel` would be after ODE
- Batch size mismatch (1 vs 2) confirms wrong tensor being evaluated

**Location:** `swift/Sources/Nightingale/S3Gen.swift` - FlowMatchingDecoder

**Root Cause:** Unknown - requires MLX computation graph debugging

### Voice Format Conversion

Swift requires NPY files in a specific directory structure. Conversion script provided:

**Script:** `python/convert_voice_to_npy_padded.py`

**Input:** `baked_voices/baked_voice.pt` (PyTorch Conditionals)

**Output:** `baked_voices/baked_voice_npy/`
- `soul_t3_256.npy` - T3 speaker embedding [1, 256]
- `soul_s3_192.npy` - S3Gen speaker embedding [1, 192]
- `t3_cond_tokens.npy` - T3 conditioning tokens [1, 150]
- `prompt_token.npy` - S3Gen prompt tokens [1, 449] (padded)
- `prompt_feat.npy` - S3Gen prompt features [1, 898, 80] (padded)

### Build & Run Swift

```bash
cd /Users/a10n/Projects/nightingale_TTS/swift/test_scripts/GenerateAudio
rm -rf .build  # Clean stale cache if needed
swift build
.build/debug/GenerateAudio
```

## Model Files

### Python Models (`models/chatterbox/`)
| File | Size | Purpose |
|------|------|---------|
| ve.pt | 5.4 MB | Voice encoder |
| t3_mtl23ls_v2.safetensors | 2.0 GB | T3 multilingual model |
| s3gen.pt | 1.0 GB | S3Gen vocoder |
| grapheme_mtl_merged_expanded_v1.json | 68 KB | Multilingual tokenizer |

### Swift/MLX Models (`models/mlx/`)
| File | Size | Purpose |
|------|------|---------|
| t3_fp32.safetensors | ~2 GB | T3 model (FP32) |
| s3gen_fp16.safetensors | ~500 MB | S3Gen encoder/decoder |
| vocoder_weights.safetensors | ~50 MB | Mel2Wav vocoder |
| rope_freqs_llama3.safetensors | ~1 MB | Pre-computed RoPE |
| config.json | <1 KB | Model configuration |

## Verification Scripts

### Python Reference Generators
Located in `python/`:
- `verify_e2e_steps1_5.py` - Generate reference outputs for all stages
- `verify_nightingale_step1_tokenization.py` - Tokenization only
- `verify_nightingale_step2_conditioning.py` - T3 conditioning
- `verify_nightingale_step3_transformer.py` - T3 transformer
- `verify_nightingale_step4_token_generation.py` - First token generation
- `verify_nightingale_step5_s3gen_embedding.py` - S3Gen embedding

### Swift Verification Tests
Located in `swift/test_scripts/`:
- `VerifyStep1Tokenization/` - Compare tokenization
- `VerifyStep2Conditioning/` - Compare conditioning
- `VerifyStep3Transformer/` - Compare transformer output
- `VerifyStep4TokenGeneration/` - Compare token generation
- `VerifyE2ESteps1_4/` - Full T3 pipeline verification

## Pass/Fail Criteria

| Stage | Metric | Threshold |
|-------|--------|-----------|
| Tokenization | Token IDs | Exact match |
| Conditioning | max_diff | < 0.001 |
| Transformer | max_diff | < 0.001 |
| Token Generation | max_diff + token | < 0.001 + exact |
| S3Gen Embedding | max_diff | < 0.001 |

## Next Steps

1. **Debug S3Gen Decoder** - Investigate MLX lazy evaluation bug
2. **Fix Conv Shape Issue** - Identify where wrong tensor is being evaluated
3. **Complete Vocoder Testing** - Once decoder works
4. **Performance Optimization** - After functional parity achieved

## Reference Documentation

- [SWIFT_COMPLETE.md](SWIFT_COMPLETE.md) - Original Swift port documentation
- [E2E_VERIFICATION.md](E2E_VERIFICATION.md) - End-to-end verification guide
- [NIGHTINGALE_VERIFICATION.md](NIGHTINGALE_VERIFICATION.md) - Detailed verification methodology
