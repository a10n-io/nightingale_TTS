# End-to-End Verification Scripts

## Purpose

These scripts provide a simple way to verify that the Nightingale pipeline produces identical results to the Python reference implementation. They're useful for:
- Regression testing after code changes
- Validating new environments or builds
- Debugging pipeline issues
- Ensuring cross-platform consistency

## Quick Start

### Run Complete Verification (Steps 1-5)

```bash
# 1. Generate Python reference outputs
cd /Users/a10n/Projects/nightingale/python
python3 verify_e2e_steps1_5.py

# 2. Run Swift verification against reference
cd ../swift/test_scripts/VerifyE2ESteps1_5
swift run
```

### Run Verification (Steps 1-4 Only)

For testing just the T3 pipeline without S3Gen:

```bash
# 1. Generate Python reference outputs
cd /Users/a10n/Projects/nightingale/python
python3 verify_e2e_steps1_4.py

# 2. Run Swift verification against reference
cd ../swift/test_scripts/VerifyE2ESteps1_4
swift run
```

Expected output:
```
================================================================================
END-TO-END VERIFICATION: STEPS 1-4
================================================================================
Test text: "Hello world"

================================================================================
STEP 1: TEXT TOKENIZATION
================================================================================
Token IDs: [284, 18, 84, 28, 179, 79]
Token count: 6

Comparison:
  Python tokens: [284, 18, 84, 28, 179, 79]
  Swift tokens:  [284, 18, 84, 28, 179, 79]
  Match: ✅ YES

================================================================================
STEP 2: T3 CONDITIONING
================================================================================
speaker_token: [1, 1, 1024]
perceiver_out: [1, 32, 1024]
emotion_token: [1, 1, 1024]
final_cond: [1, 34, 1024]

Comparison:
  speaker_token max_diff: 8.38e-09
  perceiver_out max_diff: 2.15e-06
  emotion_token max_diff: 0.00e+00
  final_cond max_diff: 2.15e-06

================================================================================
STEP 3: T3 TRANSFORMER
================================================================================
text_tokens with SOT/EOT: [3, 284, 18, 84, 28, 179, 79, 4]
text_emb (with pos): [1, 8, 1024]
transformer_input: [1, 42, 1024]
transformer_output: [1, 42, 1024]
text_hidden: [1, 8, 1024]

Comparison:
  transformer_input max_diff: 2.15e-06
  transformer_output max_diff: 1.39e-05
  text_hidden max_diff: 1.39e-05

================================================================================
STEP 4: T3 TOKEN GENERATION (First Step)
================================================================================
CFG weight: 0.5
BOS token: 6561
bos_emb (with pos): [1, 1, 1024]
cond_input: [1, 43, 1024]
uncond_input: [1, 43, 1024]
batched_input (CFG): [2, 43, 1024]
Hybrid mask: [43, 43]
transformer_output: [2, 43, 1024]
speech_head logits: [2, 1, 8194]
  cfg_logits: [8194]
Greedy prediction (argmax): 3704

Comparison:
  cond_input max_diff: 2.15e-06
  uncond_input max_diff: 2.15e-06
  batched_input max_diff: 2.15e-06
  transformer_output max_diff: 6.47e-05
  cond_logits max_diff: 2.62e-05
  uncond_logits max_diff: 4.14e-05
  cfg_logits max_diff: 2.54e-05

================================================================================
VERIFICATION SUMMARY
================================================================================
Step 1 (Tokenization): ✅ PASSED
Step 2 (Conditioning): ✅ PASSED (max_diff < 0.001)
Step 3 (Transformer): ✅ PASSED (max_diff < 0.001)
Step 4 (Token Generation): ✅ PASSED (max_diff < 0.001, token == 3704)
================================================================================
✅ ALL TESTS PASSED
================================================================================
```

Expected output for Steps 1-5:
```
================================================================================
END-TO-END VERIFICATION: STEPS 1-5
================================================================================
Test text: "Hello world"

... [Steps 1-4 output same as above] ...

================================================================================
STEP 5: S3GEN EMBEDDING
================================================================================
Loading S3Gen weights...
input_embedding.weight shape: (6561, 512)
spk_embed_affine.weight shape: (80, 192)
spk_embed_affine.bias shape: (80,)
Test speech tokens: [[3704, 3705, 3706, 3707, 3708]]
Full tokens shape: (1, 454)
Mask shape: (1, 454, 1)
Token embedding shape: (1, 454, 512)
Token embedding[0,0,:5]: [0.0023250579833984375, -0.02459716796875, ...]
After mask shape: (1, 454, 512)
Speaker embedding shape: (1, 80)
Speaker embedding[:5]: [-0.11062975972890854, 0.15128496289253235, ...]

================================================================================
VERIFICATION SUMMARY
================================================================================
✅ Step 1 (Tokenization): Generated reference
✅ Step 2 (Conditioning): Generated reference
✅ Step 3 (Transformer): Generated reference
✅ Step 4 (Token Generation): Generated reference
✅ Step 5 (S3Gen Embedding): Generated reference
```

## Individual Stage Verification

You can also verify individual stages for more targeted testing:

### Stage 1: Text Tokenization

```bash
# Python reference
cd /Users/a10n/Projects/nightingale/python
python3 verify_nightingale_step1_tokenization.py

# Swift verification
cd ../swift/test_scripts/VerifyStep1Tokenization
swift run
```

### Stage 2: T3 Conditioning

```bash
# Python reference
cd /Users/a10n/Projects/nightingale/python
python3 verify_nightingale_step2_conditioning.py

# Swift verification
cd ../swift/test_scripts/VerifyStep2Conditioning
swift run
```

### Stage 3: T3 Transformer

```bash
# Python reference
cd /Users/a10n/Projects/nightingale/python
python3 verify_nightingale_step3_transformer.py

# Swift verification
cd ../swift/test_scripts/VerifyStep3Transformer
swift run
```

### Stage 4: T3 Token Generation

```bash
# Python reference
cd /Users/a10n/Projects/nightingale/python
python3 verify_nightingale_step4_token_generation.py

# Swift verification
cd ../swift/test_scripts/VerifyStep4TokenGeneration
swift run
```

### Stage 5: S3Gen Embedding

```bash
# Python reference
cd /Users/a10n/Projects/nightingale/python
python3 verify_nightingale_step5_s3gen_embedding.py

# Swift verification
cd ../swift/test_scripts/VerifyStep5S3GenEmbedding
swift run
```

## File Structure

```
nightingale/
├── python/
│   ├── verify_e2e_steps1_5.py          # E2E reference generator (Steps 1-5)
│   ├── verify_e2e_steps1_4.py          # E2E reference generator (Steps 1-4)
│   ├── verify_nightingale_step1_tokenization.py
│   ├── verify_nightingale_step2_conditioning.py
│   ├── verify_nightingale_step3_transformer.py
│   ├── verify_nightingale_step4_token_generation.py
│   └── verify_nightingale_step5_s3gen_embedding.py
│
├── swift/test_scripts/
│   ├── VerifyE2ESteps1_5/              # E2E verification (Steps 1-5)
│   │   ├── main.swift
│   │   ├── Package.swift
│   │   └── default.metallib
│   ├── VerifyE2ESteps1_4/              # E2E verification (Steps 1-4)
│   │   ├── main.swift
│   │   ├── Package.swift
│   │   └── default.metallib
│   ├── VerifyStep1Tokenization/
│   ├── VerifyStep2Conditioning/
│   ├── VerifyStep3Transformer/
│   ├── VerifyStep4TokenGeneration/
│   └── VerifyStep5S3GenEmbedding/
│
└── verification_outputs/
    ├── e2e_steps1_5/                   # E2E reference outputs (Steps 1-5)
    │   ├── step1_text_tokens.npy
    │   ├── step2_speaker_token.npy
    │   ├── step2_perceiver_out.npy
    │   ├── step2_emotion_token.npy
    │   ├── step2_final_cond.npy
    │   ├── step3_transformer_input.npy
    │   ├── step3_transformer_output.npy
    │   ├── step3_text_hidden.npy
    │   ├── step4_cond_input.npy
    │   ├── step4_uncond_input.npy
    │   ├── step4_batched_input.npy
    │   ├── step4_transformer_output.npy
    │   ├── step4_cond_logits.npy
    │   ├── step4_uncond_logits.npy
    │   ├── step4_cfg_logits.npy
    │   ├── step5_full_tokens.npy
    │   ├── step5_mask.npy
    │   ├── step5_token_emb.npy
    │   ├── step5_spk_emb.npy
    │   └── test_config.txt
    ├── e2e_steps1_4/                   # E2E reference outputs (Steps 1-4)
    └── [other stage outputs...]
```

## Customizing Tests

You can modify the test inputs by editing the Python scripts:

```python
# In verify_e2e_steps1_5.py
verify_e2e_steps1_5(
    test_text="Your custom text here",
    voice_path="baked_voices/samantha_full",
    s3gen_weights_path="../chatterbox claude/ChatterboxApp/AppAssets/models/chatterbox/s3gen_fp16.safetensors",
    output_dir="verification_outputs/e2e_steps1_5"
)
```

## Pass/Fail Criteria

### Step 1 (Tokenization)
- **PASS**: Token IDs match exactly (max_diff = 0.0)
- **FAIL**: Any token ID differs

### Step 2 (Conditioning)
- **PASS**: max_diff < 0.001 for all components
- **FAIL**: max_diff >= 0.001 for any component

Components checked:
- `speaker_token`: Speaker embedding projection
- `perceiver_out`: Perceiver resampler output
- `emotion_token`: Emotion adversarial FC output
- `final_cond`: Final conditioning (concatenation of above)

### Step 3 (Transformer)
- **PASS**: max_diff < 0.001 for all components
- **FAIL**: max_diff >= 0.001 for any component

Components checked:
- `transformer_input`: Input to transformer ([conditioning | text with SOT/EOT])
- `transformer_output`: Output after 30 LLaMA layers + final norm
- `text_hidden`: Text portion only (excluding conditioning tokens)

### Step 4 (Token Generation)
- **PASS**: max_diff < 0.001 for all components AND greedy token matches exactly
- **FAIL**: max_diff >= 0.001 for any component OR greedy token differs

Components checked:
- `cond_input`: Conditioned input ([conditioning | text | BOS])
- `uncond_input`: Unconditioned input ([null_conditioning | text | BOS])
- `batched_input`: Batched CFG input ([cond, uncond])
- `transformer_output`: Output after transformer with hybrid mask
- `cond_logits`: Conditioned logits from speech head
- `uncond_logits`: Unconditioned logits from speech head
- `cfg_logits`: Final CFG-combined logits (uncond + cfg_weight * (cond - uncond))
- `greedy_token`: Argmax of cfg_logits (should be 3704 for "Hello world")

### Step 5 (S3Gen Embedding)
- **PASS**: max_diff < 0.001 for all components
- **FAIL**: max_diff >= 0.001 for any component

Components checked:
- `full_tokens`: Concatenated prompt tokens + speech tokens
- `mask`: Sequence mask for variable length sequences
- `token_emb`: Embedded tokens after input embedding layer
- `spk_emb`: Speaker embedding after normalize + affine projection

## Troubleshooting

### Error: "Failed to load the default metallib"

Make sure `default.metallib` exists in the test directory:
```bash
cd swift/test_scripts/VerifyE2ESteps1_4
ls -la default.metallib
```

If missing, regenerate it:
```bash
cd /Users/a10n/Projects/nightingale/swift
./generate_metallib.sh
cp default.metallib test_scripts/VerifyE2ESteps1_4/
```

### Error: "No such file or directory: verification_outputs/..."

Run the Python reference script first:
```bash
cd python
python3 verify_e2e_steps1_4.py
```

### Test Fails After Code Changes

1. Verify the Python reference is current:
   ```bash
   cd python
   python3 verify_e2e_steps1_4.py
   ```

2. Check if the failure is expected (i.e., you changed the algorithm)

3. For unexpected failures, use individual stage tests to isolate the issue

## Adding New Stages

To add verification for additional pipeline stages (e.g., Step 6: S3Gen Encoder):

1. Create Python reference script:
   ```python
   # python/verify_nightingale_step6_s3gen_encoder.py
   # Generate reference outputs for new stage
   ```

2. Create Swift verification script:
   ```swift
   // swift/test_scripts/VerifyStep6S3GenEncoder/main.swift
   // Compare against Python reference
   ```

3. Update E2E script to include new stage:
   - Update `verify_e2e_steps1_5.py` to `verify_e2e_steps1_6.py`
   - Add stage section to E2E Python script
   - Add stage section to Swift E2E test (VerifyE2ESteps1_6)
   - Update this documentation file

## Reference

- Main documentation: [NIGHTINGALE_VERIFICATION.md](NIGHTINGALE_VERIFICATION.md)
- Stage 2 investigation: [STEP2_PERCEIVER_ISSUE.md](STEP2_PERCEIVER_ISSUE.md)
- Verification methodology: [VERIFICATION_V2.md](VERIFICATION_V2.md)
