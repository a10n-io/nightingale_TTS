# E2E Verification: Python/PyTorch vs Swift/MLX

This directory contains the end-to-end verification framework for comparing Python/PyTorch and Swift/MLX implementations of Chatterbox TTS.

## Overview

The verification compares intermediate outputs at each stage of the TTS pipeline:

| Stage | Description | Comparison |
|-------|-------------|------------|
| 1 | Text Tokenization | BPE tokens must match exactly |
| 2 | T3 Conditioning | Speaker/emotion embeddings (diff < 0.001) |
| 3 | T3 Token Generation | Speech tokens (Python reference) |
| 4 | S3Gen Embedding | Voice embedding (Python reference) |

## Quick Start

**Important:** Run from project root using the venv python, or use the scripts directly:

```bash
# From project root - using venv python explicitly
cd /Users/a10n/Projects/nightingale_TTS
python/venv/bin/python E2E/verify_e2e.py

# Or run directly (scripts have correct shebang)
./E2E/verify_e2e.py

# Use existing Python references, only run Swift comparison
./E2E/verify_e2e.py --swift-only

# Filter to specific test case
./E2E/verify_e2e.py --voice samantha --sentence basic_greeting --lang en
```

**Do NOT use system python3** - it won't have the chatterbox package installed.

## Scripts

### verify_e2e.py (Main Script)

Unified verification script that:
1. Generates Python reference outputs (unless `--swift-only`)
2. Runs Swift VerifyLive binary for comparison
3. Reports stage-by-stage verification results

**Arguments:**
- `--voice`, `-v`: Filter to specific voice (samantha, sujano)
- `--sentence`, `-s`: Filter to specific sentence ID
- `--lang`, `-l`: Filter to specific language (en, nl)
- `--device`, `-d`: PyTorch device (cpu, mps, cuda)
- `--swift-only`: Skip Python generation, use existing references
- `--no-swift`: Skip Swift verification, only generate Python refs

**Example Output:**
```
================================================================================
E2E VERIFICATION: Python/PyTorch vs Swift/MLX
================================================================================
Device: cpu
Seed: 42
Temperature: 0.001
Swift verification: enabled

--------------------------------------------------------------------------------
[1/20] Voice: samantha | Sentence: basic_greeting | Lang: en
  Text: "Hello world."

  Running Swift verification...
  Stage 1: Text Tokenization — ✅ VERIFIED (Diff: 0.00e+00)
    Notes: BPE tokenizer - 8 tokens - Swift MATCH
  Stage 2: T3 Conditioning — ✅ VERIFIED (Diff: 2.26e-06)
    Notes: emotion_adv=0.500, shape (1, 34, 1024)
  Stage 3: T3 Token Generation — ✅ VERIFIED (Diff: 0.00e+00)
    Notes: Generated 32 speech tokens (Python)
  Stage 4: S3Gen Embedding — ✅ VERIFIED (Diff: 0.00e+00)
    Notes: Voice embedding shape (1, 192) (Python)

================================================================================
VERIFICATION SUMMARY
================================================================================
✅ ALL TESTS PASSED
Total test cases: 20
```

### run_python_e2e.py

Standalone Python reference generator. Useful for regenerating all references without Swift comparison.

```bash
# Run directly (uses venv shebang)
./E2E/run_python_e2e.py
./E2E/run_python_e2e.py --voice samantha --sentence basic_greeting

# Or with explicit venv python
python/venv/bin/python E2E/run_python_e2e.py
```

## Test Data

### test_sentences.json

Contains 5 test sentences in English and Dutch:

| ID | Description | Purpose |
|----|-------------|---------|
| basic_greeting | Short / Baseline | Basic tokenization |
| expressive_surprise | Expressiveness / Emotion | Emotion handling |
| narrative_flow | Length / Pacing | Long-form generation |
| interrogative | Intonation / Question | Question intonation |
| technical_status | Technical / Articulation | Technical vocabulary |

### Voices

Two baked voices are tested:
- **samantha**: Female English voice
- **sujano**: Male Dutch voice

Both voices must have `emotion_adv.npy` exported (not hardcoded 0.5).

## Directory Structure

```
E2E/
├── README.md                 # This file
├── verify_e2e.py            # Main verification script
├── run_python_e2e.py        # Python reference generator
├── test_sentences.json      # Test sentences (en/nl)
└── reference_outputs/       # Generated Python references
    ├── samantha/
    │   ├── basic_greeting_en/
    │   │   ├── config.json
    │   │   ├── step1_text_tokens.npy
    │   │   ├── step2_final_cond.npy
    │   │   ├── step2_emotion_value.npy
    │   │   ├── step3_speech_tokens.npy
    │   │   └── step4_embedding.npy
    │   └── ...
    └── sujano/
        └── ...
```

## Determinism Settings

For reproducible results:
- **Seed**: 42 (set for torch, numpy, random)
- **Temperature**: 0.001 (near-deterministic, 0.0 causes NaN)
- **CFG Weight**: 0.5
- **Repetition Penalty**: 2.0

## Prerequisites

### 1. Python Virtual Environment (Required)

The scripts require a Python virtual environment with the `chatterbox` package and dependencies.

**First-time setup:**
```bash
cd /Users/a10n/Projects/nightingale_TTS/python

# Create virtual environment (if not exists)
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install chatterbox (if not already installed)
pip install chatterbox-tts
# Or install from local:
# pip install -e /path/to/chatterbox
```

**Verify the venv works:**
```bash
# Should show the venv python path
which python
# Expected: /Users/a10n/Projects/nightingale_TTS/python/venv/bin/python

# Test import
python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; print('OK')"
```

**Important:** The E2E scripts have a shebang pointing to the venv python:
```
#!/Users/a10n/Projects/nightingale_TTS/python/venv/bin/python
```

This means you can run them directly (`./E2E/verify_e2e.py`) without activating the venv first.

**Do NOT use system python3** - it won't have chatterbox installed:
```bash
# WRONG - will fail with ImportError
python3 E2E/verify_e2e.py

# CORRECT - use venv python explicitly
python/venv/bin/python E2E/verify_e2e.py

# CORRECT - or run directly (uses shebang)
./E2E/verify_e2e.py
```

### 2. Model Files (Required)

The Chatterbox model files must be present:
```
models/chatterbox/
├── t3_23lang.safetensors      # T3 model weights
├── s3gen.safetensors          # S3Gen model weights
├── vocoder.safetensors        # Vocoder weights
├── vocab.txt                  # BPE vocabulary
├── merges.txt                 # BPE merges
└── ...
```

These are downloaded automatically by `ChatterboxMultilingualTTS.from_pretrained()` or can be placed manually.

### 3. Swift Build (Required for Swift verification)

```bash
cd /Users/a10n/Projects/nightingale_TTS/swift/test_scripts/VerifyLive
swift build
```

The binary will be at `.build/debug/VerifyLive`.

### 4. Baked Voice Files (Required)

Each voice needs:
- `baked_voices/{voice}/ref_audio.wav` - Reference audio
- `baked_voices/{voice}/baked_voice.pt` - Baked embeddings
- `baked_voices/{voice}/npy/` - NPY exports for Swift

**Bake a new voice:**
```bash
cd /Users/a10n/Projects/nightingale_TTS
source python/venv/bin/activate

# Bake embeddings from reference audio
python python/bake_voice.py --voice samantha

# Export to NPY format for Swift (includes emotion_adv)
python python/convert_voice_to_npy_padded.py --voice samantha
```

**Current voices:**
- `samantha` - Female English voice
- `sujano` - Male Dutch voice

## Troubleshooting

### "No module named 'chatterbox'" or ImportError
You're using the wrong Python. Use the venv:
```bash
# Check which python you're using
which python3
# If it shows /usr/bin/python3 or similar, you need the venv

# Option 1: Run directly (uses shebang)
./E2E/verify_e2e.py

# Option 2: Use venv python explicitly
python/venv/bin/python E2E/verify_e2e.py

# Option 3: Activate venv first
source python/venv/bin/activate
python E2E/verify_e2e.py
```

### "FileNotFoundError: t3_23lang.safetensors"
Model files are missing. Either:
1. Download via `ChatterboxMultilingualTTS.from_pretrained()` first
2. Or manually place model files in `models/chatterbox/`

### Swift binary not found
Build the Swift verification tool:
```bash
cd swift/test_scripts/VerifyLive
swift build
```

### emotion_adv mismatch
Re-export the voice with emotion_adv:
```bash
source python/venv/bin/activate
python python/convert_voice_to_npy_padded.py --voice <voice_name>
```

### Temperature 0.0 causes NaN
This is expected. Use temperature=0.001 for near-deterministic behavior.

### Token count mismatch
Ensure the tokenizer vocab and merges files match between Python and Swift.

### "Permission denied" when running ./E2E/verify_e2e.py
Make the script executable:
```bash
chmod +x E2E/verify_e2e.py E2E/run_python_e2e.py
```

## Adding New Test Cases

1. Add sentence to `test_sentences.json`:
```json
{
  "id": "new_test",
  "description": "Test description",
  "text_en": "English text.",
  "text_nl": "Dutch text."
}
```

2. Regenerate references:
```bash
./E2E/verify_e2e.py --sentence new_test
```

## Adding New Voices

1. Create voice directory with reference audio:
```bash
mkdir -p baked_voices/new_voice
cp /path/to/ref_audio.wav baked_voices/new_voice/
```

2. Bake and export voice (using venv):
```bash
cd /Users/a10n/Projects/nightingale_TTS
source python/venv/bin/activate

python python/bake_voice.py --voice new_voice
python python/convert_voice_to_npy_padded.py --voice new_voice
```

3. Add voice to test matrix in `verify_e2e.py` and `run_python_e2e.py`:
```python
voices = ["samantha", "sujano", "new_voice"]
```

4. Generate references for the new voice:
```bash
./E2E/verify_e2e.py --voice new_voice
```
