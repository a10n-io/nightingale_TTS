# E2E Verification: Python/PyTorch vs Swift/MLX

This directory contains the end-to-end verification framework for comparing Python/PyTorch and Swift/MLX implementations of Chatterbox TTS.

## Overview

The verification compares intermediate outputs at each stage of the TTS pipeline:

| Stage | Description | Swift Status | Tolerance |
|-------|-------------|--------------|-----------|
| 1 | Text Tokenization | ✅ Verified | Exact match |
| 2 | T3 Conditioning | ✅ Verified | diff < 1e-6 |
| 3 | T3 Token Generation | ✅ Verified | Exact match (greedy) |
| 4 | S3Gen Embedding | ✅ Reference | Python reference |
| 5 | S3Gen Input Prep | ⏸️ Framework ready | diff < 0.01 |
| 6 | S3Gen Encoder | ⚠️ Known issue | See notes |
| 7 | ODE Solver | ⏸️ Framework ready | diff < 2.0 |
| 8 | Vocoder | ⏸️ Framework ready | diff < 0.1 |

### Known Issues

**Swift UpsampleEncoder Attention Bug**: The Swift encoder has a shape mismatch in the upEncoder attention layers:
```
Shapes (1,80,64) and (1,564,80) cannot be broadcast
```
This affects full numerical verification for stages 5-8. The framework is complete but S3Gen model loading is currently skipped.

## Quick Start

**Important:** Run from project root using the venv python, or use the script directly:

```bash
# From project root - using venv python explicitly
cd /Users/a10n/Projects/nightingale_TTS
python/venv/bin/python E2E/verify_e2e.py

# Or run directly (script has correct shebang)
./E2E/verify_e2e.py

# Test only Steps 1-3 (tokenization, conditioning, generation)
./E2E/verify_e2e.py --steps 3

# Test only Step 1 (tokenization) - fastest verification
./E2E/verify_e2e.py --steps 1

# Test with linguistic Unicode tests (22 languages, 132 test cases)
./E2E/verify_e2e.py --steps 1 --linguistic

# Test specific language from linguistic set
./E2E/verify_e2e.py --steps 1 --linguistic --lang ar   # Arabic
./E2E/verify_e2e.py --steps 1 --linguistic --lang ja   # Japanese
./E2E/verify_e2e.py --steps 1 --linguistic --lang ru   # Russian

# Use existing Python references, only run Swift comparison
./E2E/verify_e2e.py --swift-only

# Filter to specific test case
./E2E/verify_e2e.py --voice samantha --sentence expressive_surprise --lang en

# Generate Python references only (no Swift verification)
./E2E/verify_e2e.py --no-swift
```

**Do NOT use system python3** - it won't have the chatterbox package installed.

## verify_e2e.py

Unified verification script that:
1. Generates Python reference outputs (unless `--swift-only`)
2. Runs Swift VerifyLive for comparison
3. Reports stage-by-stage verification results

**Arguments:**
- `--voice`, `-v`: Filter to specific voice (samantha, sujano)
- `--sentence`, `-s`: Filter to specific sentence ID
- `--lang`, `-l`: Filter to specific language (en, nl for standard tests; ar, zh, da, nl, en, fi, fr, de, el, he, hi, it, ja, ko, ms, no, pl, pt, ru, es, sv, sw, tr for linguistic tests)
- `--device`, `-d`: PyTorch device (cpu, mps, cuda). Use `cpu` for deterministic results.
- `--steps`: Max step to generate/verify (1-8). Default: 5. Use `--steps 3` for steps 1-3.
- `--linguistic`: Use Unicode linguistic test file (22 languages, 132 test cases) instead of standard tests (2 languages, 20 test cases)
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
[1/16] Voice: samantha | Sentence: expressive_surprise | Lang: en
  Text: "Wow! I absolutely cannot believe that it worked on the first try!"

  Running Swift verification...
  Stage 1: Text Tokenization — ✅ VERIFIED (Diff: 0.00e+00)
    Notes: BPE tokenizer - 8 tokens - Swift MATCH
  Stage 2: T3 Conditioning — ✅ VERIFIED (Diff: 2.26e-07)
    Notes: emotion_adv=0.500, shape (1, 34, 1024)
  Stage 3: T3 Token Generation — ✅ VERIFIED (Diff: 0.00e+00)
    Notes: 32 tokens - Swift PASS

================================================================================
VERIFICATION SUMMARY
================================================================================
✅ ALL TESTS PASSED
Total test cases: 16
```

## Test Data

### test_sentences.json (Standard Test Set)

Contains 4 test sentences in English and Dutch:

| ID | Description | Purpose |
|----|-------------|---------|
| expressive_surprise | Expressiveness / Emotion | Emotion handling |
| narrative_flow | Length / Pacing | Long-form generation |
| interrogative | Intonation / Question | Question intonation |
| technical_status | Technical / Articulation | Technical vocabulary |

**Test matrix:** 2 voices × 4 sentences × 2 languages = **16 test cases**

### test_sentences_unicode_linguistic.json (Linguistic Unicode Test Set)

Contains 22 languages with complex Unicode features specifically designed to test NFKD normalization and Unicode scalar handling. Each language has 3 text variants with increasing complexity.

**Languages tested:**
- **Arabic (ar)**: Ligatures & Harakat (vowel diacritics)
- **Chinese (zh)**: CJK Unified Ideographs & full-width punctuation
- **Danish (da)**: Special vowels (Å, Ø, Æ)
- **Dutch (nl)**: Digraphs (ij) & trema (ë)
- **English (en)**: Ligatures & typographic punctuation
- **Finnish (fi)**: Double vowels & umlauts (Ä, Ö)
- **French (fr)**: Ligatures (œ) & cedilla (ç)
- **German (de)**: Eszett (ß) vs SS & umlauts
- **Greek (el)**: Final sigma (ς) vs medial (σ) & tones
- **Hebrew (he)**: Nikkud (vowel points)
- **Hindi (hi)**: Conjuncts & virama (complex rendering)
- **Italian (it)**: Accented finals (à, è, ì, ò, ù)
- **Japanese (ja)**: Kanji, hiragana, katakana & full-width
- **Korean (ko)**: Hangul Jamo composition
- **Malay (ms)**: Loan words & standard Latin
- **Norwegian (no)**: Vowels (Å, Ø, Æ)
- **Polish (pl)**: Ogonek (ą, ę) & slash (ł)
- **Portuguese (pt)**: Tilde (ã, õ) & cedilla (ç)
- **Russian (ru)**: Cyrillic yo (ё) & hard/soft signs
- **Spanish (es)**: Inverted marks (¿, ¡) & enye (ñ)
- **Swedish (sv)**: Vowels (Å, Ä, Ö)
- **Swahili (sw)**: Agglutinative morphology
- **Turkish (tr)**: Dotted/dotless I (ı, İ) & cedilla

**Test matrix:** 2 voices × 22 languages × 3 variants = **132 test cases**

**Usage:**
```bash
# Run all linguistic tests
./E2E/verify_e2e.py --steps 1 --linguistic

# Test specific language
./E2E/verify_e2e.py --steps 1 --linguistic --lang ar   # 6 test cases (2 voices × 3 variants)
./E2E/verify_e2e.py --steps 1 --linguistic --lang ja   # 6 test cases
```

**Note:** The linguistic test set is designed for Step 1 (tokenization) verification and tests critical Unicode edge cases including NFKD normalization, combining marks, and grapheme cluster handling.

### Voices

Two baked voices are tested:
- **samantha**: Female English voice
- **sujano**: Male Dutch voice

Both voices must have `emotion_adv.npy` exported (not hardcoded 0.5).

## Directory Structure

```
E2E/
├── README.md                              # This file
├── verify_e2e.py                          # Main verification script
├── check_weight_keys.py                   # Weight key debugging utility
├── test_sentences.json                    # Standard test sentences (en/nl, 16 tests)
├── test_sentences_unicode_linguistic.json # Linguistic Unicode tests (22 languages, 132 tests)
└── reference_outputs/                     # Generated Python references
    ├── samantha/
    │   ├── expressive_surprise_en/
    │   │   ├── config.json
    │   │   ├── step1_text_tokens.npy
    │   │   ├── step1_text_tokens_cfg.npy
    │   │   ├── step2_final_cond.npy
    │   │   ├── step2_emotion_value.npy
    │   │   ├── step3_speech_tokens.npy
    │   │   └── ...
    │   └── ...
    └── sujano/
        └── ...
```

## Swift Verification Tool

The Swift verification tool is located at `swift/test_scripts/VerifyLive/`.

### Building

```bash
cd swift/test_scripts/VerifyLive
swift build
```

### Running Manually

The `verify_e2e.py` script calls Swift automatically, but you can also run it manually:

```bash
cd swift/test_scripts/VerifyLive
swift run VerifyLive --voice samantha --ref-dir ../../../E2E/reference_outputs/samantha/expressive_surprise_en
```

### Current Status

| Stage | Swift Implementation | Notes |
|-------|---------------------|-------|
| 1-3 | Full numerical verification | Exact match required |
| 5-8 | Framework complete | S3Gen skipped due to encoder bug |

## Determinism Settings

For reproducible results:
- **Seed**: 42 (set for torch, numpy, random)
- **Temperature**: 0.001 (triggers greedy/argmax decoding)
- **CFG Weight**: 0.5
- **Repetition Penalty**: 2.0
- **n_timesteps**: 10 (ODE solver steps)
- **cfg_rate**: 0.7 (classifier-free guidance rate)

**Note:** With temperature <= 0.01, both Python and Swift use greedy decoding (argmax) instead of multinomial sampling, ensuring deterministic token generation.

## Prerequisites

### 1. Python Virtual Environment (Required)

The script requires a Python virtual environment with the `chatterbox` package and dependencies.

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

**Important:** The E2E script has a shebang pointing to the venv python:
```
#!/Users/a10n/Projects/nightingale_TTS/python/venv/bin/python
```

This means you can run it directly (`./E2E/verify_e2e.py`) without activating the venv first.

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
├── t3_mtl23ls_v2.safetensors  # T3 model weights
├── s3gen.pt                    # S3Gen model weights
├── grapheme_mtl_merged_expanded_v1.json  # Tokenizer
└── ...
```

These are downloaded automatically by `ChatterboxMultilingualTTS.from_pretrained()` or can be placed manually.

### 3. Swift Build (Required for Swift verification)

```bash
cd /Users/a10n/Projects/nightingale_TTS/swift/test_scripts/VerifyLive
swift build
```

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

## Troubleshooting

### "No module named 'chatterbox'" or ImportError
You're using the wrong Python. Use the venv:
```bash
# Option 1: Run directly (uses shebang)
./E2E/verify_e2e.py

# Option 2: Use venv python explicitly
python/venv/bin/python E2E/verify_e2e.py

# Option 3: Activate venv first
source python/venv/bin/activate
python E2E/verify_e2e.py
```

### "RuntimeError: Attempting to deserialize object on a CUDA device"
The model was saved on CUDA but you're loading on CPU. This is fixed in the code with `map_location=device`.

### Swift binary not found
Build the Swift verification tool:
```bash
cd swift/test_scripts/VerifyLive
swift build
```

### Step 3 tokens don't match
With temperature=0.001, both implementations use greedy decoding and tokens MUST match exactly. Any mismatch indicates a bug in the T3 model implementation. Check:
1. Text tokenization matches (Step 1)
2. Conditioning matches (Step 2)
3. Model weights are identical

### "Permission denied" when running ./E2E/verify_e2e.py
Make the script executable:
```bash
chmod +x E2E/verify_e2e.py
```

## Adding New Test Cases

### Standard Test Set

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

### Linguistic Unicode Test Set

1. Add language entry to `test_sentences_unicode_linguistic.json`:
```json
{
  "id": "vi_complex",
  "description": "Vietnamese: Tone Marks & Combined Diacritics",
  "text_vi_1": "Xin chào thế giới.",
  "text_vi_2": "Tiếng Việt có sáu thanh điệu khác nhau.",
  "text_vi_3": "Tôi đang học tiếng Việt ở trường đại học."
}
```

2. Regenerate references:
```bash
./E2E/verify_e2e.py --steps 1 --linguistic --lang vi
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

3. Add voice to test matrix in `verify_e2e.py`:
```python
voices = ["samantha", "sujano", "new_voice"]
```

4. Generate references for the new voice:
```bash
./E2E/verify_e2e.py --voice new_voice
```
