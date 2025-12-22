# Cross-Validation Testing for Python/Swift Parity

This document explains the cross-validation testing framework for verifying parity between the Python (PyTorch) and Swift (MLX) implementations of Chatterbox TTS.

## Overview

The Chatterbox TTS pipeline has two main stages:

1. **T3 (Encoder)**: Converts text → speech tokens
2. **S3Gen (Decoder)**: Converts speech tokens → audio waveform

```
Text → [T3] → Speech Tokens → [S3Gen] → Audio
```

When the Swift implementation produces incorrect audio, we need to determine which stage is broken. Cross-validation isolates the problem by mixing Python and Swift components.

## Test Matrix

The cross-validation generates 4 audio files by combining T3 and S3Gen from both implementations:

| File | T3 (Encoder) | S3Gen (Decoder) | Purpose |
|------|--------------|-----------------|---------|
| `python_tokens_python_audio.wav` | Python | Python | Baseline (should be perfect) |
| `python_tokens_swift_audio.wav` | Python | Swift | Tests Swift S3Gen decoder |
| `swift_tokens_swift_audio.wav` | Swift | Swift | Full Swift pipeline |
| `swift_tokens_python_audio.wav` | Swift | Python | Tests Swift T3 encoder |

## Diagnosis Guide

Listen to each file and compare:

| If this sounds bad... | Then this is broken |
|-----------------------|---------------------|
| `python_tokens_swift_audio.wav` | Swift S3Gen (decoder) |
| `swift_tokens_python_audio.wav` | Swift T3 (encoder) |
| Both swift files | Both components |
| Only `swift_tokens_swift_audio.wav` | Interaction bug |

## Running the Tests

### Step 1: Generate Python tokens and baseline audio

```bash
python E2E/cross_validate_python.py
```

This creates:
- `python_speech_tokens.safetensors` - tokens for Swift to load
- `python_tokens_python_audio.wav` - baseline audio

### Step 2: Generate Swift tokens and cross-validate

```bash
cd swift && swift run -c release CrossValidate
```

This creates:
- `swift_speech_tokens.safetensors` - tokens for Python to load
- `swift_tokens_swift_audio.wav` - full Swift pipeline
- `python_tokens_swift_audio.wav` - Python tokens through Swift S3Gen

### Step 3: Complete cross-validation with Python

```bash
python E2E/cross_validate_python.py
```

Run again to create:
- `swift_tokens_python_audio.wav` - Swift tokens through Python S3Gen

## Output Directory

All files are saved to: `test_audio/cross_validate/`

## Token Comparison

The scripts also compare speech tokens between implementations:

```
Python tokens first 20: [1732, 2068, 2186, 1457, 680, ...]
Swift tokens first 20:  [1735, 1703, 3890, 6077, 6158, ...]

Matching: 0/98 (0.0%)
First diff at index 0: Python=1732, Swift=1735
```

A 0% match indicates the T3 encoder is generating completely different tokens.

## Test Parameters

Both implementations use identical parameters:

| Parameter | Value |
|-----------|-------|
| Temperature | 0.0001 (near-deterministic) |
| CFG Weight | 0.5 |
| Repetition Penalty | 2.0 |
| Top P | 1.0 |
| Min P | 0.05 |
| Max Tokens | 1000 |

Test text: `"Wow! I absolutely cannot believe that it worked on the first try!"`

Voice: `samantha` (from `baked_voices/samantha/baked_voice.safetensors`)

## API Usage

The ChatterboxEngine provides methods for isolated testing:

```swift
// Run T3 only (returns speech tokens)
let tokens = try engine.runT3Only(text, temperature: 0.0001)

// Run S3Gen only (returns audio samples)
let audio = try engine.synthesizeFromTokens(tokens)
```

## Common Issues

### Different token counts
- Python: 98 tokens, Swift: 124 tokens
- Indicates T3 is generating different sequences
- Check: text tokenization, CFG logic, sampling

### 0% token match
- Tokens differ from the very first position
- Check: embedding weights, attention, conditioning

### Audio sounds like noise
- If using correct tokens → S3Gen issue
- Check: weight transposition, vocoder, flow matching
