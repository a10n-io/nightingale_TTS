# Nightingale TTS

## Project Goal

Port [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) from Python/PyTorch to Swift/MLX with the goal of running entirely on iPhone.

## Overview

Chatterbox is a PyTorch-based text-to-speech system. This project aims to:

- Convert the PyTorch model architecture to MLX
- Implement the TTS pipeline in Swift
- Optimize for on-device inference on iOS devices
- Enable fully offline, privacy-preserving speech synthesis

## Status

| Component | Python | Swift |
|-----------|--------|-------|
| T3 Model (text → tokens) | ✅ Working | ✅ Working |
| S3Gen Encoder | ✅ Working | ✅ Working |
| S3Gen Decoder (ODE) | ✅ Working | ❌ Bug |
| Mel2Wav Vocoder | ✅ Working | ⚠️ Blocked |
| Voice Baking | ✅ Working | N/A (uses NPY) |

**Python:** Fully functional end-to-end TTS pipeline.
**Swift:** T3 works, S3Gen decoder has MLX lazy evaluation bug (see [docs/VERIFICATION_STATUS.md](docs/VERIFICATION_STATUS.md)).

## Requirements

- iOS device with Apple Silicon (A-series or M-series chip)
- Xcode for Swift development
- MLX framework for on-device machine learning
- Python 3.11+ for development and testing

## Python Development Environment

### Setup

The Python environment is located at `python/` and uses Python 3.11 with PyTorch 2.6.0 and Chatterbox-Multilingual TTS.

**Environment Details:**
- Python: 3.11.14
- PyTorch: 2.6.0
- Chatterbox TTS: 0.1.6 (installed from source in editable mode)
- Device Support: MPS (Metal Performance Shaders) for Apple Silicon GPU acceleration

**Virtual Environment Path:**
```
python/venv/
```

### Installed Components

- Chatterbox-Multilingual TTS model (23+ languages)
- PyTorch with MPS support
- All dependencies (transformers, diffusers, librosa, etc.)
- Baked voice system for fast inference

### Usage

#### Testing Voice Generation

To test the baked voice with English and Dutch samples:

```bash
/Users/a10n/Projects/nightingale_TTS/python/venv/bin/python /Users/a10n/Projects/nightingale_TTS/python/test_baked_voice.py
```

This will generate timestamped audio files in `test_audio/`:
- `python_test_english_YYYYMMDD_HHMMSS.wav`
- `python_test_dutch_YYYYMMDD_HHMMSS.wav`

#### Baking Voice Embeddings

To create voice embeddings from a reference audio file:

```bash
/Users/a10n/Projects/nightingale_TTS/python/venv/bin/python /Users/a10n/Projects/nightingale_TTS/python/bake_voice.py
```

#### Parameter Tuning

To test different parameter combinations for voice matching:

```bash
/Users/a10n/Projects/nightingale_TTS/python/venv/bin/python /Users/a10n/Projects/nightingale_TTS/python/tune_voice_params.py
```

## Swift Development Environment

### Architecture

```
Text Input
    ↓
T3 Model (text → speech tokens)
    ↓
S3Gen Encoder (tokens → mel features)
    ↓
S3Gen Decoder (ODE flow matching)
    ↓
Mel2Wav Vocoder (mel → audio)
    ↓
Audio Output (24kHz)
```

### Setup

The Swift implementation is located at `swift/` and uses MLX for GPU-accelerated inference.

**Dependencies:**
- MLX Swift: 0.29.1
- Swift Transformers: 0.1.24
- Swift 6.0+

### MLX Model Files

Located in `models/mlx/`:
- `t3_fp32.safetensors` - T3 model weights
- `s3gen_fp16.safetensors` - S3Gen encoder/decoder
- `vocoder_weights.safetensors` - Mel2Wav vocoder
- `rope_freqs_llama3.safetensors` - Pre-computed RoPE frequencies
- `config.json` - Model configuration

### Voice Files (NPY Format)

Swift uses NPY format instead of PyTorch .pt files. Convert using:

```bash
cd python
source venv/bin/activate
python convert_voice_to_npy_padded.py
```

This creates `baked_voices/baked_voice_npy/` with:
- `soul_t3_256.npy` - T3 speaker embedding
- `soul_s3_192.npy` - S3Gen speaker embedding
- `t3_cond_tokens.npy` - T3 conditioning tokens
- `prompt_token.npy` - S3Gen prompt tokens
- `prompt_feat.npy` - S3Gen prompt features

### Build & Run

```bash
cd swift/test_scripts/GenerateAudio
swift build
.build/debug/GenerateAudio
```

### Known Issues

The S3Gen decoder has an MLX lazy evaluation bug that causes a convolution shape mismatch during the ODE loop. See [docs/VERIFICATION_STATUS.md](docs/VERIFICATION_STATUS.md) for details.

## Model Files

All model files are centralized in `/models/chatterbox/` for consistent testing between Python/PyTorch and Swift/MLX implementations:

**Location:** `/Users/a10n/Projects/nightingale_TTS/models/chatterbox/`

**Files (3.0 GB total):**
- `ve.pt` (5.4 MB) - Voice encoder
- `t3_mtl23ls_v2.safetensors` (2.0 GB) - T3 multilingual model
- `s3gen.pt` (1.0 GB) - S3Gen vocoder
- `grapheme_mtl_merged_expanded_v1.json` (68 KB) - Multilingual tokenizer
- `tokenizer.json` (25 KB) - General tokenizer
- `conds.pt` (105 KB) - Default voice conditionals
- `Cangjie5_TC.json` (1.8 MB) - Chinese input method data

**Voice Files:** Custom baked voices are stored separately in `baked_voices/`
- `baked_voice.pt` - Pre-computed voice embeddings
- `ref_audio.wav` - Reference audio

## Architecture

### Pipeline Overview

Chatterbox TTS uses a two-stage architecture:

1. **T3 Model (The Brain)** - Converts text to speech tokens
   - Character-based tokenization
   - 30-layer Llama-style transformer
   - Classifier-free guidance (CFG) for quality
   - Outputs discrete speech tokens [0-6561]

2. **S3Gen Model (The Mouth)** - Converts speech tokens to audio
   - Token embedding + upsampling encoder
   - Flow matching decoder (ODE-based)
   - HiFiGAN-style vocoder for mel → waveform

### Voice Conditioning

Voices are "baked" by pre-computing:
- Speaker embeddings (256-dim for T3, 192-dim for S3Gen)
- Conditioning tokens from reference audio
- Prompt features (mel spectrograms)

This enables fast inference without re-encoding the reference audio.

## Documentation

- [docs/VERIFICATION_STATUS.md](docs/VERIFICATION_STATUS.md) - Current verification status
- [docs/SWIFT_COMPLETE.md](docs/SWIFT_COMPLETE.md) - Swift implementation details
- [docs/E2E_VERIFICATION.md](docs/E2E_VERIFICATION.md) - End-to-end verification guide
