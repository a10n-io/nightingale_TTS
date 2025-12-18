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

Project initialization - development in progress.

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

## Architecture

TBD - will document the model architecture and conversion process as development progresses.
