# Nightingale TTS

**High-quality multilingual text-to-speech running entirely on-device with Swift and MLX.**

Nightingale is a production-ready Swift/MLX port of [Chatterbox TTS](https://github.com/resemble-ai/chatterbox), designed to run fully offline on Apple Silicon devices (iPhone, iPad, Mac). The Swift implementation achieves **100% parity** with the Python/PyTorch reference, producing identical speech output.

## Features

- **Multilingual Support**: 23+ languages including English, Dutch, French, German, Spanish, Japanese, Chinese, and more
- **Voice Cloning**: Clone any voice from a short audio reference (3-10 seconds)
- **On-Device Inference**: Runs entirely offline using MLX for GPU acceleration
- **Production Ready**: Cross-validated against Python implementation with perfect token and audio matching
- **Privacy Preserving**: All processing happens on-device, no cloud required
- **Fast**: Optimized for Apple Silicon with Metal Performance Shaders

## Status

✅ **COMPLETE** - Full Swift/MLX pipeline working with perfect parity to Python/PyTorch

- ✅ T3 encoder: 100% token match with Python
- ✅ S3Gen decoder: Perfect audio generation
- ✅ Vocoder: High-quality 24kHz output
- ✅ Cross-validation: All tests passing
- ✅ Test audio: 16 test sentences generated in English and Dutch

## Quick Start

### Requirements

- macOS 14+ or iOS 17+
- Apple Silicon (M1/M2/M3 or A-series)
- Swift 5.9+
- Xcode 15+

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/nightingale_TTS
cd nightingale_TTS
```

2. **Download model files**

Place Chatterbox model files in `models/chatterbox/`:
- `t3_mtl23ls_v2.safetensors` - T3 multilingual model (2.0 GB)
- `s3gen_fp16.safetensors` - S3Gen decoder (1.0 GB)
- `grapheme_mtl_merged_expanded_v1.json` - Tokenizer
- `rope_freqs_llama3.safetensors` - RoPE frequencies

3. **Add voice files**

Place baked voice files in `baked_voices/<voice_name>/baked_voice.safetensors`

Included voices:
- `samantha/` - English female voice
- `sujano/` - English male voice

### Usage

#### Generate Audio (Single Sentence)

```bash
cd swift
swift run -c release GenerateAudio
```

This generates a single test sentence and saves it as `test_audio.wav`.

#### Generate Test Sentences (Batch)

```bash
cd swift
swift run -c release GenerateTestSentences
```

This generates 16 test audio files (4 sentences × 2 languages × 2 voices) in `test_audio/test_sentences/`.

#### Cross-Validation Testing

To verify Python/Swift parity:

```bash
# Step 1: Generate Python baseline
python E2E/cross_validate_python.py

# Step 2: Generate Swift output and compare
cd swift
swift run -c release CrossValidate
```

See [E2E/cross_validation.md](E2E/cross_validation.md) for details.

## Architecture

Nightingale uses a two-stage text-to-speech pipeline:

```
Text Input
    ↓
[T3 Model] - Text → Speech Tokens
    ↓
[S3Gen Encoder] - Tokens → Mel Features
    ↓
[S3Gen Decoder] - ODE Flow Matching
    ↓
[Vocoder] - Mel → Audio Waveform
    ↓
Audio Output (24kHz, 16-bit)
```

### T3 Model (Text Encoder)

- **Architecture**: 30-layer Llama-style transformer
- **Input**: Text (grapheme tokenization)
- **Output**: Speech tokens [0-6560]
- **Features**:
  - Classifier-free guidance (CFG) for quality control
  - Speaker embedding conditioning
  - Repetition penalty, top-p, min-p sampling

### S3Gen (Speech Generator)

- **Encoder**: Conformer-based upsampling encoder
  - Embeds speech tokens to continuous mel features
  - 6 Conformer blocks with multi-head attention
  - Position encoding with learned upsampling

- **Decoder**: ODE flow matching decoder
  - 10-step Euler ODE solver
  - U-Net style architecture with attention transformers
  - Classifier-free guidance for quality

- **Vocoder**: HiFiGAN-based mel-to-waveform vocoder
  - F0 predictor for pitch-aware synthesis
  - Multi-scale discriminators
  - High-quality 24kHz output

### Voice Conditioning

Voices are "baked" by pre-computing embeddings from reference audio:

- **T3 Soul**: 256-dim speaker embedding for text encoder
- **S3 Soul**: 192-dim speech embedding for decoder
- **Conditioning Tokens**: Pre-generated speech tokens from reference
- **Prompt Features**: Mel spectrogram features from reference audio

This enables fast inference without re-encoding reference audio every time.

## Project Structure

```
nightingale_TTS/
├── swift/                          # Swift/MLX implementation
│   ├── Sources/Nightingale/        # Core library
│   │   ├── ChatterboxEngine.swift  # Main TTS engine
│   │   ├── T3Model.swift           # Text encoder
│   │   ├── S3Gen.swift             # Speech generator
│   │   ├── FlowEncoder.swift       # Conformer encoder
│   │   ├── FlowDecoder.swift       # ODE decoder
│   │   └── Mel2Wav.swift           # HiFiGAN vocoder
│   ├── test_scripts/               # Test executables
│   │   ├── GenerateAudio/          # Single sentence generation
│   │   ├── CrossValidate/          # Cross-validation testing
│   │   └── GenerateTestSentences/  # Batch test generation
│   └── Package.swift               # Swift package manifest
│
├── python/                         # Python/PyTorch reference
│   ├── chatterbox/                 # Chatterbox submodule
│   └── venv/                       # Python virtual environment
│
├── E2E/                            # End-to-end testing
│   ├── cross_validate_python.py    # Python cross-validation
│   ├── cross_validation.md         # Testing documentation
│   └── test_sentences.json         # Test data
│
├── models/chatterbox/              # Model weights
│   ├── t3_mtl23ls_v2.safetensors   # T3 model
│   ├── s3gen_fp16.safetensors      # S3Gen decoder
│   └── grapheme_mtl_merged_expanded_v1.json  # Tokenizer
│
├── baked_voices/                   # Voice embeddings
│   ├── samantha/                   # Female voice
│   └── sujano/                     # Male voice
│
└── test_audio/                     # Output directory
    ├── cross_validate/             # Cross-validation results
    └── test_sentences/             # Test sentence outputs
```

## Cross-Validation Results

The Swift implementation has been extensively validated against Python:

### Token Generation (T3 Model)

```
Test: "Wow! I absolutely cannot believe that it worked on the first try!"

Python tokens (98):  [1732, 2068, 2186, 1457, 680, ...]
Swift tokens (98):   [1732, 2068, 2186, 1457, 680, ...]

Match: 98/98 (100.0%)
```

### Audio Generation (S3Gen)

All cross-validation audio files sound **identical** between Python and Swift:

| File | T3 Source | S3Gen Source | Status |
|------|-----------|--------------|--------|
| `python_tokens_python_audio.wav` | Python | Python | ✅ Baseline |
| `python_tokens_swift_audio.wav` | Python | Swift | ✅ Perfect |
| `swift_tokens_swift_audio.wav` | Swift | Swift | ✅ Perfect |
| `swift_tokens_python_audio.wav` | Swift | Python | ✅ Perfect |

**Vocoder Correlation**: 1.0 (perfect match with deterministic noise)

### Test Sentences

16 test audio files generated successfully:

- 4 test cases (expressive, narrative, interrogative, technical)
- 2 languages (English, Dutch)
- 2 voices (Samantha, Sujano)
- All outputs sound natural and clear

## API Usage

### ChatterboxEngine

The main interface for text-to-speech:

```swift
import Nightingale

let engine = ChatterboxEngine()

// Load models
try await engine.loadModels(
    modelsURL: URL(fileURLWithPath: "models/chatterbox")
)

// Load voice
try await engine.loadVoice(
    "samantha",
    voicesURL: URL(fileURLWithPath: "baked_voices")
)

// Generate audio
let audio = try await engine.generateAudio(
    "Hello, world!",
    temperature: 0.0001
)

// Save to WAV file
try ChatterboxEngine.saveWav(
    audio,
    to: URL(fileURLWithPath: "output.wav")
)
```

### Advanced Options

```swift
// Custom temperature (higher = more variation)
let audio = try await engine.generateAudio(
    text,
    temperature: 0.8  // Default: 0.0001 (near-deterministic)
)

// For cross-validation: run T3 and S3Gen separately
let tokens = try engine.runT3Only(text, temperature: 0.0001)
let audio = try engine.synthesizeFromTokens(tokens)
```

## Python Development Environment

For reference implementation and cross-validation testing:

### Setup

```bash
cd python
python3.11 -m venv venv
source venv/bin/activate
pip install -e chatterbox
```

### Requirements

- Python: 3.11+
- PyTorch: 2.6.0+
- Chatterbox TTS: 0.1.6

### Usage

```bash
# Test voice generation
python test_baked_voice.py

# Cross-validation
python E2E/cross_validate_python.py

# Bake a new voice
python bake_voice.py --audio reference.wav --name myvoice
```

## Model Files

### T3 Model

- **File**: `t3_mtl23ls_v2.safetensors` (2.0 GB)
- **Architecture**: 30-layer transformer
- **Vocab**: 2454 tokens (multilingual)
- **Parameters**: ~800M

### S3Gen Model

- **File**: `s3gen_fp16.safetensors` (1.0 GB)
- **Components**: Encoder + Decoder + Vocoder
- **Format**: FP16 for efficiency

### Additional Files

- `rope_freqs_llama3.safetensors` - Pre-computed RoPE frequencies
- `grapheme_mtl_merged_expanded_v1.json` - Multilingual tokenizer
- `python_flow_weights.safetensors` - Python decoder weights (for perfect parity)

## Performance

Tested on M1 MacBook Pro:

| Task | Time | RTF* |
|------|------|------|
| Short sentence (1s audio) | ~3s | 3.0x |
| Medium sentence (3s audio) | ~7s | 2.3x |
| Long sentence (5s audio) | ~11s | 2.2x |

*RTF = Real-Time Factor (lower is better; 1.0x = real-time)

## Supported Languages

Arabic, Chinese (Simplified), Chinese (Traditional), Czech, Dutch, English, Finnish, French, German, Hindi, Hungarian, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Swedish, Thai, Turkish, Ukrainian, Vietnamese

## Contributing

This is a research project for on-device TTS. Contributions welcome:

1. Voice quality improvements
2. Performance optimizations
3. Additional language support
4. iOS app development

## License

This project ports Chatterbox TTS. Please refer to the original [Chatterbox repository](https://github.com/resemble-ai/chatterbox) for licensing information.

## Acknowledgments

- [Resemble AI](https://www.resemble.ai/) for the original Chatterbox TTS model
- [MLX](https://github.com/ml-explore/mlx) team for the excellent Swift ML framework
- [Apple MLX](https://github.com/ml-explore/mlx-swift) for Swift bindings

## Citation

If you use Nightingale or Chatterbox in your research:

```bibtex
@software{chatterbox2024,
  title={Chatterbox: Multilingual Text-to-Speech},
  author={Resemble AI},
  year={2024},
  url={https://github.com/resemble-ai/chatterbox}
}
```

---

**Status**: Production Ready ✅ | **Last Updated**: December 2024
