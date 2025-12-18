# Nightingale Swift

Swift/MLX implementation of Chatterbox TTS for iOS and macOS.

## Status

🚧 **Work in Progress** - Swift port in development

## Directory Structure

```
swift/
├── Package.swift              # Swift Package Manager configuration
├── Sources/
│   └── Nightingale/           # Core TTS engine
│       ├── ChatterboxEngine.swift      # Main engine
│       ├── T3Model.swift               # T3 (Brain) - Text to speech tokens
│       ├── S3Gen.swift                 # S3Gen (Mouth) - Tokens to audio
│       ├── AlignmentStreamAnalyzer.swift  # Control system
│       ├── NPYLoader.swift             # Load prebaked voice .npy files
│       ├── FlowEncoder.swift           # Flow matching
│       ├── MelSpectrogram.swift        # Mel spectrogram generation
│       ├── LinearFactory.swift         # Linear layer factory
│       ├── Perceiver.swift             # Perceiver architecture
│       ├── RelPos.swift                # Relative positional encoding
│       └── T3Config.swift              # T3 configuration
└── test_scripts/              # Test scripts (TBD)
```

## Core Components

### ChatterboxEngine
Main TTS engine that coordinates T3 and S3Gen.

### T3Model (The Brain)
- Converts text → abstract speech tokens
- 520M parameter Llama-style transformer
- Uses AlignmentStreamAnalyzer for script adherence

### S3Gen (The Mouth)
- Converts speech tokens → audio waveform
- Flow matching + HiFiGAN vocoder
- Generates 24kHz audio

### NPYLoader
Loads prebaked voice files (.npy format) created by Python.

## Dependencies

- [mlx-swift](https://github.com/ml-explore/mlx-swift) - Apple MLX for neural networks
- [swift-transformers](https://github.com/huggingface/swift-transformers) - Transformer utilities

## Setup

### Install Dependencies

```bash
cd swift
swift package resolve
```

### Build

```bash
swift build
```

### Generate Metal Library

MLX requires a compiled Metal library for GPU operations. Generate it once after building:

```bash
bash generate_metallib.sh .build/debug
```

This compiles all Metal shaders into [.build/debug/default.metallib](.build/debug/default.metallib).

### Run Tests

Tests must be run from the build directory where [default.metallib](.build/debug/default.metallib) is located:

```bash
# Build and generate metallib
swift build --product TestLoadVoice
bash generate_metallib.sh .build/debug

# Run from build directory
cd .build/debug
./TestLoadVoice
```

## Usage (Coming Soon)

```swift
import Nightingale

// Load engine with prebaked voice
let engine = ChatterboxEngine()
try engine.loadVoice("samantha_full", voicesURL: voicesURL)

// Generate speech
let audio = try engine.generate(text: "Hello world", temperature: 0.4)
```

## Python Integration

This Swift implementation uses prebaked voices created by the Python flow:

1. **Bake voice in Python:**
   ```bash
   cd ../python
   python bake_voice_full.py ref_audio.wav samantha_full
   ```

2. **Use in Swift:**
   - Voice files: `../baked_voices/samantha_full/`
   - Swift loads the .npy files directly
   - No Python runtime needed on device!

## Architecture

### Dual Soul Strategy

Voice is represented by two separate embeddings:

1. **soul_t3_256.npy** (256-dim) - The Actor
   - Controls prosody, pacing, emotion
   - Used by T3 model

2. **soul_s3_192.npy** (192-dim) - The Instrument
   - Controls timbre, pitch, voice quality
   - Used by S3Gen model

This allows both models to receive their native embedding formats without compression or quality loss.

## Development Status

- [x] Core source files ported from old project
- [x] Package.swift configured
- [x] Clean directory structure
- [x] Test prebaked voice loading ✅ **COMPLETE**
- [x] Test T3 generation ✅ **COMPLETE** (14.8 tokens/sec)
- [x] Test S3Gen vocoding ✅ **COMPLETE** (0.52x realtime, 6s audio generated)
- [ ] Verify output matches Python
- [ ] Performance optimization
- [ ] iOS app integration

## Next Steps

1. ~~Create test script to load prebaked voice~~ ✅ Done
2. ~~Verify .npy loading works correctly~~ ✅ Done
3. Test T3 text→tokens generation
4. Test S3Gen tokens→audio generation
5. Compare Swift output with Python reference

## Notes

- All sources copied from working implementation
- Cleaned up structure (no old test clutter)
- Ready for focused development
- Python flow must be approved before Swift work
