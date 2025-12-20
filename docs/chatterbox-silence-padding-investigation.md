# Chatterbox Silence Padding Issue - Investigation Report

## Problem Summary

The Chatterbox TTS model generates excessive trailing silence after short text inputs. This affects audio quality and increases file sizes unnecessarily.

**Observed symptoms:**
- Audio files for short phrases ("Hello world.", "Hi.", "I see.") contain 2-10+ seconds of silence after the spoken words
- The model generates 700+ speech tokens for text that should only need ~30-50 tokens
- Issue is most pronounced with short utterances (< 15 text tokens)

## Technical Background

### Chatterbox Architecture

```
Text → T3 (Text-to-Token) → Speech Tokens → S3Gen → Mel Spectrogram → Vocoder → Audio
```

The T3 model generates speech tokens autoregressively until it emits an EOS (End-of-Speech) token. The problem is that for short texts, the model often fails to emit EOS at the appropriate time.

### The AlignmentStreamAnalyzer

Chatterbox includes an `AlignmentStreamAnalyzer` (in `models/t3/inference/alignment_stream_analyzer.py`) that:
1. Monitors attention patterns during generation
2. Detects when the model has "completed" speaking the text
3. Can force EOS when it detects hallucination patterns

**Critical limitation:** This analyzer is **only enabled for multilingual models** (`if self.hp.is_multilingual`), so English-only models have no such safeguard.

## Root Cause Analysis

### 1. EOS Token Not Reliably Emitted

The T3 model doesn't reliably emit the EOS token for short texts. The model continues generating "silence tokens" or low-energy speech tokens after completing the actual speech.

### 2. Completion Detection Issues

The `AlignmentStreamAnalyzer` uses attention patterns to detect completion:
```python
self.complete = self.complete or self.text_position >= S - 3
```

For very short texts (S < 5 tokens), this detection can be unreliable.

### 3. Multilingual-Only Activation

The alignment analyzer is only created for multilingual models:
```python
# In t3.py inference()
alignment_stream_analyzer = None
if self.hp.is_multilingual:
    alignment_stream_analyzer = AlignmentStreamAnalyzer(...)
```

English-only models bypass all hallucination detection.

## Attempted Fixes

### Attempt 1: Enable Analyzer for All Models
**Result:** Caused crashes due to empty tensor slices for very short texts (S <= 5).

### Attempt 2: Add Guards for Empty Slices
**Result:** Fixed crashes, but checks were too aggressive and cut off speech prematurely.

### Attempt 3: Relax Thresholds
Changes made:
- `token_repetition`: Require 4+ repeated tokens (was 2)
- `post_completion_excess`: Allow 20 frames after completion (was 10)
- Added `absolute_cap` and `universal_cap` based on text length

**Result:** Either too aggressive (cuts off speech) or too permissive (silence remains).

### Core Issue
The attention-based completion detection doesn't work reliably across all text lengths and model types. The heuristics that work for one case break another.

## Recommended Solution: Audio-Level Silence Trimming

Instead of trying to predict during generation, **trim silence from the generated audio**:

```python
import librosa
import numpy as np

def trim_trailing_silence(audio, sr, threshold_db=-40.0, min_silence_ms=100.0):
    """
    Trim trailing silence from audio.

    Args:
        audio: Audio waveform as numpy array
        sr: Sample rate
        threshold_db: Silence threshold in dB
        min_silence_ms: Minimum silence to preserve at end

    Returns:
        Trimmed audio
    """
    # Use librosa's built-in trim (fast, C-optimized)
    trimmed, _ = librosa.effects.trim(audio, top_db=abs(threshold_db))

    # Add back a small tail for natural ending
    min_samples = int(sr * min_silence_ms / 1000)
    end_idx = len(trimmed) + min_samples
    if end_idx < len(audio):
        return audio[:end_idx]
    return trimmed
```

### Why This Approach?

| Aspect | Generation-Time Detection | Audio-Level Trimming |
|--------|---------------------------|---------------------|
| Reliability | Low - heuristics fail edge cases | High - based on actual signal |
| Language agnostic | No - different models need tuning | Yes - works on any audio |
| Complexity | High - attention patterns, thresholds | Low - simple energy calculation |
| Risk of cutting speech | Yes - aggressive thresholds | No - looks at actual energy |
| Processing cost | Per-token during generation | Once after generation (~10ms) |

### Implementation Options

**Option A: In the library (tts.py)**
```python
# In ChatterboxTTS.generate()
wav = wav.squeeze(0).detach().cpu().numpy()
wav = trim_trailing_silence(wav, self.sr)  # Add this line
watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
```

**Option B: In the calling script (recommended for testing)**
```python
# In generate_short_phrases.py
wav = model.generate(text=text, ...)
wav_np = wav.squeeze().numpy()
wav_trimmed = librosa.effects.trim(wav_np, top_db=40)[0]
wav = torch.from_numpy(wav_trimmed).unsqueeze(0)
```

## Current State of the Code

### Modified Files (not yet reverted)

1. **alignment_stream_analyzer.py**
   - Added `_hook_handles` list and `cleanup()` method
   - Added guards for empty tensor slices
   - Added multiple detection heuristics (most are too aggressive or too permissive)

2. **t3.py**
   - Added `cleanup()` call after generation
   - Currently set to multilingual-only (reverted from all-models)

3. **tts.py**
   - Added `numpy` import
   - Added `trim_trailing_silence()` function (not yet integrated)

### Recommended Next Steps

1. **Revert alignment_stream_analyzer.py** to vanilla state (keep only hook cleanup)
2. **Implement audio-level trimming** in the calling script first for testing
3. **Test across multiple languages and text lengths** to find optimal threshold
4. **Optionally add to library** as an opt-in parameter: `generate(..., trim_silence=True)`

## Test Cases

Use these to validate any fix:

| Text | Voice | Expected Duration | Issue |
|------|-------|-------------------|-------|
| "Hi." | samantha | ~0.5s | Generates 5+ seconds |
| "Hello world." | samantha | ~1.0s | Generates 10+ seconds |
| "I see." | samantha | ~0.8s | Long trailing silence |
| "12:30 PM." | sujano | ~1.5s | Numbers cause issues |
| "Please." | samantha | ~0.6s | Single word silence |

## Files Reference

- `python/chatterbox/src/chatterbox/tts.py` - Main TTS class
- `python/chatterbox/src/chatterbox/models/t3/t3.py` - T3 inference
- `python/chatterbox/src/chatterbox/models/t3/inference/alignment_stream_analyzer.py` - Hallucination detection
- `python/generate_short_phrases.py` - Test script
- `E2E/short_phrases.json` - Test cases (25 phrases x 2 voices)
