# Chatterbox Short Text Hallucination Bug - INVESTIGATION

## Issue Summary

When generating very short texts like "Hi.", the model produces a "now" or "ah" sound before the actual word, resulting in audio that sounds like "now hi" instead of just "hi".

**Reproduction:**
```python
wav = model.generate(text="Hi.", audio_prompt_path="baked_voices/samantha/ref_audio.wav")
# Original: ~1960ms of audio that sounds like "now hi"
```

---

## Root Cause Analysis: TWO-LAYER HALLUCINATION

We discovered there are **two distinct sources** of hallucination:

### Layer 1: Reference Token Leakage (FIXED)
The model was directly copying discrete speech tokens from the reference audio.

**Before fix:** 43.1% of generated tokens matched conditioning tokens
**After fix:** 0% match - eliminated by `ReferenceSuppressionLogitsProcessor`

### Layer 2: Style Conditioning (ONGOING)
Even with reference suppression, the **Perceiver's style latent** still causes hallucination.

**The Conflict:**
- Text says "Hi" (starts with /h/, unvoiced, low energy)
- Style demands "High Energy Start" (from reference audio prosody)
- Model generates "Now" (/n/ = voiced, nasal) to satisfy Style, then "Hi" for Text

**Token Probability Analysis (Step 0):**
```
CFG=0.5 (original):  Token 1641 prob=0.0338 (3.4%)
CFG=2.0 (annealing): Token 1641 prob=0.2803 (28%)
CFG=5.0 (extreme):   Token 1641 prob=0.5987 (60%)
```

Key insight: **Higher CFG just amplifies the wrong choice**, because the conditional (text-guided) output is already predicting "Now" sounds due to style conditioning. CFG doesn't fix the style-text conflict; it amplifies it.

---

## Solution Attempts

### 1. Reference Suppression (WORKS for Layer 1)
Penalizes conditioning tokens in logits.
- Eliminated 43.1% → 0% conditioning token copying
- Does NOT fix style-induced hallucination

### 2. Temperature Annealing (MAKES IT WORSE)
Low temperature at start to force determinism.
- **Problem:** Makes model MORE confident about wrong choice
- Step 0: Token 1641 goes from 3.4% → 99.99% with temp=0.1
- Low temp amplifies the peak, but the peak is wrong

### 3. CFG Annealing (PARTIALLY HELPS)
High CFG at start for text adherence.
- Reduces duration: 960ms → 480ms
- But still generates "Now" tokens at start
- The conditional output is already wrong (style overrides text)

---

## Current Implementation

### Files Changed

#### 1. `inference/logits_processors.py`
```python
class ReferenceSuppressionLogitsProcessor:
    """Penalizes reference tokens (fixes Layer 1)"""
    penalty=5.0, min_text_tokens=16

class StepwiseTemperatureLogitsProcessor:
    """Dynamic temperature (disabled - made things worse)"""
    start_temp=0.8, end_temp=0.8  # Effectively disabled
```

#### 2. `t3.py`
- Reference suppression: penalty=5.0
- CFG annealing: start=5.0, end=0.5, warmup=5 steps
- AlignmentStreamAnalyzer enabled for ALL models

#### 3. `alignment_stream_analyzer.py`
- Fixed crash on short texts (guard S <= 5)
- Fixed 3x token repetition check

---

## Results with Current Fix

| Metric | Original | After Fixes |
|--------|----------|-------------|
| Duration | 1960ms | 480ms |
| Conditioning Match | 43.1% | 0% |
| Still has "Now" | Yes | Likely (needs verification) |

---

## Why the "Now" Persists (Technical Deep Dive)

The Perceiver Resampler compresses reference audio into 32 style tokens that encode:
- Prosodic patterns (energy, rhythm)
- Voice characteristics
- Speaking style

For the reference audio "Samantha", the style likely demands:
- Confident onset (high energy start)
- Voiced beginning (opposite of unvoiced /h/)

The model's conditional output (text + style) resolves this conflict by:
1. Generating "Now" tokens (voiced, satisfies style demand)
2. Then generating "Hi" tokens (satisfies text content)

**CFG formula:** `logits = cond + cfg * (cond - uncond)`

Since `cond` already predicts "Now" (due to style), high CFG just makes it more confident about "Now".

---

## Potential Next Steps

### Option A: Style Attenuation
Reduce influence of Perceiver style for first N frames.
```python
# Possible approach: Scale down style contribution early
if step < warmup:
    style_weight = step / warmup
    cond_logits = text_cond + style_weight * (style_cond - text_cond)
```

### Option B: Force Silence Start
Inject silence/breath tokens at start to prevent voiced onset.
```python
# Force first 2-3 tokens to be silence
if step < 3:
    logits[silence_tokens] += 10.0
```

### Option C: Negative CFG on Style
Add CFG that penalizes style-driven predictions.
```python
# Subtract style contribution
logits = text_cond + text_cfg * (text_cond - uncond) - style_cfg * style_cond
```

### Option D: Text-Speech Alignment Forcing
Use alignment matrix to detect when model deviates from text and force correction.

---

## Test Files

- `python/debug_token_probs.py` - Step-by-step token probability analysis
- `python/diagnose_hi_tokens.py` - Token generation analysis
- `python/test_hi_only.py` - Simple "Hi." test
- `python/test_short_text_fix.py` - Multi-phrase test suite

---

## Summary

| Fix | Layer 1 (Reference) | Layer 2 (Style) |
|-----|---------------------|-----------------|
| Reference Suppression | ✅ Fixed (0% match) | ❌ No effect |
| Temperature Annealing | - | ❌ Made worse |
| CFG Annealing | - | ⚠️ Reduces duration, doesn't eliminate |

The fundamental issue is that **style conditioning overrides text at the start**. Solving this requires either:
1. Modifying how style is applied (attenuation)
2. Changing the generation strategy (force specific tokens)
3. Architectural changes (separate style/text heads)
