# Chatterbox Smart Gate Fix

## Problem

Short text inputs (e.g., "Hello world.", "Hi.") produced audio with:
1. **Cold start artifacts** - "throat clearing" sounds before the first word
2. **Trailing silence** - excessive silence after speech ends
3. **Hallucinations** - repeated words or gibberish for very short texts

## Root Cause

The `AlignmentStreamAnalyzer` monitors attention patterns during T3 generation, but:
1. **Noise detection was too permissive** - only filtered 2 frames when 10+ were needed
2. **Early frames have erratic attention** - focus jumps between tokens 0, 4, 8 before stabilizing
3. **Stop conditions checked before speech started** - caused premature EOS on low-energy noise frames

## Solution: Smart Gate

### 1. Stable Speech Detection

Instead of starting speech when `energy > 0.15`, require **3 consecutive stable frames**:

```python
is_stable_focus = (focus_token >= 1 and
                  max_energy > 0.25 and
                  abs(focus_token - self.last_focus) <= 1)

if is_stable_focus:
    self.stable_count += 1
else:
    self.stable_count = 0

if self.stable_count >= 3:
    self.speech_started = True
```

### 2. Noise Frame Filtering

Frames before speech stabilizes are marked as noise and excluded from output:

```python
# In t3.py generation loop
if not is_noise:
    predicted.append(next_token)  # Only non-noise frames in output
```

### 3. Rule 0: No Early Stopping

All stop conditions now require speech to have started first:

```python
if not self.speech_started:
    self.curr_frame_pos += 1
    return (False, is_noise)  # Never stop, just mark as noise
```

## Files Changed

| File | Change |
|------|--------|
| `alignment_stream_analyzer.py` | Added `stable_count`, `last_focus`, `analyze()` method |
| `t3.py` | Filter noise frames: `if not is_noise: predicted.append()` |
| `generate_short_phrases.py` | Removed `. ` prefix workaround (made things worse) |

## Results

| Metric | Before | After |
|--------|--------|-------|
| "Hello world." duration | 1.64s | 1.32s |
| Noise frames filtered | 2 | 10 |
| Cold start artifacts | Present | Filtered |

## Debug Script

Use `python/debug_hello_world.py` to troubleshoot specific phrases:

```bash
cd /Users/a10n/Projects/nightingale_TTS/python
./venv/bin/python debug_hello_world.py
```

Output shows frame-by-frame attention analysis:
```
Frame   0 | gen=  0 | focus_tok=0/9 | energy=0.009 | stable=0 | speech=False
  -> NOISE (will be filtered)
Frame   1 | gen=  1 | focus_tok=4/9 | energy=0.081 | stable=0 | speech=False
  -> NOISE (will be filtered)
...
Frame  11 | gen= 11 | focus_tok=6/9 | energy=0.406 | stable=3 | speech=True
```

## Key Thresholds

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `stable_count >= 3` | 3 frames | Consecutive stable frames to start speech |
| `max_energy > 0.25` | 25% | Minimum attention energy for stable focus |
| `focus_token >= 1` | Token 1+ | Ignore focus on start token (token 0) |
| `gen_frames >= 10` | 10 frames | Force speech start if no stability found |
| `speech_frames < 12` | 12 frames | Minimum floor before allowing stop |
