# Chatterbox Short Text Hallucination Bug

## Summary

The Chatterbox TTS model exhibits hallucination behavior (repetition, gibberish) when generating audio for short text inputs (< 15 tokens). This is caused by overly permissive thresholds in the `AlignmentStreamAnalyzer` and missing hook cleanup.

## Affected Version

- Repository: [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox)
- File: `src/chatterbox/models/t3/inference/alignment_stream_analyzer.py`

## Symptoms

| Voice | Input | Expected | Actual |
|-------|-------|----------|--------|
| samantha | "Hello world" | ~19 tokens, clean audio | 46 tokens, repeats "hello world" twice |
| sujano | "Hello world" | ~23 tokens, clean audio | 252 tokens, says "hello world" then random gibberish |

The issue is more pronounced with:
- Short utterances (< 15 text tokens)
- Multiple consecutive generations (hook accumulation)
- Certain voice profiles

## Root Cause

### 1. Long-tail threshold too permissive for short texts

The `long_tail` detection uses a fixed threshold of 5 frames (~200ms) for all text lengths:

```python
# Original code (line ~140)
long_tail = self.complete and (A[self.completed_at:, -3:].sum(dim=0).max() >= 5)
```

For short texts, 5 frames is too long - the model continues generating hallucinated audio before triggering EOS.

### 2. No excessive generation check

There's no safeguard for when the model generates far more speech tokens than expected for the text length. Normal speech produces ~4-6 speech tokens per text token, but hallucinating models can produce 10x or more.

### 3. Hook accumulation across calls

Forward hooks registered on attention layers are never removed:

```python
# Original code - hooks registered but never cleaned up
target_layer.register_forward_hook(attention_forward_hook)
```

When `AlignmentStreamAnalyzer` is instantiated multiple times (e.g., generating multiple samples), hooks accumulate on the same layers, potentially causing interference.

## Fix

### 1. Add short-text detection with stricter thresholds

```python
# Short text detection - use stricter thresholds for short utterances
is_short_text = S < 15
long_tail_threshold = 3 if is_short_text else 5  # 120ms vs 200ms

# Activations for the final token that last too long are likely hallucinations.
long_tail = self.complete and (A[self.completed_at:, -3:].sum(dim=0).max() >= long_tail_threshold)
```

### 2. Add excessive generation check

```python
# For short texts, also check for excessive generation
max_expected_frames = S * 8  # Allow up to 8 speech tokens per text token
excessive_generation = is_short_text and T > max_expected_frames and self.complete

# Include in EOS forcing condition
if long_tail or alignment_repetition or token_repetition or excessive_generation:
    # Force EOS
```

### 3. Add hook cleanup

```python
class AlignmentStreamAnalyzer:
    def __init__(self, ...):
        self._hook_handles = []  # Store handles
        # ...
        handle = target_layer.register_forward_hook(attention_forward_hook)
        self._hook_handles.append(handle)

    def cleanup(self):
        """Remove registered hooks to prevent accumulation across calls."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
```

And in `t3.py`, call cleanup after generation:

```python
# After generation loop
if self.patched_model.alignment_stream_analyzer is not None:
    self.patched_model.alignment_stream_analyzer.cleanup()
```

## Results After Fix

| Voice | Input | Before | After |
|-------|-------|--------|-------|
| samantha | "Hello world" | 46 tokens (repetition) | 19 tokens (clean) |
| sujano | "Hello world" | 252 tokens (gibberish) | 23 tokens (clean) |

## Files Changed

1. `src/chatterbox/models/t3/inference/alignment_stream_analyzer.py`
   - Added `is_short_text` detection
   - Added `long_tail_threshold` variable (3 for short, 5 for long)
   - Added `excessive_generation` check
   - Added `_hook_handles` list and `cleanup()` method

2. `src/chatterbox/models/t3/t3.py`
   - Added `cleanup()` call after generation loop

## Recommendation

Consider submitting this fix as a PR to `resemble-ai/chatterbox` or opening an issue to bring attention to the bug.
