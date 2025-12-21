# AlignmentStreamAnalyzer Port Specification

**Date:** 2025-12-11
**Priority:** CRITICAL - Required for production text accuracy
**Estimated Effort:** 3-5 days engineering

---

## Executive Summary

The PyTorch Chatterbox T3 model uses an `AlignmentStreamAnalyzer` that monitors attention patterns during generation and enforces text-speech alignment. **MLX completely lacks this component**, which is why generated speech doesn't match input text.

This is not a bug fix - it's porting a **critical control system** that the model requires to function correctly.

---

## The Problem

Without the analyzer, T3 is an **unguided autoregressive generator**:

| Behavior | With Analyzer | Without Analyzer (Current MLX) |
|----------|---------------|-------------------------------|
| Text tracking | Monotonic left-to-right | Random jumping |
| Word accuracy | ~99% | ~70-80% |
| Repetition | Detected & stopped | Unconstrained |
| Hallucination | Detected & stopped | Common |
| Early termination | Prevented | Frequent |

---

## Source Code Location

**PyTorch Implementation:**
```
chatterbox/models/t3/inference/alignment_stream_analyzer.py
```

**Key Constants:**
```python
LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]
# Format: (layer_index, head_index)
# These specific heads track text-speech alignment
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    GENERATION LOOP                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌───────────────────────┐                │
│  │  T3 Forward  │────▶│  Attention Matrices   │                │
│  │    Pass      │     │  (Layers 9, 12, 13)   │                │
│  └──────────────┘     └───────────┬───────────┘                │
│                                   │                              │
│                                   ▼                              │
│                       ┌───────────────────────┐                 │
│                       │ AlignmentStreamAnalyzer│                │
│                       │                       │                 │
│                       │  • Extract attention  │                 │
│                       │  • Track text position│                 │
│                       │  • Detect anomalies   │                 │
│                       │  • Modify logits      │                 │
│                       └───────────┬───────────┘                 │
│                                   │                              │
│                                   ▼                              │
│                       ┌───────────────────────┐                 │
│                       │   Modified Logits     │                 │
│                       │  (Safe for sampling)  │                 │
│                       └───────────────────────┘                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Algorithm

### 1. Attention Extraction

Hook into specific attention heads during forward pass:

```python
# PyTorch hooks into these specific (layer, head) pairs
LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]

# Extract attention: shape (1, seq_len) for KV-cached generation
# First pass: (conditioning + text + bos, text_len)
# Subsequent: (1, text_len)
```

### 2. Alignment Matrix Construction

```python
def step(self, logits, next_token=None):
    # Average attention across the 3 alignment heads
    aligned_attn = torch.stack(self.last_aligned_attns).mean(dim=0)

    # Extract text portion of attention
    i, j = self.text_tokens_slice  # Start/end of text tokens in sequence

    if self.curr_frame_pos == 0:
        # First chunk: full attention matrix
        A_chunk = aligned_attn[j:, i:j]  # (speech_frames, text_len)
    else:
        # Subsequent: single frame due to KV cache
        A_chunk = aligned_attn[:, i:j]   # (1, text_len)

    # Monotonic masking: prevent looking at future text
    A_chunk[:, self.curr_frame_pos + 1:] = 0

    # Accumulate alignment matrix
    self.alignment = torch.cat((self.alignment, A_chunk), dim=0)
```

### 3. Position Tracking

```python
# Find current text position from attention peak
cur_text_posn = A_chunk[-1].argmax()

# Detect discontinuity (jumping too far in text)
discontinuity = not(-4 < cur_text_posn - self.text_position < 7)

if not discontinuity:
    self.text_position = cur_text_posn
```

### 4. Anomaly Detection

```python
# FALSE START: Hallucinations at beginning
# Check if attention is focused on end of sequence (wrong) vs beginning (correct)
false_start = (not self.started) and (
    A[-2:, -2:].max() > 0.1 or  # Looking at end
    A[:, :4].max() < 0.5        # Not looking at beginning
)

# COMPLETION CHECK: Has model reached end of text?
self.complete = self.complete or self.text_position >= S - 3

# LONG TAIL: Hallucinations at end
# If attention on final tokens lasts too long
long_tail = self.complete and (A[self.completed_at:, -3:].sum(dim=0).max() >= 5)

# ALIGNMENT REPETITION: Attention jumping back to earlier text
alignment_repetition = self.complete and (
    A[self.completed_at:, :-5].max(dim=1).values.sum() > 5
)

# TOKEN REPETITION: Same token generated 3+ times
token_repetition = (
    len(self.generated_tokens) >= 3 and
    len(set(self.generated_tokens[-2:])) == 1
)
```

### 5. Logit Modification

```python
# SUPPRESS EOS: Prevent early termination
if cur_text_posn < S - 3 and S > 5:
    logits[..., self.eos_idx] = -2**15  # -32768

# FORCE EOS: On detected errors
if long_tail or alignment_repetition or token_repetition:
    logits = -(2**15) * torch.ones_like(logits)
    logits[..., self.eos_idx] = 2**15   # +32768
```

---

## MLX Implementation Requirements

### Challenge: Optimized Attention

Standard MLX uses `mx.fast.scaled_dot_product_attention` which doesn't expose attention weights. You must:

1. **Subclass or modify** the attention layers for inspection
2. **Only de-optimize layers 9, 12, 13** (keep others fast)
3. **Capture specific heads** (15, 11, 2 respectively)

### Required Changes to T3

```python
# In mlx_audio/tts/models/chatterbox/t3/t3.py

class InspectableAttention(Attention):
    """Attention that exposes weights for alignment tracking."""

    def __init__(self, args, layer_idx, capture_heads=None):
        super().__init__(args)
        self.layer_idx = layer_idx
        self.capture_heads = capture_heads or []
        self.last_attention_weights = None

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape for multi-head attention
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Apply RoPE
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # MANUAL attention computation (not optimized kernel)
        scale = queries.shape[-1] ** -0.5
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)

        if mask is not None:
            scores = scores + mask

        weights = mx.softmax(scores, axis=-1)

        # CAPTURE specific heads for alignment analysis
        if self.capture_heads:
            self.last_attention_weights = {
                head: weights[:, head, :, :]
                for head in self.capture_heads
            }

        output = weights @ values
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output)
```

### Analyzer Class (MLX Port)

```python
# New file: mlx_audio/tts/models/chatterbox/t3/alignment_analyzer.py

import mlx.core as mx
from dataclasses import dataclass

@dataclass
class AlignmentResult:
    false_start: bool
    long_tail: bool
    repetition: bool
    discontinuity: bool
    complete: bool
    position: int

LLAMA_ALIGNED_HEADS = {
    9: 2,    # Layer 9, head 2
    12: 15,  # Layer 12, head 15
    13: 11,  # Layer 13, head 11
}

class AlignmentStreamAnalyzer:
    def __init__(self, text_tokens_slice, eos_idx=6562):
        self.text_tokens_slice = text_tokens_slice
        self.eos_idx = eos_idx
        self.alignment = None
        self.curr_frame_pos = 0
        self.text_position = 0
        self.started = False
        self.complete = False
        self.completed_at = None
        self.generated_tokens = []

    def step(self, attention_weights: dict, logits: mx.array, next_token=None):
        """
        Process one generation step.

        Args:
            attention_weights: Dict[layer_idx, mx.array] from InspectableAttention
            logits: Current logits (1, vocab_size)
            next_token: Last generated token for repetition tracking

        Returns:
            Modified logits
        """
        # Average attention across alignment heads
        attns = [attention_weights[layer][:, head, :, :]
                 for layer, head in LLAMA_ALIGNED_HEADS.items()]
        aligned_attn = mx.mean(mx.stack(attns), axis=0).squeeze(0)

        i, j = self.text_tokens_slice

        if self.curr_frame_pos == 0:
            A_chunk = aligned_attn[j:, i:j]
        else:
            A_chunk = aligned_attn[-1:, i:j]

        # Monotonic masking
        if self.curr_frame_pos + 1 < A_chunk.shape[-1]:
            mask = mx.zeros_like(A_chunk)
            mask = mask.at[:, self.curr_frame_pos + 1:].set(0)
            A_chunk = A_chunk * (1 - mask)

        # Accumulate
        if self.alignment is None:
            self.alignment = A_chunk
        else:
            self.alignment = mx.concatenate([self.alignment, A_chunk], axis=0)

        A = self.alignment
        T, S = A.shape

        # Position tracking
        cur_text_posn = int(mx.argmax(A_chunk[-1]))
        discontinuity = not(-4 < cur_text_posn - self.text_position < 7)
        if not discontinuity:
            self.text_position = cur_text_posn

        # False start detection
        false_start = (not self.started) and (
            float(mx.max(A[-2:, -2:])) > 0.1 or
            float(mx.max(A[:, :4])) < 0.5
        )
        self.started = not false_start

        # Completion check
        self.complete = self.complete or self.text_position >= S - 3
        if self.complete and self.completed_at is None:
            self.completed_at = T

        # Long tail detection
        long_tail = False
        if self.complete and self.completed_at is not None:
            long_tail = float(mx.max(mx.sum(A[self.completed_at:, -3:], axis=0))) >= 5

        # Alignment repetition
        alignment_repetition = False
        if self.complete and self.completed_at is not None:
            alignment_repetition = float(mx.sum(mx.max(A[self.completed_at:, :-5], axis=1))) > 5

        # Token repetition tracking
        if next_token is not None:
            self.generated_tokens.append(int(next_token))
            if len(self.generated_tokens) > 8:
                self.generated_tokens = self.generated_tokens[-8:]

        token_repetition = (
            len(self.generated_tokens) >= 3 and
            len(set(self.generated_tokens[-2:])) == 1
        )

        # LOGIT MODIFICATION

        # Suppress EOS early
        if cur_text_posn < S - 3 and S > 5:
            logits = logits.at[..., self.eos_idx].set(-32768.0)

        # Force EOS on errors
        if long_tail or alignment_repetition or token_repetition:
            logits = mx.full_like(logits, -32768.0)
            logits = logits.at[..., self.eos_idx].set(32768.0)

        self.curr_frame_pos += 1

        return logits
```

### Modified Inference Loop

```python
def inference_with_alignment(self, t3_cond, text_tokens, max_new_tokens=1024, ...):
    """T3 inference with alignment enforcement."""

    # ... setup code ...

    # Initialize analyzer
    text_start = len_cond  # After conditioning tokens
    text_end = text_start + text_tokens.shape[1]
    analyzer = AlignmentStreamAnalyzer(
        text_tokens_slice=(text_start, text_end),
        eos_idx=self.hp.stop_speech_token,
    )

    # Generation loop
    for step in range(max_new_tokens):
        # Forward pass (attention layers capture weights)
        hidden = self.tfmr.model(...)

        # Get captured attention from inspectable layers
        attention_weights = {
            9: self.tfmr.model.layers[9].self_attn.last_attention_weights,
            12: self.tfmr.model.layers[12].self_attn.last_attention_weights,
            13: self.tfmr.model.layers[13].self_attn.last_attention_weights,
        }

        # Get logits
        logits = self.speech_head(hidden[:, -1:, :]).squeeze(1)

        # ANALYZER INTERVENTION
        logits = analyzer.step(attention_weights, logits,
                              next_token=generated_ids[-1] if generated_ids else None)

        # Apply other processors (repetition penalty, etc.)
        # ...

        # Sample
        next_token = sampler(logits)

        if next_token == self.hp.stop_speech_token:
            break

        generated_ids.append(next_token)
        # ... continue loop ...
```

---

## Swift Port Notes

For iOS deployment, the same logic applies:

```swift
// Swift pseudo-code

class AlignmentStreamAnalyzer {
    let textTokensSlice: (Int, Int)
    let eosIdx: Int
    var alignment: MLXArray?
    var currFramePos: Int = 0
    var textPosition: Int = 0
    var started: Bool = false
    var complete: Bool = false
    var generatedTokens: [Int] = []

    func step(attentionWeights: [Int: MLXArray], logits: MLXArray) -> MLXArray {
        // Port the Python logic 1:1
        // MLX operations map directly to MLX-Swift
    }
}

// In generation loop:
while !finished {
    let hidden = model.forward(input)

    // Get attention from layers 9, 12, 13
    let attentions = model.getCapturedAttentions()

    var logits = model.speechHead(hidden)

    // ALIGNMENT ENFORCEMENT
    logits = analyzer.step(attentionWeights: attentions, logits: logits)

    let nextToken = sample(logits)
    // ...
}
```

---

## Testing Checklist

- [ ] Short text (<10 words): Verify exact match
- [ ] Medium text (10-50 words): Verify no word swapping
- [ ] Long text (>100 words): Verify no hallucination/repetition
- [ ] Compare attention patterns: MLX vs PyTorch
- [ ] Verify monotonic text consumption
- [ ] Verify EOS suppression works
- [ ] Verify forced EOS on repetition

---

## Performance Considerations

| Aspect | Impact |
|--------|--------|
| De-optimized attention (3 layers) | ~10-15% slowdown |
| Analyzer computation | Negligible (<1%) |
| Memory (attention matrices) | +~50MB for long sequences |

The slowdown is acceptable given the massive improvement in accuracy.

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `t3/alignment_analyzer.py` | CREATE - Port analyzer |
| `t3/t3.py` | MODIFY - Add InspectableAttention, new inference loop |
| `t3/__init__.py` | MODIFY - Export new classes |

---

## Success Criteria

After implementation:

1. "Hello I'm Seity." generates exactly those words (not "Hello let's I'm Seity")
2. Long text (200+ words) generates with ~99% accuracy
3. No hallucinations at start or end
4. No word repetition loops

---

*Engineering specification - 2025-12-11*
