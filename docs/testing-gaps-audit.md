# Testing Gaps Audit - Swift TTS Port

**Date:** 2025-12-21
**Status:** In Progress
**Priority:** High

## Summary

During the Swift port of Chatterbox TTS, we discovered that our E2E pipeline testing methodology missed critical runtime behaviors. This document captures the gaps and recommended improvements.

---

## Gap 0: E2E Tests Only Verified Deterministic Behavior (ROOT CAUSE)

### The Core Issue

**E2E verification used `TEMPERATURE = 0.001` (essentially greedy decoding).**

From `E2E/verify_e2e.py:44`:
```python
TEMPERATURE = 0.001  # Near-deterministic
```

This means:
- Both Python and Swift pick the **argmax** of logits at each step
- Small logit differences don't affect token selection
- Tests pass because deterministic outputs match exactly

But in real usage with `temperature=0.8`:
- Softmax amplifies small logit differences into probability differences
- Different tokens get sampled due to stochastic selection
- Once ONE token differs → sequences diverge completely (autoregressive)

### Comparison of Test vs Production Settings

| Setting | E2E Test | Real Usage | Impact |
|---------|----------|------------|--------|
| Temperature | **0.001** | **0.8** | Greedy vs stochastic |
| Top-P | **1.0** | **0.95** | No filter vs filtered |
| Rep Penalty | **2.0** | **1.05** | Aggressive vs weak |

### What This Means

- E2E tests verified: **"Do logits produce the same argmax?"** ✅
- E2E tests did NOT verify: **"Are logits close enough for stochastic sampling?"** ❌

The logit differences are small enough for deterministic matching but large enough that at temperature=0.8, probability distributions diverge and different tokens get sampled.

### Why Tests Passed But Production Fails

1. At step 0-20: Logit argmax matches → same tokens generated
2. Small numerical drift accumulates (FP16 vs FP32, RoPE differences, etc.)
3. At step ~21: Drift causes different token to be sampled under stochastic conditions
4. Autoregressive generation diverges completely from that point
5. Swift gets stuck in repetition loop (token 6405), Python continues normally

---

## Gap 1: AlignmentStreamAnalyzer Not Tested

### What Was Missed
The `AlignmentStreamAnalyzer` is a runtime intervention that monitors attention patterns during T3 generation and modifies logits to:
- Suppress EOS when text isn't complete (prevents early termination)
- Force EOS on repetition/hallucination detection (prevents infinite loops)

### Why It Was Missed

| Factor | Description |
|--------|-------------|
| **Conditional code path** | Python only enables it for multilingual models (`if self.hp.is_multilingual`). Our English model testing never triggered it. |
| **Deterministic testing** | E2E tests used `temperature=0` for reproducibility - no sampling variance to expose loops |
| **Short reference outputs** | Pre-captured golden outputs were controlled, not stress-testing 1000-token generation |
| **Unit vs integration** | We verified components work, not runtime interactions |

### Location in Code
- Python: `chatterbox/models/t3/inference/alignment_stream_analyzer.py`
- Swift: `Sources/Nightingale/AlignmentStreamAnalyzer.swift`
- Conditional enable: `chatterbox/models/t3/t3.py:280` (`if self.hp.is_multilingual`)

### Current Status
- [x] Swift AlignmentStreamAnalyzer implemented
- [x] Attention capture from layers 9, 12, 13 working
- [x] Token repetition detection (8x) as fallback enabled
- [ ] EOS suppression disabled (textPos tracking not working correctly)
- [ ] Natural EOS generation not matching Python behavior

---

## Gap 2: Runtime Logit Processing Differences

### Components to Verify

| Component | Python Implementation | Swift Implementation | Status |
|-----------|----------------------|---------------------|--------|
| CFG formula | `cond + cfg * (cond - uncond)` | Same | Verified |
| RepetitionPenalty | HuggingFace `RepetitionPenaltyLogitsProcessor` | Custom implementation | Needs audit |
| Temperature | Division after filtering | Division after filtering | Verified |
| MinP filtering | HuggingFace `MinPLogitsWarper` | Custom implementation | Needs audit |
| TopP filtering | HuggingFace `TopPLogitsWarper` | Custom implementation | Needs audit |
| Sampling | `torch.multinomial` | Custom categorical | Needs audit |

### Order of Operations

**Python:**
1. CFG combine
2. AlignmentStreamAnalyzer.step (if multilingual)
3. RepetitionPenaltyProcessor
4. Temperature scaling
5. MinP filtering
6. TopP filtering
7. Softmax
8. Multinomial sampling
9. EOS check

**Swift:** Same order - verified

---

## Gap 3: Model Divergence Under Stochastic Sampling

### Observation
- Python (FP32): Generates ~72 tokens naturally, hits EOS
- Swift (FP16): Gets stuck at ~21 tokens, repeats token 6405 infinitely

### Potential Causes
- [ ] FP16 vs FP32 logit drift accumulation
- [ ] RoPE implementation differences
- [ ] Attention computation differences (SDPA vs manual)
- [ ] Weight loading/transposition issues
- [ ] KV cache implementation differences

---

## Recommended Testing Improvements

### 1. Stochastic Integration Tests
```python
# Run generation with temperature > 0 multiple times
# Verify output distribution, not just single outputs
for _ in range(10):
    audio = model.generate(text, temperature=0.8)
    assert len(audio) > min_length
    assert no_infinite_loops(audio)
```

### 2. Long-Form Generation Tests
- Generate 500+ tokens to catch infinite loops
- Verify natural EOS occurs within expected range

### 3. Conditional Feature Matrix
Test all combinations:
- `is_multilingual`: True / False
- `cfg_weight`: 0.0 / 0.5 / 1.0
- `temperature`: 0.0 / 0.5 / 0.8 / 1.0
- `repetition_penalty`: 1.0 / 1.2 / 1.5

### 4. Step-by-Step Logit Comparison
```python
# Compare raw logits at each generation step
for step in range(100):
    python_logits = python_model.get_logits(step)
    swift_logits = swift_model.get_logits(step)
    assert max_diff(python_logits, swift_logits) < threshold
```

### 5. Attention Pattern Verification
- Capture attention from aligned heads (9,2), (12,15), (13,11)
- Verify attention progresses through text monotonically
- Compare Python vs Swift attention distributions

---

## Action Items

- [ ] Create comprehensive stochastic test suite
- [ ] Add long-form generation regression tests
- [ ] Implement step-by-step logit comparison tool
- [ ] Audit MinP/TopP implementations against HuggingFace
- [ ] Investigate FP16 vs FP32 divergence root cause
- [ ] Add attention pattern visualization for debugging
- [ ] Document all conditional code paths in Python that affect behavior

---

## Related Files

- `E2E/test_sentences.json` - Test sentences
- `E2E/reference_outputs/` - Golden outputs (deterministic only)
- `swift/Sources/Nightingale/T3Model.swift` - Swift T3 generation
- `swift/Sources/Nightingale/AlignmentStreamAnalyzer.swift` - Swift analyzer
- `python/chatterbox/src/chatterbox/models/t3/t3.py` - Python T3 generation
