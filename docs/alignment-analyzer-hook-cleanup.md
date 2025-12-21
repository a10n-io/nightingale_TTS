# AlignmentStreamAnalyzer Hook Cleanup Fix

## Problem

When running multiple TTS inference calls in sequence, hooks registered by `AlignmentStreamAnalyzer` accumulate on the transformer layers. This causes:

1. **Memory leaks** - Each inference call adds new hooks without removing old ones
2. **Incorrect behavior** - Multiple hooks fire for each forward pass, corrupting attention capture
3. **Test failures** - Batch E2E tests fail while individual tests pass

## Root Cause

The `AlignmentStreamAnalyzer` registers PyTorch forward hooks on specific attention layers to capture alignment patterns:

```python
# In _add_attention_spy()
target_layer.register_forward_pre_hook(attention_pre_hook, with_kwargs=True)
target_layer.register_forward_hook(attention_forward_hook)
```

However:
1. Hook handles were not stored
2. No cleanup mechanism existed
3. `T3.inference()` creates a new analyzer each call (`self.compiled = False`)

## Solution

### 1. Store hook handles in AlignmentStreamAnalyzer

```python
class AlignmentStreamAnalyzer:
    def __init__(self, ...):
        # Store hook handles for cleanup
        self._hook_handles = []
        # ... rest of init
```

### 2. Save handles when registering hooks

```python
def _add_attention_spy(self, tfmr, buffer_idx, layer_idx, head_idx):
    # ...
    # Register both hooks and store handles for cleanup
    pre_handle = target_layer.register_forward_pre_hook(attention_pre_hook, with_kwargs=True)
    post_handle = target_layer.register_forward_hook(attention_forward_hook)
    self._hook_handles.append(pre_handle)
    self._hook_handles.append(post_handle)
```

### 3. Add cleanup method

```python
def cleanup(self):
    """Remove registered hooks to prevent accumulation across inference calls."""
    for handle in self._hook_handles:
        handle.remove()
    self._hook_handles = []
```

### 4. Call cleanup in T3.inference()

```python
# Before creating new analyzer - cleanup old one
if hasattr(self, 'patched_model') and self.patched_model is not None:
    if self.patched_model.alignment_stream_analyzer is not None:
        self.patched_model.alignment_stream_analyzer.cleanup()

# ... inference loop ...

# After inference completes
if self.patched_model.alignment_stream_analyzer is not None:
    self.patched_model.alignment_stream_analyzer.cleanup()

return predicted_tokens
```

## Files Changed

| File | Change |
|------|--------|
| `alignment_stream_analyzer.py` | Added `_hook_handles` list, stored handles, added `cleanup()` method |
| `t3.py` | Call `cleanup()` before creating new analyzer and after inference completes |

## Testing

Run E2E verification with multiple test cases:

```bash
python E2E/verify_e2e.py --voice samantha --steps 5
```

Individual tests should match batch results after this fix.

## Related

- See `docs/chatterbox-smart-gate-fix.md` for related AlignmentStreamAnalyzer improvements
- PyTorch hook documentation: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
