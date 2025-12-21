"""Debug script to compare Python attention patterns."""
import torch
import numpy as np
import sys
import logging
sys.path.insert(0, "chatterbox/src")

# Patch the analyzer BEFORE importing
from chatterbox.models.t3.inference import alignment_stream_analyzer

original_step = alignment_stream_analyzer.AlignmentStreamAnalyzer.step

def debug_step(self, logits, next_token=None):
    # Get position info before calling original
    aligned_attn = torch.stack(self.last_aligned_attns).mean(dim=0)
    i, j = self.text_tokens_slice
    
    if self.curr_frame_pos == 0:
        A_chunk = aligned_attn[j:, i:j].clone().cpu()
    else:
        A_chunk = aligned_attn[:, i:j].clone().cpu()
    
    S = j - i  # Text length
    
    # Raw position before masking
    raw_posn = A_chunk[-1].argmax().item()
    raw_max_attn = A_chunk[-1].max().item()
    
    # After masking  
    A_chunk_masked = A_chunk.clone()
    A_chunk_masked[:, self.curr_frame_pos + 1:] = 0
    masked_posn = A_chunk_masked[-1].argmax().item()
    
    if self.curr_frame_pos % 10 == 0 or self.curr_frame_pos < 5:
        print(f"🐍 PYTHON frame={self.curr_frame_pos}: rawPosn={raw_posn}, maskedPosn={masked_posn}, rawMaxAttn={raw_max_attn:.4f}, textPos={self.text_position}, S={S}")
    
    return original_step(self, logits, next_token)

alignment_stream_analyzer.AlignmentStreamAnalyzer.step = debug_step

from chatterbox import ChatterboxTTS

# Initialize model
print("Loading model...")
model = ChatterboxTTS.from_pretrained(device="mps")
print("Model loaded")

# Test the same sentence as Swift
text = "She was personally convinced that it worked on the first try."

# Generate with debug logging
print(f"\nGenerating: {text}")
print("="*60)

# Generate
audio = model.generate(text)
print(f"\nGenerated {len(audio)} samples")
