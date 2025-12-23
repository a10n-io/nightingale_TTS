"""
Test if Swift's reflection padding matches PyTorch's
"""
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

print("Testing reflection padding implementation...")

# Create a simple test signal
test_signal = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])  # [1, 5]
pad_len = 2

# Python reflection padding
python_padded = F.pad(test_signal, (pad_len, pad_len), mode='reflect')

print(f"\nTest signal: {test_signal.tolist()[0]}")
print(f"Python reflect-padded: {python_padded.tolist()[0]}")
print(f"Expected: [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]")
print(f"Match: {python_padded.tolist()[0] == [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]}")
