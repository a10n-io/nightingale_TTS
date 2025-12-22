#!/usr/bin/env python3
"""Check transformer configuration from weights."""

import torch
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
model_path = PROJECT_ROOT / "models" / "chatterbox" / "s3gen.pt"
state_dict = torch.load(str(model_path), map_location='cpu')

prefix = 'flow.decoder.estimator.down_blocks.0.1.0'

print("First transformer block weights:")
for key in sorted(state_dict.keys()):
    if key.startswith(prefix):
        tensor = state_dict[key]
        print(f"  {key[len('flow.decoder.estimator.'):]:60s} {list(tensor.shape)}")
