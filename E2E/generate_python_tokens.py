#!/usr/bin/env python3
"""Generate tokens using Python T3 and save them for Swift."""

import sys
sys.path.insert(0, '/Users/a10n/Projects/nightingale_TTS/python')

import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

print("=" * 80)
print("PYTHON TOKEN GENERATION")
print("=" * 80)

# Load model
print("\nLoading Python model...")
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals

model_dir = PROJECT_ROOT / "models" / "chatterbox"
device = "cpu"

model = ChatterboxMultilingualTTS.from_local(str(model_dir), device=device)
voice_path = PROJECT_ROOT / "baked_voices" / "samantha" / "baked_voice.pt"
model.conds = Conditionals.load(str(voice_path), map_location=device)

# Tokenize text
text = "Wow! I absolutely cannot believe that it worked on the first try!"
print(f"\nText: '{text}'")

text_tokens = model.tokenizer.text_to_tokens(text, language_id="en")
print(f"Text tokens: {text_tokens.shape}")

# Generate speech tokens using T3
print("\nGenerating speech tokens with Python T3...")
import torch.nn.functional as F
from chatterbox.models.s3tokenizer import drop_invalid_tokens

text_tokens = text_tokens.to(device)
text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
sot = model.t3.hp.start_text_token
eot = model.t3.hp.stop_text_token
text_tokens = F.pad(text_tokens, (1, 0), value=sot)
text_tokens = F.pad(text_tokens, (0, 1), value=eot)

with torch.inference_mode():
    speech_tokens = model.t3.inference(
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        max_new_tokens=1000,
        temperature=0.001,
        cfg_weight=0.5,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
    )
    speech_tokens = speech_tokens[0]
    speech_tokens = drop_invalid_tokens(speech_tokens).to(device)

print(f"\nGenerated {len(speech_tokens)} speech tokens")
print(f"First 10 tokens: {speech_tokens[:10].tolist()}")
print(f"All tokens: {speech_tokens.tolist()}")

# Save as numpy array
tokens_np = speech_tokens.cpu().numpy()
output_path = PROJECT_ROOT / "E2E" / "python_generated_tokens.npy"
np.save(output_path, tokens_np)

print(f"\n✅ Saved Python tokens to: {output_path}")
print(f"   Shape: {tokens_np.shape}")
print("\n" + "=" * 80)
