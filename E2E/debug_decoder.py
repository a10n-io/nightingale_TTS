#!/usr/bin/env python3
"""
Debug script to trace decoder inputs and outputs between Python and Swift.
"""
import torch
import numpy as np
from pathlib import Path
import safetensors.torch as st

# Load Python tokens
cross_val_dir = Path("test_audio/cross_validate")
python_tokens = st.load_file(cross_val_dir / "python_speech_tokens.safetensors")
tokens = python_tokens["speech_tokens"]  # Should be [1, 98]

print(f"Python tokens shape: {tokens.shape}")
print(f"First 10 tokens: {tokens[0, :10].tolist()}")

# Load the chatterbox model
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("\nLoading Chatterbox model...")
model = ChatterboxMultilingualTTS()
model.load_voice("samantha")

# Get encoder output (mu)
print("\nRunning encoder...")
with torch.no_grad():
    # Prepare inputs similar to what S3Gen does
    tokens_input = tokens.to(model.device)

    # Get speaker embedding
    spk_emb = model.voice_spk_emb.unsqueeze(0)  # [1, 256]

    # Run through encoder to get mu
    # This matches what s3gen.generate() does
    from chatterbox.models.s3gen.s3gen import S3Gen
    s3gen: S3Gen = model.s3gen

    # Token embedding
    x = s3gen.flow.input_emb(tokens_input)  # [1, T, 512]
    print(f"Token embedding shape: {x.shape}, range: [{x.min():.4f}, {x.max():.4f}]")

    # Encoder (includes upsampling 2x)
    h = s3gen.flow.encoder(x)  # [1, 2*T, 512]
    mu = s3gen.flow.encoder_proj(h)  # [1, 2*T, 80]
    print(f"Mu shape: {mu.shape}, range: [{mu.min():.4f}, {mu.max():.4f}]")

    # Transpose to [1, 80, 2*T] for decoder
    mu_t = mu.transpose(1, 2)
    print(f"Mu transposed shape: {mu_t.shape}")

    # Save for Swift to load
    save_dict = {
        "mu": mu_t.cpu(),
        "spk_emb": spk_emb.cpu(),
        "tokens": tokens_input.cpu()
    }
    st.save_file(save_dict, "test_audio/debug_decoder_inputs.safetensors")
    print(f"\nSaved decoder inputs to test_audio/debug_decoder_inputs.safetensors")
    print(f"  mu: {mu_t.shape}")
    print(f"  spk_emb: {spk_emb.shape}")
