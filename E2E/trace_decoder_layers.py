#!/usr/bin/env python3
"""Trace Python decoder layer-by-layer for detailed comparison with Swift."""
import torch
import numpy as np
from pathlib import Path
import safetensors.torch as st

# Load model
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Loading model...")
MODELS_DIR = Path("models/chatterbox")
VOICES_DIR = Path("baked_voices")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load voice
print("Loading voice: samantha")
from safetensors.torch import load_file
voice_path = VOICES_DIR / "samantha" / "baked_voice.safetensors"
voice_data = load_file(str(voice_path))
model.prompt_token = voice_data["gen.prompt_token"]
model.prompt_feat = voice_data["gen.prompt_feat"]
model.voice_spk_emb = voice_data["gen.embedding"]
print("Voice loaded")

# Load Python tokens
cross_val_dir = Path("test_audio/cross_validate")
python_tokens_file = cross_val_dir / "python_speech_tokens.safetensors"
tokens_data = st.load_file(python_tokens_file)
tokens = tokens_data["speech_tokens"].to(model.device)  # [98]
if tokens.ndim == 1:
    tokens = tokens.unsqueeze(0)  # [1, 98]

print(f"Tokens shape: {tokens.shape}")

# Get the flow model and decoder
flow = model.s3gen.flow
decoder = flow.decoder.estimator  # The actual decoder network

# Prepare inputs exactly like before
with torch.no_grad():
    # 1. Speaker embedding
    spk_emb = model.voice_spk_emb.to(model.device)  # [1, 192]
    spk_emb_norm = spk_emb / (torch.sqrt(torch.sum(spk_emb * spk_emb, dim=1, keepdim=True)) + 1e-8)
    spk_cond = flow.spk_embed_affine_layer(spk_emb_norm).squeeze(1)  # [1, 80]

    # 2. Prompt
    prompt_token = model.prompt_token.to(model.device)  # [1, 250]
    prompt_feat = model.prompt_feat.to(model.device)   # [1, 500, 80]

    # 3. Concatenate tokens
    inputs = torch.cat([prompt_token, tokens], dim=1)  # [1, 348]

    # 4. Embed
    x = flow.input_embedding(inputs)  # [1, 348, 512]

    # 5. Encode
    xs_lens = torch.tensor([x.shape[1]], device=model.device)
    h, _ = flow.encoder(x, xs_lens)
    mu = flow.encoder_proj(h)  # [1, 696, 80]

    # 6. Prepare conditioning
    L_total = mu.shape[1]  # 696
    L_pm = 500
    L_new = L_total - L_pm   # 196

    mu_zeros = torch.zeros(1, L_new, 80, device=model.device, dtype=mu.dtype)
    conds = torch.cat([prompt_feat, mu_zeros], dim=1)  # [1, 696, 80]

    # 7. Transpose for decoder [B, C, T]
    mu_t = mu.transpose(1, 2)      # [1, 80, 696]
    conds_t = conds.transpose(1, 2)  # [1, 80, 696]

    # 8. Create noise
    torch.manual_seed(0)
    noise = torch.randn(1, 80, L_total, device=model.device, dtype=mu.dtype)  # [1, 80, 696]

    # 9. Create mask
    mask = torch.ones(1, 1, L_total, device=model.device, dtype=mu.dtype)

    # 10. Time embedding (t=0)
    t = torch.tensor([0.0], device=model.device, dtype=mu.dtype)

    print(f"\n=== DECODER LAYER-BY-LAYER TRACE ===")

    # Get time embedding - match decoder's forward pass
    # Decoder concatenates t with spk embedding first
    t_expanded = t.unsqueeze(1)  # [1, 1]
    spk_t = torch.cat([t_expanded, spk_cond], dim=1)  # [1, 81]
    t_emb = decoder.time_mlp(spk_t)  # [1, 1024]
    print(f"time_mlp: [{t_emb.min():.6f}, {t_emb.max():.6f}], mean={t_emb.mean():.6f}")

    # Expand speaker embedding
    spk_expanded = spk_cond.unsqueeze(2).expand(-1, -1, L_total)  # [1, 80, 696]

    # Concatenate inputs
    h = torch.cat([noise, mu_t, spk_expanded, conds_t], dim=1)  # [1, 320, 696]
    print(f"concat: [{h.min():.6f}, {h.max():.6f}], mean={h.mean():.6f}")

    # Down block
    down = decoder.down_blocks[0]
    h = down[0](h, mask=mask, t_emb=t_emb)  # CausalResNetBlock
    print(f"down.resnet: [{h.min():.6f}, {h.max():.6f}], mean={h.mean():.6f}")

    h_reshaped = h.transpose(1, 2)  # [1, 696, 256] for transformers
    for i, tfmr in enumerate(down[1]):
        h_reshaped = tfmr(h_reshaped)
        if i == 0:
            print(f"down.tfmr[{i}]: [{h_reshaped.min():.6f}, {h_reshaped.max():.6f}], mean={h_reshaped.mean():.6f}")
    h = h_reshaped.transpose(1, 2)  # Back to [1, 256, 696]
    print(f"down.transformers: [{h.min():.6f}, {h.max():.6f}], mean={h.mean():.6f}")

    skip = h
    if hasattr(down, 'down'):
        h = down.down(h * mask)

    # Mid blocks
    for i, mid in enumerate(decoder.mid_blocks):
        h = mid[0](h, mask=mask, t_emb=t_emb)  # ResNet
        h_reshaped = h.transpose(1, 2)
        for tfmr in mid[1]:
            h_reshaped = tfmr(h_reshaped)
        h = h_reshaped.transpose(1, 2)
        if i == 0 or i == 11:
            print(f"mid[{i}]: [{h.min():.6f}, {h.max():.6f}], mean={h.mean():.6f}")

    # Up block
    h = torch.cat([h[:, :, :skip.shape[2]], skip], dim=1)  # [1, 512, 696]
    print(f"up.concat: [{h.min():.6f}, {h.max():.6f}], mean={h.mean():.6f}")

    up = decoder.up_blocks[0]
    h = up[0](h, mask=mask, t_emb=t_emb)  # ResNet
    print(f"up.resnet: [{h.min():.6f}, {h.max():.6f}], mean={h.mean():.6f}")

    h_reshaped = h.transpose(1, 2)
    for tfmr in up[1]:
        h_reshaped = tfmr(h_reshaped)
    h = h_reshaped.transpose(1, 2)
    print(f"up.transformers: [{h.min():.6f}, {h.max():.6f}], mean={h.mean():.6f}")

    if hasattr(up, 'up'):
        h = up.up(h * mask)
        print(f"up.upLayer: [{h.min():.6f}, {h.max():.6f}], mean={h.mean():.6f}")

    # Final block
    h = decoder.final_block(h, mask=mask)
    print(f"finalBlock: [{h.min():.6f}, {h.max():.6f}], mean={h.mean():.6f}")

    # Final projection
    h = decoder.final_proj(h.transpose(1, 2))  # Conv expects [B, T, C]
    h = h.transpose(1, 2)  # Back to [B, C, T]
    print(f"finalProj: [{h.min():.6f}, {h.max():.6f}], mean={h.mean():.6f}")

    print(f"\n✅ Python decoder trace complete!")
