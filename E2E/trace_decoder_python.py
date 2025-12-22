#!/usr/bin/env python3
"""Trace Python decoder step-by-step for comparison with Swift."""
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
print(f"First 10: {tokens[0, :10].tolist()}")

# Get the flow model
flow = model.s3gen.flow

# Prepare inputs exactly like s3gen.generate()
with torch.no_grad():
    # 1. Speaker embedding
    spk_emb = model.voice_spk_emb.to(model.device)  # [1, 192]
    spk_emb_norm = spk_emb / (torch.sqrt(torch.sum(spk_emb * spk_emb, dim=1, keepdim=True)) + 1e-8)
    spk_cond = flow.spk_embed_affine_layer(spk_emb_norm).squeeze(1)  # [1, 80]

    print(f"\nSpeaker embedding:")
    print(f"  spk_cond shape: {spk_cond.shape}")
    print(f"  spk_cond range: [{spk_cond.min():.6f}, {spk_cond.max():.6f}]")
    print(f"  spk_cond mean: {spk_cond.mean():.6f}")
    print(f"  spk_cond first 10: {spk_cond[0, :10].tolist()}")

    # 2. Prompt (use same as cross-validation)
    prompt_token = model.prompt_token.to(model.device)  # [1, 250]
    prompt_feat = model.prompt_feat.to(model.device)   # [1, 500, 80]

    # 3. Concatenate tokens
    inputs = torch.cat([prompt_token, tokens], dim=1)  # [1, 348]

    # 4. Embed
    x = flow.input_embedding(inputs)  # [1, 348, 512]
    print(f"\nToken embedding:")
    print(f"  x shape: {x.shape}")
    print(f"  x range: [{x.min():.6f}, {x.max():.6f}]")

    # 5. Encode
    xs_lens = torch.tensor([x.shape[1]], device=model.device)  # [1] = sequence length
    h, _ = flow.encoder(x, xs_lens)  # [1, 696, 512], returns (output, mask)
    mu = flow.encoder_proj(h)  # [1, 696, 80]
    print(f"\nEncoder output (mu):")
    print(f"  mu shape: {mu.shape}")
    print(f"  mu range: [{mu.min():.6f}, {mu.max():.6f}]")

    # 6. Prepare conditioning (prompt mels + zeros)
    # Prompt is 250 tokens * 2 upsampling = 500 mels
    # New tokens: 98 tokens * 2 upsampling = 196 mels
    # Total: 696 mels
    L_total = mu.shape[1]  # 696
    L_pm = 500  # prompt mel length (250 tokens * 2)
    L_new = L_total - L_pm   # 196

    # Conds: ground truth prompt mels (500) + zeros for new mels (196)
    mu_zeros = torch.zeros(1, L_new, 80, device=model.device, dtype=mu.dtype)
    conds = torch.cat([prompt_feat, mu_zeros], dim=1)  # [1, 696, 80]

    print(f"\nConditioning:")
    print(f"  conds shape: {conds.shape}")
    print(f"  conds range: [{conds.min():.6f}, {conds.max():.6f}]")
    print(f"  L_total={L_total}, L_pm={L_pm}, L_new={L_new}")

    # 7. Transpose for decoder [B, C, T]
    mu_t = mu.transpose(1, 2)      # [1, 80, 696]
    conds_t = conds.transpose(1, 2)  # [1, 80, 696]
    spk_cond_expanded = spk_cond.unsqueeze(2).expand(-1, -1, mu_t.shape[2])  # [1, 80, 696]

    # 8. Create noise
    torch.manual_seed(0)
    noise = torch.randn(1, 80, L_total, device=model.device, dtype=mu.dtype)  # [1, 80, 696]

    print(f"\nNoise:")
    print(f"  noise shape: {noise.shape}")
    print(f"  noise range: [{noise.min():.6f}, {noise.max():.6f}]")

    # 9. Create mask (all ones for no padding)
    mask = torch.ones(1, 1, L_total, device=model.device, dtype=mu.dtype)  # [1, 1, 696]

    # 10. Call decoder for ONE step (t=0)
    t = torch.tensor([0.0], device=model.device, dtype=mu.dtype)

    print(f"\n=== DECODER FORWARD PASS (t=0) ===")
    print(f"Inputs:")
    print(f"  x (noise): {noise.shape}, range=[{noise.min():.6f}, {noise.max():.6f}]")
    print(f"  mu: {mu_t.shape}, range=[{mu_t.min():.6f}, {mu_t.max():.6f}]")
    print(f"  spks: {spk_cond.shape}, range=[{spk_cond.min():.6f}, {spk_cond.max():.6f}]")
    print(f"  cond: {conds_t.shape}, range=[{conds_t.min():.6f}, {conds_t.max():.6f}]")
    print(f"  mask: {mask.shape}, all ones")

    # Call decoder estimator (single conditional pass)
    output = flow.decoder.estimator(
        x=noise,
        mask=mask,
        mu=mu_t,
        t=t,
        spks=spk_cond,
        cond=conds_t
    )

    print(f"\nDecoder output:")
    print(f"  shape: {output.shape}")
    print(f"  range: [{output.min():.6f}, {output.max():.6f}]")
    print(f"  mean: {output.mean():.6f}")

    # Save everything for Swift to compare (make contiguous)
    save_dict = {
        "noise": noise.contiguous().cpu(),
        "mu": mu_t.contiguous().cpu(),
        "spk_cond": spk_cond.contiguous().cpu(),
        "conds": conds_t.contiguous().cpu(),
        "mask": mask.contiguous().cpu(),
        "t": t.contiguous().cpu(),
        "decoder_output": output.contiguous().cpu(),
    }

    output_file = "test_audio/python_decoder_trace.safetensors"
    st.save_file(save_dict, output_file)
    print(f"\n✅ Saved trace to {output_file}")
    print(f"Swift can now load these and compare outputs!")
