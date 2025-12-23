"""
Save Python decoder intermediate outputs at each ODE step
"""
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "python" / "chatterbox" / "src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICE_DIR = PROJECT_ROOT / "baked_voices" / "samantha"
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "forensic"
OUTPUT_DIR.mkdir(exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading Python model on {device}...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load baked voice and tokens
voice_data = load_file(str(VOICE_DIR / "baked_voice.safetensors"))
tokens_data = load_file(str(PROJECT_ROOT / "test_audio/cross_validate/python_speech_tokens.safetensors"))

speaker_emb = voice_data["t3.speaker_emb"].to(device)
speech_emb_matrix = voice_data["gen.embedding"].to(device)
prompt_token = voice_data["gen.prompt_token"].to(device)
prompt_feat = voice_data["gen.prompt_feat"].to(device)
generated_tokens = tokens_data["speech_tokens"].to(device)

# Prepare lengths
token_len = torch.tensor([generated_tokens.shape[0]], dtype=torch.long, device=device)
prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.long, device=device)
prompt_feat_len = torch.tensor([prompt_feat.shape[1]], dtype=torch.long, device=device)

print("\n" + "=" * 80)
print("TRACING PYTHON DECODER INTERMEDIATE OUTPUTS")
print("=" * 80)

with torch.no_grad():
    # Get the flow model
    flow = model.s3gen.flow

    # Manually trace through inference to capture intermediates
    # 1. Token embedding
    token = generated_tokens.unsqueeze(0)
    full_token = torch.cat([prompt_token, token], dim=1)
    full_token_len = torch.tensor([full_token.shape[1]], dtype=torch.long, device=device)

    # Embed tokens first
    token_emb = flow.input_embedding(full_token)

    # 2. Encoder
    h, _ = flow.encoder(token_emb, full_token_len)
    mu = flow.encoder_proj(h)

    print(f"\n1. Encoder output (mu):")
    print(f"   Shape: {mu.shape}")
    print(f"   Mean: {mu.mean().item():.8f}")
    print(f"   Std: {mu.std().item():.8f}")
    print(f"   Range: [{mu.min().item():.8f}, {mu.max().item():.8f}]")

    # Save mu
    save_file(
        {"mu": mu.squeeze(0).cpu().contiguous()},
        str(OUTPUT_DIR / "python_decoder_mu.safetensors")
    )

    # 3. Speaker embedding projection
    spk_emb_normalized = speech_emb_matrix / (torch.norm(speech_emb_matrix, dim=1, keepdim=True) + 1e-8)
    spk = flow.spk_embed_affine_layer(spk_emb_normalized)

    print(f"\n2. Speaker embedding projection (spk):")
    print(f"   Input (speech_emb_matrix): {speech_emb_matrix.shape}, mean={speech_emb_matrix.mean().item():.8f}")
    print(f"   After normalization: mean={spk_emb_normalized.mean().item():.8f}")
    print(f"   After projection (spk): {spk.shape}")
    print(f"   Mean: {spk.mean().item():.8f}")
    print(f"   Std: {spk.std().item():.8f}")
    print(f"   First 5 values: {spk[0, :5].tolist()}")

    save_file(
        {"spk": spk.cpu().contiguous()},
        str(OUTPUT_DIR / "python_decoder_spk.safetensors")
    )

    # 4. Prepare decoder inputs
    prompt_len = prompt_feat.shape[1]
    L_total = mu.shape[1]

    # conds: prompt mels + zeros
    zeros = torch.zeros(1, L_total - prompt_len, 80, device=device)
    conds = torch.cat([prompt_feat, zeros], dim=1)

    print(f"\n3. Decoder conditioning (conds):")
    print(f"   Shape: {conds.shape}")
    print(f"   Prompt region mean: {conds[0, :prompt_len].mean().item():.8f}")
    print(f"   Generated region (zeros) mean: {conds[0, prompt_len:].mean().item():.8f}")

    # 5. Initial noise
    torch.manual_seed(0)
    noise = torch.randn(1, 80, L_total, device=device)

    print(f"\n4. Initial noise:")
    print(f"   Shape: {noise.shape}")
    print(f"   Mean: {noise.mean().item():.8f}")
    print(f"   Std: {noise.std().item():.8f}")
    print(f"   First 5 values [0,0,:5]: {noise[0, 0, :5].tolist()}")

    save_file(
        {"noise": noise.cpu().contiguous()},
        str(OUTPUT_DIR / "python_decoder_noise.safetensors")
    )

    # Now run full inference to get final mel
    mel, _ = flow.inference(
        token=token,
        token_len=token_len,
        prompt_token=prompt_token,
        prompt_token_len=prompt_token_len,
        prompt_feat=prompt_feat,
        prompt_feat_len=prompt_feat_len,
        embedding=speech_emb_matrix,
        finalize=speaker_emb,
        n_timesteps=10
    )

    print(f"\n5. Final decoder output:")
    print(f"   Shape: {mel.shape}")
    print(f"   Mean: {mel.mean().item():.8f}")
    print(f"   Std: {mel.std().item():.8f}")

print("\n✅ Saved intermediate outputs to forensic/")
