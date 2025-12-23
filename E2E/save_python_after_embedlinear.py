"""
Run Python encoder and save output after embedLinear (first layer)
to see where divergence starts
"""
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / "python" / "chatterbox" / "src"))

from chatterbox.models.s3gen.s3gen import S3Token2Mel

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICE_DIR = PROJECT_ROOT / "baked_voices" / "samantha"
FORENSIC_DIR = PROJECT_ROOT / "test_audio" / "forensic"

print("Loading Python model...")
flow_weights = load_file(str(MODELS_DIR / "s3gen.safetensors"))
model = S3Token2Mel()
model.load_state_dict(flow_weights, strict=False)
model.eval()

# Load tokens
voice_data = load_file(str(VOICE_DIR / "baked_voice.safetensors"))
tokens_data = load_file(str(PROJECT_ROOT / "test_audio/cross_validate/python_speech_tokens.safetensors"))
prompt_token = voice_data["gen.prompt_token"]  # [1, T_prompt]
speech_tokens = tokens_data["speech_tokens"]    # [T_speech]

# Concatenate tokens
full_tokens = torch.cat([prompt_token.squeeze(0), speech_tokens], dim=0)  # [348]

# Get input embeddings
with torch.no_grad():
    token_embs = model.flow.input_embedding(full_tokens)  # [348, 512]
    print(f"Token embeddings: {token_embs.shape}, mean={token_embs.mean().item():.6f}")

    # Add batch dimension
    x = token_embs.unsqueeze(0)  # [1, 348, 512]

    # Run through embedLinear (first layer of encoder)
    encoder = model.flow.encoder
    h = encoder.embed.out[0](x)  # embedLinear

    print(f"After embedLinear: {h.shape}")
    print(f"  Mean: {h.mean().item():.6f}")
    print(f"  Std: {h.std().item():.6f}")
    print(f"  Range: [{h.min().item():.6f}, {h.max().item():.6f}]")

    # Save
    save_file(
        {"after_embedlinear": h.squeeze(0).contiguous()},  # Remove batch dim to match Swift
        str(FORENSIC_DIR / "python_after_embedlinear.safetensors")
    )
    print(f"\n✅ Saved to: {FORENSIC_DIR}/python_after_embedlinear.safetensors")
