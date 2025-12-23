"""
Save Python decoder output (mel spectrogram) for comparison with Swift
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

print(f"Speaker emb shape: {speaker_emb.shape}")
print(f"Speech emb matrix shape: {speech_emb_matrix.shape}")
print(f"Prompt token shape: {prompt_token.shape}")
print(f"Prompt feat shape: {prompt_feat.shape}")
print(f"Generated tokens shape: {generated_tokens.shape}")

# Prepare lengths
token_len = torch.tensor([generated_tokens.shape[0]], dtype=torch.long, device=device)
prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.long, device=device)
prompt_feat_len = torch.tensor([prompt_feat.shape[1]], dtype=torch.long, device=device)

# Run decoder
with torch.no_grad():
    mel, _ = model.s3gen.flow.inference(
        token=generated_tokens.unsqueeze(0),
        token_len=token_len,
        prompt_token=prompt_token,
        prompt_token_len=prompt_token_len,
        prompt_feat=prompt_feat,
        prompt_feat_len=prompt_feat_len,
        embedding=speech_emb_matrix,
        finalize=speaker_emb,
        n_timesteps=10
    )
    
    print(f"\nPython decoder mel output:")
    print(f"  Shape: {mel.shape}")
    print(f"  Mean: {mel.mean().item():.6f}")
    print(f"  Std: {mel.std().item():.6f}")
    print(f"  Range: [{mel.min().item():.6f}, {mel.max().item():.6f}]")

    # Save
    save_file(
        {"mel": mel.squeeze(0).cpu().contiguous()},  # Remove batch dim and move to CPU
        str(OUTPUT_DIR / "python_decoder_mel.safetensors")
    )
    print(f"\n✅ Saved to: {OUTPUT_DIR}/python_decoder_mel.safetensors")
