"""
Compare encoder outputs between Python and Swift.
The encoder produces the conditioning signal that feeds into the ODE solver.
"""
import torch
from safetensors.torch import load_file, save_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("COMPARE ENCODER OUTPUTS")
print("=" * 80)

MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICE_DIR = PROJECT_ROOT / "baked_voices" / "samantha"
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "forensic"

device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"\nLoading Python model on {device}...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load voice
voice_data = load_file(str(VOICE_DIR / "baked_voice.safetensors"))
speaker_emb = voice_data["t3.speaker_emb"].to(device)
speech_emb_matrix = voice_data["gen.embedding"].to(device)

# Load Python tokens
tokens_path = PROJECT_ROOT / "test_audio/cross_validate/python_speech_tokens.safetensors"
tokens_data = load_file(str(tokens_path))
generated_tokens = tokens_data["speech_tokens"].to(device)

print(f"\n📊 Inputs:")
print(f"   tokens: {generated_tokens.shape}")

# Hook the encoder to capture its output
flow = model.s3gen.flow

saved_encoder_output = None

# We need to hook into flow.inference to capture encoder output
# Let's manually run the encoder
print(f"\n🔬 Running encoder...")

with torch.no_grad():
    # Prepare inputs
    token_len = torch.tensor([generated_tokens.shape[0]], dtype=torch.long, device=device)

    # Run encoder - but we need to figure out the correct API
    # Check encoder signature
    import inspect
    print(f"   Encoder type: {type(flow.encoder).__name__}")
    print(f"   Encoder forward signature: {inspect.signature(flow.encoder.forward)}")

    # Try to call encoder
    # The encoder needs: x (tokens), x_lengths, and might need embedding
    # From Swift, we call: encoder(speechTokens, speechEmbMatrix)

    # Let's look at what flow.inference does
    # It likely calls encoder with tokens and embedding

    # Try different signatures
    try:
        # Attempt 1: encoder(tokens, embedding)
        encoder_output = flow.encoder(generated_tokens.unsqueeze(0), speech_emb_matrix)
        print(f"   ✅ Encoder called successfully")
    except Exception as e:
        print(f"   ❌ Encoder call failed: {e}")
        print(f"\n   Trying different signature...")
        try:
            # Attempt 2: encoder(tokens, lengths)
            encoder_output = flow.encoder(generated_tokens.unsqueeze(0), token_len)
            print(f"   ✅ Encoder called successfully (with lengths)")
        except Exception as e2:
            print(f"   ❌ Also failed: {e2}")
            print(f"\n   Encoder attributes: {[attr for attr in dir(flow.encoder) if not attr.startswith('_')][:20]}")
            exit(1)

    # encoder_output might be a tuple (output, lengths) or just output
    if isinstance(encoder_output, tuple):
        encoder_output, encoder_lengths = encoder_output
        print(f"   Encoder returned tuple")

    print(f"\n📊 Python encoder output:")
    print(f"   Shape: {encoder_output.shape}")
    print(f"   Range: [{encoder_output.min().item():.6f}, {encoder_output.max().item():.6f}]")
    print(f"   Mean: {encoder_output.mean().item():.6f}")
    print(f"   Std: {encoder_output.std().item():.6f}")

    # Save encoder output
    save_file({
        "encoder_output": encoder_output.cpu().contiguous()
    }, str(OUTPUT_DIR / "python_encoder_output.safetensors"))

    print(f"\n✅ Saved Python encoder output")

print("\n" + "=" * 80)
print("NEXT: Run Swift to save encoder output, then compare")
print("=" * 80)
