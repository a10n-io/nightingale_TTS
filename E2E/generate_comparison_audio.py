#!/usr/bin/env python3
"""
Generate comparison audio files for Swift/Python verification.

Creates 4 audio files:
- python_tokens_python_audio.wav: Python T3 tokens -> Python S3Gen audio
- swift_tokens_python_audio.wav: Swift T3 tokens -> Python S3Gen audio
- python_tokens_swift_audio.wav: Python T3 tokens -> Swift S3Gen audio (via Swift GenerateAudio)
- swift_tokens_swift_audio.wav: Swift T3 tokens -> Swift S3Gen audio (via Swift GenerateAudio)

This script generates the Python audio files. Swift GenerateAudio generates the Swift audio files.
"""

import torch
import torchaudio as ta
import numpy as np
import torch.nn.functional as F
from pathlib import Path

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
MODEL_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICE_DIR = PROJECT_ROOT / "baked_voices"
OUTPUT_DIR = PROJECT_ROOT / "test_audio"

# Note: Swift GenerateAudio cleans files, Python script just adds to them

print("=" * 60)
print("GENERATING COMPARISON AUDIO FILES")
print("=" * 60)

# Set deterministic seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Test text (same as Swift uses)
TEXT = "Do you think the model can handle the rising intonation at the end of this sentence?"
LANGUAGE = "en"

# Load model
print("\nLoading Python model...")
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals, punc_norm
from chatterbox.models.s3gen.s3gen import drop_invalid_tokens

device = "cpu"
model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device=device)

# Load voice
voice_path = VOICE_DIR / "samantha" / "baked_voice.pt"
model.conds = Conditionals.load(str(voice_path), map_location=device)
print(f"  Voice: samantha")

# ============================================================================
# Generate Python speech tokens with T3
# ============================================================================
print("\n" + "=" * 60)
print("1. GENERATING PYTHON SPEECH TOKENS (T3)")
print("=" * 60)

# Reset seed
torch.manual_seed(SEED)

# Tokenize text
text_normalized = punc_norm(TEXT)
text_tokens = model.tokenizer.text_to_tokens(text_normalized, language_id=LANGUAGE.lower())
text_tokens = text_tokens.to(device)
print(f"  Text tokens: {text_tokens.shape}")

# Prepare for T3 (CFG + SOT/EOT padding)
text_tokens_cfg = torch.cat([text_tokens, text_tokens], dim=0)
sot = model.t3.hp.start_text_token
eot = model.t3.hp.stop_text_token
text_tokens_cfg = F.pad(text_tokens_cfg, (1, 0), value=sot)
text_tokens_cfg = F.pad(text_tokens_cfg, (0, 1), value=eot)
print(f"  Text tokens with CFG: {text_tokens_cfg.shape}")

# Generate speech tokens
print("  Running T3 generation...")
with torch.inference_mode():
    speech_tokens = model.t3.inference(
        t3_cond=model.conds.t3,
        text_tokens=text_tokens_cfg,
        max_new_tokens=100,  # Match Swift
        temperature=0.001,   # Near deterministic
        cfg_weight=0.5,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
    )
    speech_tokens = speech_tokens[0]
    if speech_tokens.dim() == 1:
        speech_tokens = speech_tokens.unsqueeze(0)
    speech_tokens = drop_invalid_tokens(speech_tokens)

if speech_tokens.dim() == 1:
    speech_tokens = speech_tokens.unsqueeze(0)

python_speech_tokens = speech_tokens.squeeze(0).cpu().numpy()
print(f"  Generated {len(python_speech_tokens)} speech tokens")
print(f"  Token values (first 20): {python_speech_tokens[:20].tolist()}")

# Save Python tokens for Swift to use
np.save(OUTPUT_DIR / "python_speech_tokens.npy", python_speech_tokens.astype(np.int32))
print(f"  Saved: python_speech_tokens.npy")

# ============================================================================
# Generate audio from Python tokens (Python S3Gen)
# ============================================================================
print("\n" + "=" * 60)
print("2. PYTHON TOKENS -> PYTHON AUDIO")
print("=" * 60)

gen_conds = model.conds.gen

print("  Running S3Gen inference...")
with torch.inference_mode():
    audio = model.s3gen.inference(
        speech_tokens=speech_tokens,
        ref_dict=gen_conds,
        drop_invalid_tokens=False,
        n_cfm_timesteps=10,
    )

if isinstance(audio, tuple):
    audio = audio[0]

print(f"  Audio shape: {audio.shape}")
print(f"  Audio range: [{audio.min().item():.4f}, {audio.max().item():.4f}]")

# Normalize and save
if audio.dim() == 3:
    audio = audio.squeeze(0)
elif audio.dim() == 1:
    audio = audio.unsqueeze(0)

audio_max = audio.abs().max()
if audio_max > 1.0:
    audio = audio / audio_max

wav_path = OUTPUT_DIR / "python_tokens_python_audio.wav"
ta.save(str(wav_path), audio.cpu(), 24000)
print(f"  Saved: {wav_path}")
print(f"  Duration: {audio.shape[1] / 24000:.2f}s")

# ============================================================================
# Check for Swift tokens
# ============================================================================
print("\n" + "=" * 60)
print("3. SWIFT TOKENS -> PYTHON AUDIO")
print("=" * 60)

# Try to load Swift tokens from safetensors
swift_tokens_path = PROJECT_ROOT / "E2E" / "swift_generated_tokens.safetensors"
if swift_tokens_path.exists():
    from safetensors import safe_open
    with safe_open(swift_tokens_path, framework="pt") as f:
        swift_tokens_tensor = f.get_tensor("tokens")
    swift_speech_tokens = swift_tokens_tensor.squeeze().numpy()
    print(f"  Loaded Swift tokens: {len(swift_speech_tokens)} tokens")
    print(f"  Token values (first 20): {swift_speech_tokens[:20].tolist()}")

    # Convert to torch
    speech_tokens_swift = torch.tensor(swift_speech_tokens, dtype=torch.long, device=device).unsqueeze(0)

    print("  Running S3Gen inference...")
    with torch.inference_mode():
        audio = model.s3gen.inference(
            speech_tokens=speech_tokens_swift,
            ref_dict=gen_conds,
            drop_invalid_tokens=False,
            n_cfm_timesteps=10,
        )

    if isinstance(audio, tuple):
        audio = audio[0]

    print(f"  Audio shape: {audio.shape}")
    print(f"  Audio range: [{audio.min().item():.4f}, {audio.max().item():.4f}]")

    if audio.dim() == 3:
        audio = audio.squeeze(0)
    elif audio.dim() == 1:
        audio = audio.unsqueeze(0)

    audio_max = audio.abs().max()
    if audio_max > 1.0:
        audio = audio / audio_max

    wav_path = OUTPUT_DIR / "swift_tokens_python_audio.wav"
    ta.save(str(wav_path), audio.cpu(), 24000)
    print(f"  Saved: {wav_path}")
    print(f"  Duration: {audio.shape[1] / 24000:.2f}s")
else:
    print(f"  Swift tokens not found at {swift_tokens_path}")
    print("  Run Swift GenerateAudio with usePythonTokens=false first")
    print("  Skipping swift_tokens_python_audio.wav")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\nPython tokens saved for Swift to use:")
print(f"  test_audio/python_speech_tokens.npy ({len(python_speech_tokens)} tokens)")
print(f"\nToken array for Swift code:")
print(f"let pythonTokens = {python_speech_tokens.tolist()}")

print("\nGenerated files:")
for f in ["python_tokens_python_audio.wav", "swift_tokens_python_audio.wav"]:
    path = OUTPUT_DIR / f
    if path.exists():
        print(f"  ✓ {f}")
    else:
        print(f"  ✗ {f} (not generated)")

print("\nNext steps:")
print("  1. Update Swift GenerateAudio with the Python token array above")
print("  2. Run Swift GenerateAudio with usePythonTokens=true")
print("  3. Run Swift GenerateAudio with usePythonTokens=false")
print("  4. Run this script again to generate swift_tokens_python_audio.wav")
