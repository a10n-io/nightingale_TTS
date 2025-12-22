#!/usr/bin/env python3
"""
Cross-validation script for Python/Swift parity testing.
Generates speech tokens and audio, saves tokens for Swift to use.
Also loads Swift-generated tokens and synthesizes audio from them.
"""
import sys
sys.path.insert(0, str(__file__).replace('/E2E/cross_validate_python.py', '/python/chatterbox/src'))

import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file, save_file
import soundfile as sf

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICES_DIR = PROJECT_ROOT / "baked_voices"
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "cross_validate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CROSS-VALIDATION: Python T3 + S3Gen")
print("=" * 80)

# Test text - MUST MATCH Swift test
TEST_TEXT = "Wow! I absolutely cannot believe that it worked on the first try!"
print(f"\nTest text: \"{TEST_TEXT}\"")

# Parameters - MUST MATCH Swift
TEMPERATURE = 0.0001
CFG_WEIGHT = 0.5
REPETITION_PENALTY = 2.0
TOP_P = 1.0
MIN_P = 0.05

# Load model
print("\nLoading ChatterboxMultilingualTTS...")
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, punc_norm, Conditionals
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.s3tokenizer import drop_invalid_tokens
import torch.nn.functional as F

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)
print("Model loaded")

# Load voice - use samantha (same as Swift)
print("\nLoading voice: samantha")
voice_path = VOICES_DIR / "samantha" / "baked_voice.safetensors"
voice_data = load_file(str(voice_path))

t3_cond = T3Cond(
    speaker_emb=voice_data["t3.speaker_emb"].to(device),
    cond_prompt_speech_tokens=voice_data["t3.cond_prompt_speech_tokens"].to(device),
    emotion_adv=(torch.ones(1, 1, 1) * 0.5).to(device),
)

prompt_token = voice_data["gen.prompt_token"].to(device)
prompt_feat = voice_data["gen.prompt_feat"].to(device)
gen_dict = {
    "embedding": voice_data["gen.embedding"].to(device),
    "prompt_token": prompt_token,
    "prompt_feat": prompt_feat,
    "prompt_token_len": torch.tensor([prompt_token.shape[1]], device=device),
    "prompt_feat_len": torch.tensor([prompt_feat.shape[1]], device=device),
}

model.conds = Conditionals(t3_cond, gen_dict)
print("Voice loaded")

# =============================================================================
# STEP 1: Generate speech tokens with Python T3
# =============================================================================
print("\n" + "=" * 80)
print("STEP 1: Generate speech tokens with Python T3")
print("=" * 80)

# Tokenize text (same as model.generate but we want intermediate tokens)
text = punc_norm(TEST_TEXT)
text_tokens = model.tokenizer.text_to_tokens(text, language_id="en").to(device)
text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # CFG needs 2 seqs

sot = model.t3.hp.start_text_token
eot = model.t3.hp.stop_text_token
text_tokens = F.pad(text_tokens, (1, 0), value=sot)
text_tokens = F.pad(text_tokens, (0, 1), value=eot)

print(f"Text tokens shape: {text_tokens.shape}")
print(f"Text tokens[:20]: {text_tokens[0, :20].tolist()}")

with torch.inference_mode():
    speech_tokens = model.t3.inference(
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        max_new_tokens=1000,
        temperature=TEMPERATURE,
        cfg_weight=CFG_WEIGHT,
        repetition_penalty=REPETITION_PENALTY,
        min_p=MIN_P,
        top_p=TOP_P,
    )
    # Extract conditional batch
    python_speech_tokens = speech_tokens[0]
    python_speech_tokens = drop_invalid_tokens(python_speech_tokens)

print(f"Python speech tokens: {len(python_speech_tokens)}")
print(f"First 20: {python_speech_tokens[:20].tolist()}")
print(f"Last 20: {python_speech_tokens[-20:].tolist()}")

# Save Python tokens
save_file(
    {"speech_tokens": python_speech_tokens.cpu()},
    str(OUTPUT_DIR / "python_speech_tokens.safetensors")
)
print(f"Saved: {OUTPUT_DIR / 'python_speech_tokens.safetensors'}")

# =============================================================================
# STEP 2: Generate audio from Python tokens with Python S3Gen
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: Python tokens -> Python S3Gen")
print("=" * 80)

with torch.inference_mode():
    wav, _ = model.s3gen.inference(
        speech_tokens=python_speech_tokens.to(device),
        ref_dict=model.conds.gen,
    )
    python_audio = wav.squeeze(0).detach().cpu().numpy()

print(f"Generated {len(python_audio)} samples ({len(python_audio)/24000:.2f}s)")
sf.write(str(OUTPUT_DIR / "python_tokens_python_audio.wav"), python_audio, 24000)
print(f"Saved: python_tokens_python_audio.wav")

# =============================================================================
# STEP 3: Load Swift tokens and generate audio with Python S3Gen
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: Swift tokens -> Python S3Gen")
print("=" * 80)

swift_tokens_path = OUTPUT_DIR / "swift_speech_tokens.safetensors"
if swift_tokens_path.exists():
    swift_data = load_file(str(swift_tokens_path))
    swift_speech_tokens = swift_data["speech_tokens"].to(device)

    print(f"Swift speech tokens: {len(swift_speech_tokens)}")
    print(f"First 20: {swift_speech_tokens[:20].tolist()}")
    print(f"Last 20: {swift_speech_tokens[-20:].tolist()}")

    # Compare tokens
    min_len = min(len(python_speech_tokens), len(swift_speech_tokens))
    matches = (python_speech_tokens[:min_len] == swift_speech_tokens[:min_len]).sum().item()
    print(f"\nToken comparison (first {min_len}):")
    print(f"  Matching: {matches}/{min_len} ({100*matches/min_len:.1f}%)")

    # Find first difference
    for i in range(min_len):
        if python_speech_tokens[i] != swift_speech_tokens[i]:
            print(f"  First diff at index {i}: Python={python_speech_tokens[i].item()}, Swift={swift_speech_tokens[i].item()}")
            break

    with torch.inference_mode():
        wav, _ = model.s3gen.inference(
            speech_tokens=swift_speech_tokens,
            ref_dict=model.conds.gen,
        )
        swift_python_audio = wav.squeeze(0).detach().cpu().numpy()

    print(f"Generated {len(swift_python_audio)} samples ({len(swift_python_audio)/24000:.2f}s)")
    sf.write(str(OUTPUT_DIR / "swift_tokens_python_audio.wav"), swift_python_audio, 24000)
    print(f"Saved: swift_tokens_python_audio.wav")
else:
    print(f"Swift tokens not found: {swift_tokens_path}")
    print("Run the Swift cross-validation test first to generate swift_speech_tokens.safetensors")

print("\n" + "=" * 80)
print("CROSS-VALIDATION COMPLETE")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("Files:")
print("  - python_speech_tokens.safetensors (for Swift to load)")
print("  - python_tokens_python_audio.wav (baseline)")
if swift_tokens_path.exists():
    print("  - swift_tokens_python_audio.wav (tests Swift T3)")
