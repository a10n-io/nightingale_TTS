#!/usr/bin/env python3
"""Test Swift vocoder with Python-generated mel spectrogram."""

import sys
sys.path.insert(0, '/Users/a10n/Projects/nightingale_TTS/python')

import torch
import numpy as np
from pathlib import Path
import scipy.io.wavfile as wavfile

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")

print("="*80)
print("GENERATE PYTHON MEL FOR SWIFT VOCODER TEST")
print("="*80)

# Load chatterbox model
print("\nLoading model...")
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals

model_dir = PROJECT_ROOT / "models" / "chatterbox"
device = "cpu"

model = ChatterboxMultilingualTTS.from_local(str(model_dir), device=device)
voice_path = PROJECT_ROOT / "baked_voices" / "samantha" / "baked_voice.pt"
model.conds = Conditionals.load(str(voice_path), map_location=device)

# Generate audio and extract mel
text = "Wow! I absolutely cannot believe that it worked on the first try!"
print(f"\nGenerating audio to get mel spectrogram...")
print(f"  Text: '{text}'")

with torch.inference_mode():
    # Get mel from s3gen
    import torch.nn.functional as F
    from chatterbox.models.t3.inference.sampling import drop_invalid_tokens

    # Tokenize and generate speech tokens
    text_tokens = model.tokenizer.text_to_tokens(text, language_id="en").to(device)
    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    text_tokens = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)

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

    # Get mel from s3gen
    mel, _ = model.s3gen.flow.inference(
        speech_tokens=speech_tokens,
        ref_dict=model.conds.gen,
    )

    print(f"\nPython-generated mel:")
    print(f"  Shape: {mel.shape}")
    print(f"  Range: [{mel.min().item():.4f}, {mel.max().item():.4f}]")
    print(f"  Mean: {mel.mean().item():.4f}")
    print(f"  Dtype: {mel.dtype}")

    # Save mel for Swift
    mel_np = mel.cpu().numpy()
    np.save(PROJECT_ROOT / "E2E" / "python_mel_for_swift.npy", mel_np)
    print(f"\n✅ Saved: E2E/python_mel_for_swift.npy")

    # Also generate audio with Python vocoder for comparison
    wav, _ = model.s3gen.mel2wav.inference(speech_feat=mel, f0=None)
    wav_np = wav.squeeze(0).detach().cpu().numpy()

    # Apply watermark
    watermarked = model.watermarker.apply_watermark(wav_np, sample_rate=24000)

    output_path = PROJECT_ROOT / "test_audio" / "python_vocoder_from_mel.wav"
    wavfile.write(str(output_path), 24000, watermarked)
    print(f"✅ Saved Python vocoder output: {output_path}")

print("\n" + "="*80)
print("Now run Swift with this mel to test the vocoder")
print("="*80)
