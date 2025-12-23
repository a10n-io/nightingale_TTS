"""
Systematic comparison of Python vs Swift mel generation pipeline.
"""
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path
import sys

PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("SYSTEMATIC PYTHON MEL GENERATION - TRACE ALL STAGES")
print("=" * 80)

MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model
print(f"\nLoading model on {device}...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Test both voices
for voice_name in ["samantha", "sujano"]:
    print("\n" + "=" * 80)
    print(f"VOICE: {voice_name.upper()}")
    print("=" * 80)
    
    # Load voice
    voice_ref = str(PROJECT_ROOT / f"baked_voices/{voice_name}/ref_audio.wav")
    print(f"\nPreparing conditionals with: {voice_ref}")
    model.prepare_conditionals(voice_ref)
    
    # Use same text as Swift
    test_text = "Wow! I absolutely cannot believe that it worked on the first try!"
    
    # Load tokens if they exist (from previous run), otherwise generate
    token_path = PROJECT_ROOT / f"test_audio/cross_validate/python_speech_tokens.safetensors"
    if token_path.exists():
        print(f"\nLoading existing tokens from: {token_path.name}")
        token_data = load_file(str(token_path))
        tokens = token_data["speech_tokens"].to(device)
    else:
        print("\nGenerating new tokens...")
        # Use generate to get speech tokens, then extract them
        text_tokens = model.tokenizer.text_to_tokens(test_text, language_id="en").to(device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
        text_tokens = torch.nn.functional.pad(text_tokens, (1, 0), value=model.t3.hp.start_text_token)
        text_tokens = torch.nn.functional.pad(text_tokens, (0, 1), value=model.t3.hp.stop_text_token)
        
        with torch.inference_mode():
            tokens = model.t3.inference(
                t3_cond=model.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=0.0001,
                cfg_weight=0.5,
                repetition_penalty=2.0,
                min_p=0.05,
                top_p=1.0,
            )[0]  # Extract conditional batch
    
    print(f"\n1️⃣ SPEECH TOKENS:")
    print(f"   Shape: {tokens.shape}")
    print(f"   First 20: {tokens[:20].tolist()}")
    print(f"   Last 20: {tokens[-20:].tolist()}")
    
    # Save tokens
    token_out_path = PROJECT_ROOT / f"test_audio/python_{voice_name}_tokens.safetensors"
    save_file({"speech_tokens": tokens.cpu()}, str(token_out_path))
    print(f"   ✅ Saved to: {token_out_path.name}")
    
    # Generate with detailed tracing
    with torch.no_grad():
        # Get encoder output
        s3gen = model.s3gen
        tokens_input = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens
        encoder_out = s3gen.encoder(tokens_input)
        
        print(f"\n2️⃣ ENCODER OUTPUT:")
        print(f"   Shape: {encoder_out.shape}")
        print(f"   Range: [{encoder_out.min().item():.6f}, {encoder_out.max().item():.6f}]")
        print(f"   Mean: {encoder_out.mean().item():.6f}, Std: {encoder_out.std().item():.6f}")
        
        # Save encoder output
        enc_path = PROJECT_ROOT / f"test_audio/python_{voice_name}_encoder.safetensors"
        save_file({"encoder_out": encoder_out.cpu()}, str(enc_path))
        print(f"   ✅ Saved to: {enc_path.name}")
        
        # Get full mel (includes decoder/flow processing)
        mel = s3gen.generate_mel(tokens_input, model.conds.s3gen_dict)
        
        print(f"\n3️⃣ FINAL MEL SPECTROGRAM:")
        print(f"   Shape: {mel.shape}")
        print(f"   Range: [{mel.min().item():.6f}, {mel.max().item():.6f}]")
        print(f"   Mean: {mel.mean().item():.6f}, Std: {mel.std().item():.6f}")
        
        # Analyze prompt vs generated regions
        L_pm = model.conds.s3gen_dict['prompt_feat'].shape[1]
        if mel.shape[2] > L_pm:
            prompt_mel = mel[:, :, :L_pm]
            gen_mel = mel[:, :, L_pm:]
            
            print(f"\n   📊 Prompt region (0-{L_pm}):")
            print(f"      Range: [{prompt_mel.min().item():.6f}, {prompt_mel.max().item():.6f}]")
            print(f"      Mean: {prompt_mel.mean().item():.6f}")
            
            print(f"\n   📊 Generated region ({L_pm}+):")
            print(f"      Range: [{gen_mel.min().item():.6f}, {gen_mel.max().item():.6f}]")
            print(f"      Mean: {gen_mel.mean().item():.6f}")
            
            if gen_mel.max().item() > 0:
                print(f"      ⚠️  WARNING: Generated mel has positive values!")
        
        # Save mel
        mel_path = PROJECT_ROOT / f"test_audio/python_{voice_name}_mel.safetensors"
        save_file({"mel": mel.cpu()}, str(mel_path))
        print(f"\n   ✅ Saved to: {mel_path.name}")
        
        # Generate audio
        audio = s3gen.mel2wav(mel)
        
        print(f"\n4️⃣ VOCODER OUTPUT:")
        print(f"   Shape: {audio.shape}")
        print(f"   Range: [{audio.min().item():.6f}, {audio.max().item():.6f}]")
        print(f"   Mean: {audio.mean().item():.6f}, Std: {audio.std().item():.6f}")
        
        # Save audio
        import torchaudio
        audio_path = PROJECT_ROOT / f"test_audio/python_{voice_name}_full.wav"
        torchaudio.save(str(audio_path), audio.cpu(), 24000)
        print(f"   ✅ Saved to: {audio_path.name}")

print("\n" + "=" * 80)
print("PYTHON TRACING COMPLETE")
print("=" * 80)
