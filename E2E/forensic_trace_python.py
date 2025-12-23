"""
Forensic trace of Python decoder - save intermediate outputs at every stage.
This establishes the ground truth that Swift must match.
"""
import torch
from safetensors.torch import load_file, save_file
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python/chatterbox/src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 80)
print("FORENSIC TRACE: PYTHON DECODER")
print("=" * 80)

MODELS_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICE_DIR = PROJECT_ROOT / "baked_voices" / "samantha"
OUTPUT_DIR = PROJECT_ROOT / "test_audio" / "forensic"
OUTPUT_DIR.mkdir(exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"\nLoading Python model on {device}...")
model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device)

# Load samantha voice
voice_data = load_file(str(VOICE_DIR / "baked_voice.safetensors"))
speaker_emb = voice_data["t3.speaker_emb"].to(device)
speech_emb_matrix = voice_data["gen.embedding"].to(device)
prompt_token = voice_data["gen.prompt_token"].to(device)
prompt_feat = voice_data["gen.prompt_feat"].to(device)

# Load Python tokens
tokens_path = PROJECT_ROOT / "test_audio/cross_validate/python_speech_tokens.safetensors"
tokens_data = load_file(str(tokens_path))
generated_tokens = tokens_data["speech_tokens"].to(device)

print(f"\n📊 Inputs:")
print(f"   tokens: {generated_tokens.shape}")
print(f"   speaker_emb: {speaker_emb.shape}")
print(f"   speech_emb_matrix: {speech_emb_matrix.shape}")
print(f"   prompt_token: {prompt_token.shape}")
print(f"   prompt_feat: {prompt_feat.shape}")

# Save inputs for Swift
save_file({
    "tokens": generated_tokens.cpu(),
    "speaker_emb": speaker_emb.cpu(),
    "speech_emb_matrix": speech_emb_matrix.cpu(),
    "prompt_token": prompt_token.cpu(),
    "prompt_feat": prompt_feat.cpu()
}, str(OUTPUT_DIR / "inputs.safetensors"))
print(f"\n✅ Saved inputs to forensic/inputs.safetensors")

# Trace through decoder with instrumentation
print(f"\n" + "=" * 80)
print("STAGE 1: CONDITION ENCODER")
print("=" * 80)

with torch.no_grad():
    # Compute condition
    cond = model.s3gen.cond_enc(speech_emb_matrix, speaker_emb)
    print(f"   cond output: {cond.shape}")
    print(f"   Range: [{cond.min().item():.6f}, {cond.max().item():.6f}]")
    print(f"   Mean: {cond.mean().item():.6f}")

    save_file({"cond": cond.cpu()}, str(OUTPUT_DIR / "01_cond.safetensors"))

    print(f"\n" + "=" * 80)
    print("STAGE 2: ENCODER")
    print("=" * 80)

    # Get encoder output
    encoder = model.s3gen.decoder.encoder
    encoder_out = encoder(generated_tokens, cond)
    print(f"   encoder output: {encoder_out.shape}")
    print(f"   Range: [{encoder_out.min().item():.6f}, {encoder_out.max().item():.6f}]")
    print(f"   Mean: {encoder_out.mean().item():.6f}")

    save_file({"encoder_out": encoder_out.cpu()}, str(OUTPUT_DIR / "02_encoder.safetensors"))

    print(f"\n" + "=" * 80)
    print("STAGE 3: DECODER INPUT PREPARATION")
    print("=" * 80)

    # Prepare decoder inputs (from decoder.forward)
    B, L = generated_tokens.shape
    prompt_len = prompt_token.shape[1]
    gen_len = L - prompt_len

    # Create x0 (initial noise)
    x0_prompt = prompt_feat  # [B, T_prompt, 80]
    x0_gen = torch.randn(B, gen_len, 80, device=device)
    x0 = torch.cat([x0_prompt, x0_gen], dim=1)  # [B, T, 80]

    print(f"   x0: {x0.shape}")
    print(f"   x0_prompt range: [{x0_prompt.min().item():.6f}, {x0_prompt.max().item():.6f}]")
    print(f"   x0_gen range: [{x0_gen.min().item():.6f}, {x0_gen.max().item():.6f}]")

    save_file({
        "x0": x0.cpu(),
        "x0_prompt": x0_prompt.cpu(),
        "x0_gen": x0_gen.cpu()
    }, str(OUTPUT_DIR / "03_x0.safetensors"))

    print(f"\n" + "=" * 80)
    print("STAGE 4: ODE SOLVER")
    print("=" * 80)

    # Trace ODE steps manually
    from torchdiffeq import odeint

    estimator = model.s3gen.decoder.estimator
    t_span = torch.linspace(0, 1, 11, device=device)

    # Store ODE trajectory
    ode_states = []

    def ode_func_traced(t, x):
        # Compute velocity
        v = estimator(x, encoder_out, t, cond)
        return v

    # Run ODE
    solution = odeint(ode_func_traced, x0, t_span, method='euler')
    xt = solution[-1]  # Final state

    print(f"   ODE trajectory: {solution.shape}")
    print(f"   Final xt: {xt.shape}")
    print(f"   xt range: [{xt.min().item():.6f}, {xt.max().item():.6f}]")
    print(f"   xt mean: {xt.mean().item():.6f}")

    # Analyze prompt vs generated regions
    xt_prompt = xt[:, :prompt_len, :]
    xt_gen = xt[:, prompt_len:, :]
    print(f"   xt_prompt mean: {xt_prompt.mean().item():.6f}, max: {xt_prompt.max().item():.6f}")
    print(f"   xt_gen mean: {xt_gen.mean().item():.6f}, max: {xt_gen.max().item():.6f}")

    save_file({
        "xt": xt.cpu(),
        "xt_prompt": xt_prompt.cpu(),
        "xt_gen": xt_gen.cpu()
    }, str(OUTPUT_DIR / "04_ode_output.safetensors"))

    print(f"\n" + "=" * 80)
    print("STAGE 5: FINAL PROJECTION")
    print("=" * 80)

    # Apply final projection
    final_proj = model.s3gen.decoder.final_proj

    # Need to transpose for conv: [B, T, C] -> [B, C, T]
    xt_transposed = xt.transpose(1, 2)
    mel = final_proj(xt_transposed)  # [B, 80, T]

    print(f"   final_proj output: {mel.shape}")
    print(f"   Range: [{mel.min().item():.6f}, {mel.max().item():.6f}]")
    print(f"   Mean: {mel.mean().item():.6f}")

    # Analyze regions
    mel_prompt = mel[:, :, :prompt_len]
    mel_gen = mel[:, :, prompt_len:]
    print(f"   mel_prompt mean: {mel_prompt.mean().item():.6f}, max: {mel_prompt.max().item():.6f}")
    print(f"   mel_gen mean: {mel_gen.mean().item():.6f}, max: {mel_gen.max().item():.6f}")

    save_file({
        "mel": mel.cpu(),
        "mel_prompt": mel_prompt.cpu(),
        "mel_gen": mel_gen.cpu()
    }, str(OUTPUT_DIR / "05_final_mel.safetensors"))

    print(f"\n" + "=" * 80)
    print("STAGE 6: VOCODER")
    print("=" * 80)

    # Run vocoder
    audio = model.s3gen.mel2wav(mel)
    print(f"   audio: {audio.shape}")
    print(f"   Range: [{audio.min().item():.6f}, {audio.max().item():.6f}]")
    print(f"   RMS: {audio.pow(2).mean().sqrt().item():.6f}")

    # Save audio
    import torchaudio
    torchaudio.save(str(OUTPUT_DIR / "python_audio.wav"), audio.cpu(), 24000)

print(f"\n" + "=" * 80)
print("✅ FORENSIC TRACE COMPLETE")
print("=" * 80)
print(f"\nAll intermediate outputs saved to: {OUTPUT_DIR}")
print(f"\nNext: Run forensic_trace_swift.swift to compare")
print("=" * 80)
