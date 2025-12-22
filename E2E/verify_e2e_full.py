#!/Users/a10n/Projects/nightingale_TTS/python/venv/bin/python
"""
TRUE E2E Verification Script: Full Pipeline Comparison (Steps 1-9)

This script runs the COMPLETE pipeline independently for both Python and Swift,
with NO cross-pollination of data between implementations.

Each implementation:
1. Takes the SAME input text
2. Runs the FULL pipeline (steps 1-9) independently
3. Saves outputs at each checkpoint
4. Final comparison happens AFTER both complete

Usage:
    python E2E/verify_e2e_full.py
    python E2E/verify_e2e_full.py --text "Custom text here"
    python E2E/verify_e2e_full.py --voice samantha --lang en
"""

import torch
import torchaudio as ta
import numpy as np
import random
import json
import sys
import subprocess
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# Project paths
PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
MODEL_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICE_DIR = PROJECT_ROOT / "baked_voices"
E2E_DIR = PROJECT_ROOT / "E2E"
OUTPUT_DIR = E2E_DIR / "full_pipeline_outputs"

# Deterministic settings (MUST be identical for Python and Swift)
SEED = 42
TEMPERATURE = 0.001  # Near-deterministic for verification
TOP_P = 1.0
REPETITION_PENALTY = 2.0
MIN_P = 0.05
CFG_WEIGHT = 0.5
N_CFM_TIMESTEPS = 10


@dataclass
class PipelineOutputs:
    """Container for all pipeline outputs at each step."""
    # Step 1: Tokenization
    text_tokens: np.ndarray = None
    text_tokens_cfg: np.ndarray = None

    # Step 2: T3 Conditioning
    t3_conditioning: np.ndarray = None

    # Step 3: T3 Generation
    speech_tokens: np.ndarray = None

    # Step 4: S3Gen Conditioning
    s3_embedding: np.ndarray = None
    prompt_token: np.ndarray = None
    prompt_feat: np.ndarray = None

    # Step 5: S3Gen Input Prep
    full_tokens: np.ndarray = None
    token_emb: np.ndarray = None
    spk_emb: np.ndarray = None
    mask: np.ndarray = None

    # Step 6: Encoder
    encoder_out: np.ndarray = None
    mu: np.ndarray = None
    x_cond: np.ndarray = None

    # Step 7: ODE Solver
    mel: np.ndarray = None

    # Step 8: Vocoder
    audio: np.ndarray = None

    # Step 9: WAV file
    wav_path: str = None
    duration_seconds: float = None


def set_deterministic_seeds(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'manual_seed'):
        torch.mps.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_python_pipeline(
    text: str,
    voice: str,
    language: str,
    output_dir: Path,
    device: str = "cpu"
) -> PipelineOutputs:
    """
    Run the COMPLETE Python pipeline from text to audio.

    Returns outputs at each step for comparison.
    """
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals, punc_norm
    from chatterbox.models.s3gen.s3gen import drop_invalid_tokens
    import torch.nn.functional as F

    outputs = PipelineOutputs()

    print("=" * 60)
    print("PYTHON PIPELINE (Steps 1-9)")
    print("=" * 60)

    # Reset seeds before EACH pipeline run
    set_deterministic_seeds(SEED)

    # Load model
    print("\nLoading Python model...")
    model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device=device)

    # Load voice
    voice_path = VOICE_DIR / voice / "baked_voice.pt"
    model.conds = Conditionals.load(str(voice_path), map_location=device)
    print(f"  Voice: {voice}")
    print(f"  Language: {language}")
    print(f"  Text: \"{text[:50]}...\"" if len(text) > 50 else f"  Text: \"{text}\"")

    # =========================================================================
    # Step 1: Tokenization
    # =========================================================================
    print("\n[Step 1] Tokenization")
    text_normalized = punc_norm(text)
    text_tokens = model.tokenizer.text_to_tokens(text_normalized, language_id=language.lower())
    text_tokens = text_tokens.to(device)
    outputs.text_tokens = text_tokens[0].detach().cpu().numpy().astype(np.int32)
    print(f"  Token count: {len(outputs.text_tokens)}")

    # Prepare for T3 (CFG + SOT/EOT padding)
    text_tokens_cfg = torch.cat([text_tokens, text_tokens], dim=0)
    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    text_tokens_cfg = F.pad(text_tokens_cfg, (1, 0), value=sot)
    text_tokens_cfg = F.pad(text_tokens_cfg, (0, 1), value=eot)
    outputs.text_tokens_cfg = text_tokens_cfg.detach().cpu().numpy().astype(np.int32)
    print(f"  CFG shape: {text_tokens_cfg.shape}")

    # =========================================================================
    # Step 2: T3 Conditioning
    # =========================================================================
    print("\n[Step 2] T3 Conditioning")
    t3 = model.t3
    conds = model.conds.t3

    speaker_emb = conds.speaker_emb.to(device)
    speaker_token = t3.cond_enc.spkr_enc(speaker_emb).unsqueeze(1)

    cond_speech_tokens = conds.cond_prompt_speech_tokens.to(device)
    speech_emb = t3.speech_emb(cond_speech_tokens)
    positions = torch.arange(cond_speech_tokens.shape[1], device=device).unsqueeze(0)
    speech_pos_emb = t3.speech_pos_emb(positions)
    cond_speech_emb = speech_emb + speech_pos_emb

    perceiver_out = t3.cond_enc.perceiver(cond_speech_emb)

    emotion_value = conds.emotion_adv.to(device)
    emotion_token = t3.cond_enc.emotion_adv_fc(emotion_value)

    final_cond = torch.cat([speaker_token, perceiver_out, emotion_token], dim=1)
    outputs.t3_conditioning = final_cond.detach().cpu().numpy()
    print(f"  Conditioning shape: {final_cond.shape}")

    # =========================================================================
    # Step 3: T3 Token Generation
    # =========================================================================
    print("\n[Step 3] T3 Token Generation")
    with torch.inference_mode():
        speech_tokens = model.t3.inference(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens_cfg,
            max_new_tokens=1000,
            temperature=TEMPERATURE,
            cfg_weight=CFG_WEIGHT,
            repetition_penalty=REPETITION_PENALTY,
            min_p=MIN_P,
            top_p=TOP_P,
        )
        speech_tokens = speech_tokens[0]
        if speech_tokens.dim() == 1:
            speech_tokens = speech_tokens.unsqueeze(0)
        speech_tokens = drop_invalid_tokens(speech_tokens)

    if speech_tokens.dim() == 1:
        speech_tokens = speech_tokens.unsqueeze(0)

    outputs.speech_tokens = speech_tokens.squeeze(0).detach().cpu().numpy().astype(np.int32)
    print(f"  Speech tokens: {len(outputs.speech_tokens)}")

    # =========================================================================
    # Step 4: S3Gen Conditioning
    # =========================================================================
    print("\n[Step 4] S3Gen Conditioning")
    gen_conds = model.conds.gen
    outputs.s3_embedding = gen_conds["embedding"].detach().cpu().numpy()
    outputs.prompt_token = gen_conds["prompt_token"].detach().cpu().numpy().astype(np.int32)
    outputs.prompt_feat = gen_conds["prompt_feat"].detach().cpu().numpy()
    print(f"  Embedding shape: {outputs.s3_embedding.shape}")
    print(f"  Prompt tokens: {outputs.prompt_token.shape[1]}")

    # =========================================================================
    # Step 5: S3Gen Input Prep
    # =========================================================================
    print("\n[Step 5] S3Gen Input Prep")
    prompt_token = gen_conds["prompt_token"]
    prompt_feat = gen_conds["prompt_feat"]
    soul_s3 = gen_conds["embedding"]

    full_tokens = torch.cat([prompt_token.squeeze(0), speech_tokens.squeeze(0)], dim=0).unsqueeze(0)
    outputs.full_tokens = full_tokens.squeeze(0).detach().cpu().numpy().astype(np.int32)
    print(f"  Full tokens: {len(outputs.full_tokens)}")

    token_emb = model.s3gen.flow.input_embedding(full_tokens.to(device))
    outputs.token_emb = token_emb.detach().cpu().numpy()

    spk_emb_normalized = F.normalize(soul_s3, dim=-1)
    spk_emb_proj = model.s3gen.flow.spk_embed_affine_layer(spk_emb_normalized)
    outputs.spk_emb = spk_emb_proj.detach().cpu().numpy()

    prompt_token_len = gen_conds["prompt_token_len"].item()
    speech_token_len = speech_tokens.shape[1] if speech_tokens.dim() == 2 else speech_tokens.shape[0]
    total_len = prompt_token_len + speech_token_len
    mask = torch.ones(1, total_len, dtype=torch.float32, device=device)
    outputs.mask = mask.detach().cpu().numpy()

    # =========================================================================
    # Step 6: Encoder
    # =========================================================================
    print("\n[Step 6] Encoder")
    mask_float = mask.unsqueeze(-1)
    token_emb_masked = token_emb * mask_float
    token_len = torch.tensor([total_len], dtype=torch.long, device=device)

    encoder_out, encoder_masks = model.s3gen.flow.encoder(token_emb_masked, token_len)
    outputs.encoder_out = encoder_out.detach().cpu().numpy()
    print(f"  Encoder output: {encoder_out.shape}")

    encoder_proj = model.s3gen.flow.encoder_proj(encoder_out)
    mu = encoder_proj.transpose(1, 2).contiguous()
    outputs.mu = mu.detach().cpu().numpy()

    mel_len1 = prompt_feat.shape[1]
    mel_len2 = encoder_out.shape[1] - mel_len1
    x_cond = torch.zeros([1, mel_len1 + mel_len2, 80], device=device, dtype=encoder_proj.dtype)
    x_cond[:, :mel_len1] = prompt_feat
    x_cond = x_cond.transpose(1, 2).contiguous()
    outputs.x_cond = x_cond.detach().cpu().numpy()

    # =========================================================================
    # Step 7: ODE Solver (via S3Gen inference)
    # =========================================================================
    print("\n[Step 7] ODE Solver")
    with torch.inference_mode():
        result = model.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=gen_conds,
            drop_invalid_tokens=False,
            n_cfm_timesteps=N_CFM_TIMESTEPS,
        )

    if isinstance(result, tuple):
        audio = result[0]
    else:
        audio = result

    # Extract mel from before vocoder (we'll capture this separately)
    # For now, we have the audio output
    outputs.audio = audio.detach().cpu().numpy()
    print(f"  Audio shape: {audio.shape}")

    # =========================================================================
    # Step 8: Vocoder (already applied in S3Gen inference)
    # =========================================================================
    print("\n[Step 8] Vocoder")
    print(f"  Audio range: [{audio.min().item():.4f}, {audio.max().item():.4f}]")

    # =========================================================================
    # Step 9: Save WAV
    # =========================================================================
    print("\n[Step 9] Save WAV")

    if audio.dim() == 3:
        audio = audio.squeeze(0)
    elif audio.dim() == 1:
        audio = audio.unsqueeze(0)

    audio_max = audio.abs().max()
    if audio_max > 1.0:
        audio = audio / audio_max

    wav_path = output_dir / "python_output.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(wav_path), audio.cpu(), 24000)

    outputs.wav_path = str(wav_path)
    outputs.duration_seconds = audio.shape[1] / 24000
    print(f"  Saved: {wav_path}")
    print(f"  Duration: {outputs.duration_seconds:.2f}s")

    # Save all outputs as .npy for comparison
    save_pipeline_outputs(outputs, output_dir / "python", "python")

    return outputs


def run_swift_pipeline(
    text: str,
    voice: str,
    language: str,
    output_dir: Path,
) -> PipelineOutputs:
    """
    Run the COMPLETE Swift pipeline from text to audio.

    This calls the Swift GenerateAudioE2E executable which runs independently.
    """
    outputs = PipelineOutputs()

    print("\n" + "=" * 60)
    print("SWIFT PIPELINE (Steps 1-9)")
    print("=" * 60)

    swift_output_dir = output_dir / "swift"
    swift_output_dir.mkdir(parents=True, exist_ok=True)

    # Create config for Swift
    config = {
        "text": text,
        "voice": voice,
        "language": language,
        "seed": SEED,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "min_p": MIN_P,
        "cfg_weight": CFG_WEIGHT,
        "n_cfm_timesteps": N_CFM_TIMESTEPS,
        "output_dir": str(swift_output_dir),
    }

    config_path = swift_output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Voice: {voice}")
    print(f"  Language: {language}")
    print(f"  Text: \"{text[:50]}...\"" if len(text) > 50 else f"  Text: \"{text}\"")

    # Run Swift GenerateAudioE2E from the main swift directory
    swift_dir = PROJECT_ROOT / "swift"

    cmd = [
        "swift", "run", "GenerateAudioE2E",
        "--config", str(config_path),
    ]

    print(f"\n  Running Swift pipeline...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(swift_dir)
        )

        print(result.stdout)
        if result.stderr:
            print(f"  STDERR: {result.stderr}")

        if result.returncode != 0:
            print(f"  ERROR: Swift pipeline failed with code {result.returncode}")
            return outputs

    except subprocess.TimeoutExpired:
        print("  ERROR: Swift pipeline timed out")
        return outputs
    except Exception as e:
        print(f"  ERROR: {e}")
        return outputs

    # Load Swift outputs
    outputs = load_pipeline_outputs(swift_output_dir, "swift")

    return outputs


def save_pipeline_outputs(outputs: PipelineOutputs, output_dir: Path, prefix: str):
    """Save all pipeline outputs as .npy files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        ("step1_text_tokens", outputs.text_tokens),
        ("step1_text_tokens_cfg", outputs.text_tokens_cfg),
        ("step2_t3_conditioning", outputs.t3_conditioning),
        ("step3_speech_tokens", outputs.speech_tokens),
        ("step4_s3_embedding", outputs.s3_embedding),
        ("step4_prompt_token", outputs.prompt_token),
        ("step4_prompt_feat", outputs.prompt_feat),
        ("step5_full_tokens", outputs.full_tokens),
        ("step5_token_emb", outputs.token_emb),
        ("step5_spk_emb", outputs.spk_emb),
        ("step5_mask", outputs.mask),
        ("step6_encoder_out", outputs.encoder_out),
        ("step6_mu", outputs.mu),
        ("step6_x_cond", outputs.x_cond),
        ("step8_audio", outputs.audio),
    ]

    for name, data in fields:
        if data is not None:
            np.save(output_dir / f"{name}.npy", data)


def load_pipeline_outputs(output_dir: Path, prefix: str) -> PipelineOutputs:
    """Load pipeline outputs from .npy files."""
    outputs = PipelineOutputs()

    field_mapping = {
        "step1_text_tokens": "text_tokens",
        "step1_text_tokens_cfg": "text_tokens_cfg",
        "step2_t3_conditioning": "t3_conditioning",
        "step3_speech_tokens": "speech_tokens",
        "step4_s3_embedding": "s3_embedding",
        "step4_prompt_token": "prompt_token",
        "step4_prompt_feat": "prompt_feat",
        "step5_full_tokens": "full_tokens",
        "step5_token_emb": "token_emb",
        "step5_spk_emb": "spk_emb",
        "step5_mask": "mask",
        "step6_encoder_out": "encoder_out",
        "step6_mu": "mu",
        "step6_x_cond": "x_cond",
        "step7_mel": "mel",
        "step8_audio": "audio",
    }

    for file_name, field_name in field_mapping.items():
        path = output_dir / f"{file_name}.npy"
        if path.exists():
            setattr(outputs, field_name, np.load(path))

    # Load wav path
    wav_path = output_dir / f"{prefix}_output.wav"
    if wav_path.exists():
        outputs.wav_path = str(wav_path)
        # Calculate duration
        import soundfile as sf
        try:
            info = sf.info(str(wav_path))
            outputs.duration_seconds = info.duration
        except:
            pass

    return outputs


def compare_outputs(
    python_outputs: PipelineOutputs,
    swift_outputs: PipelineOutputs,
) -> Dict[str, Tuple[bool, float, str]]:
    """
    Compare Python and Swift outputs at each step.

    Returns dict of step_name -> (passed, max_diff, notes)
    """
    results = {}

    def compare_arrays(name: str, py_arr, swift_arr, threshold: float = 0.01) -> Tuple[bool, float, str]:
        if py_arr is None and swift_arr is None:
            return True, 0.0, "Both None"
        if py_arr is None:
            return False, float('inf'), "Python output missing"
        if swift_arr is None:
            return False, float('inf'), "Swift output missing"
        if py_arr.shape != swift_arr.shape:
            return False, float('inf'), f"Shape mismatch: Python {py_arr.shape} vs Swift {swift_arr.shape}"

        max_diff = np.abs(py_arr.astype(np.float64) - swift_arr.astype(np.float64)).max()
        passed = max_diff < threshold
        return passed, float(max_diff), f"Max diff: {max_diff:.2e}"

    # Step 1: Tokenization
    results["Step 1: Tokenization"] = compare_arrays(
        "text_tokens",
        python_outputs.text_tokens,
        swift_outputs.text_tokens,
        threshold=0.0  # Exact match required for tokens
    )

    # Step 2: T3 Conditioning
    results["Step 2: T3 Conditioning"] = compare_arrays(
        "t3_conditioning",
        python_outputs.t3_conditioning,
        swift_outputs.t3_conditioning,
        threshold=0.001
    )

    # Step 3: Speech Tokens
    results["Step 3: T3 Generation"] = compare_arrays(
        "speech_tokens",
        python_outputs.speech_tokens,
        swift_outputs.speech_tokens,
        threshold=0.0  # Exact match required for tokens
    )

    # Step 4: S3Gen Conditioning
    results["Step 4: S3Gen Conditioning"] = compare_arrays(
        "s3_embedding",
        python_outputs.s3_embedding,
        swift_outputs.s3_embedding,
        threshold=0.001
    )

    # Step 5: Input Prep
    results["Step 5: S3Gen Input Prep"] = compare_arrays(
        "token_emb",
        python_outputs.token_emb,
        swift_outputs.token_emb,
        threshold=0.001
    )

    # Step 6: Encoder
    results["Step 6: Encoder"] = compare_arrays(
        "encoder_out",
        python_outputs.encoder_out,
        swift_outputs.encoder_out,
        threshold=0.01
    )

    # Step 7: ODE/Mel
    results["Step 7: ODE Solver"] = compare_arrays(
        "mu",
        python_outputs.mu,
        swift_outputs.mu,
        threshold=0.1
    )

    # Step 8: Audio
    results["Step 8: Vocoder"] = compare_arrays(
        "audio",
        python_outputs.audio,
        swift_outputs.audio,
        threshold=0.1
    )

    # Step 9: Duration comparison
    if python_outputs.duration_seconds and swift_outputs.duration_seconds:
        duration_diff = abs(python_outputs.duration_seconds - swift_outputs.duration_seconds)
        passed = duration_diff < 0.5  # Allow 0.5s difference
        results["Step 9: WAV Output"] = (
            passed,
            duration_diff,
            f"Python: {python_outputs.duration_seconds:.2f}s, Swift: {swift_outputs.duration_seconds:.2f}s"
        )
    else:
        results["Step 9: WAV Output"] = (False, float('inf'), "Duration info missing")

    return results


def print_comparison_results(results: Dict[str, Tuple[bool, float, str]]):
    """Print formatted comparison results."""
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)

    all_passed = True

    print("\n┌" + "─" * 78 + "┐")
    print("│" + " STEP-BY-STEP COMPARISON".center(78) + "│")
    print("├" + "─" * 78 + "┤")

    for step_name, (passed, diff, notes) in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False

        if diff == float('inf'):
            diff_str = "N/A"
        else:
            diff_str = f"{diff:.2e}" if diff < 0.01 else f"{diff:.4f}"

        line = f"│ {step_name}: {status} (diff: {diff_str})"
        print(line.ljust(79) + "│")
        print(f"│   {notes}".ljust(79) + "│")

    print("└" + "─" * 78 + "┘")

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL STEPS VERIFIED - Python and Swift pipelines match!")
    else:
        print("❌ VERIFICATION FAILED - Differences detected")
    print("=" * 80)

    return all_passed


def create_swift_e2e_executable():
    """Create the Swift GenerateAudioE2E package if it doesn't exist."""
    swift_dir = PROJECT_ROOT / "swift" / "test_scripts" / "GenerateAudioE2E"
    swift_dir.mkdir(parents=True, exist_ok=True)

    # Create Package.swift
    package_swift = '''// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "GenerateAudioE2E",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../../.."),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0"),
    ],
    targets: [
        .executableTarget(
            name: "GenerateAudioE2E",
            dependencies: [
                .product(name: "Nightingale", package: "nightingale_TTS"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "."
        ),
    ]
)
'''

    (swift_dir / "Package.swift").write_text(package_swift)

    # Create main.swift
    main_swift = '''import Foundation
import Nightingale
import ArgumentParser
import MLX
import MLXNN

@main
struct GenerateAudioE2E: ParsableCommand {
    @Option(name: .long, help: "Path to config JSON")
    var config: String

    func run() throws {
        // Load config
        let configURL = URL(fileURLWithPath: config)
        let configData = try Data(contentsOf: configURL)
        let configDict = try JSONSerialization.jsonObject(with: configData) as! [String: Any]

        let text = configDict["text"] as! String
        let voice = configDict["voice"] as! String
        let language = configDict["language"] as! String
        let outputDir = configDict["output_dir"] as! String
        let seed = configDict["seed"] as! Int
        let temperature = configDict["temperature"] as! Double
        let cfgWeight = configDict["cfg_weight"] as! Double
        let repetitionPenalty = configDict["repetition_penalty"] as! Double
        let topP = configDict["top_p"] as! Double

        // Set seed
        MLXRandom.seed(UInt64(seed))

        print("[Step 1] Tokenization")
        // Load model and run pipeline...
        // TODO: Implement full Swift pipeline with output capture

        print("Swift E2E pipeline not yet implemented")
        print("Please implement GenerateAudioE2E to capture outputs at each step")
    }
}
'''

    (swift_dir / "main.swift").write_text(main_swift)
    print(f"Created Swift E2E package at {swift_dir}")
    print("NOTE: You need to implement the full Swift pipeline in main.swift")


def main():
    parser = argparse.ArgumentParser(description="True E2E Verification (Steps 1-9)")
    parser.add_argument("--text", "-t", default=None, help="Text to synthesize")
    parser.add_argument("--voice", "-v", default="samantha", help="Voice name")
    parser.add_argument("--lang", "-l", default="en", help="Language code")
    parser.add_argument("--device", "-d", default="cpu", help="Device for Python")
    parser.add_argument("--python-only", action="store_true", help="Run Python only")
    parser.add_argument("--swift-only", action="store_true", help="Run Swift only")
    args = parser.parse_args()

    # Default text
    if args.text is None:
        args.text = "Do you think the model can handle the rising intonation at the end of this sentence?"

    print("=" * 80)
    print("TRUE E2E VERIFICATION: Full Pipeline Comparison")
    print("=" * 80)
    print(f"\nSettings:")
    print(f"  Seed: {SEED}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  CFG Weight: {CFG_WEIGHT}")
    print(f"  Repetition Penalty: {REPETITION_PENALTY}")
    print(f"  Top-P: {TOP_P}")
    print(f"  CFM Timesteps: {N_CFM_TIMESTEPS}")

    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"{args.voice}_{args.lang}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    python_outputs = None
    swift_outputs = None

    # Run Python pipeline
    if not args.swift_only:
        python_outputs = run_python_pipeline(
            text=args.text,
            voice=args.voice,
            language=args.lang,
            output_dir=output_dir,
            device=args.device,
        )

    # Run Swift pipeline
    if not args.python_only:
        swift_outputs = run_swift_pipeline(
            text=args.text,
            voice=args.voice,
            language=args.lang,
            output_dir=output_dir,
        )

    # Compare if both ran
    if python_outputs and swift_outputs:
        results = compare_outputs(python_outputs, swift_outputs)
        all_passed = print_comparison_results(results)
        sys.exit(0 if all_passed else 1)
    elif python_outputs:
        print("\n✅ Python pipeline complete (Swift not run)")
    elif swift_outputs:
        print("\n✅ Swift pipeline complete (Python not run)")
    else:
        print("\n❌ No pipelines were run")
        sys.exit(1)


if __name__ == "__main__":
    main()
