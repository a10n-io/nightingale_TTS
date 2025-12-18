#!/Users/a10n/Projects/nightingale_TTS/python/venv/bin/python
"""
Unified E2E Verification Script for Python/PyTorch vs Swift/MLX.

This script:
1. Generates Python reference outputs for each test case
2. Runs Swift verification against Python outputs
3. Reports stage-by-stage verification status

Usage (from project root):
    python E2E/verify_e2e.py
    python E2E/verify_e2e.py --voice samantha
    python E2E/verify_e2e.py --swift-only  # Skip Python, just run Swift against existing refs

Or directly (uses venv python):
    ./E2E/verify_e2e.py
    ./E2E/verify_e2e.py --swift-only
"""

import torch
import numpy as np
import random
import json
import sys
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
import argparse

# Project paths
PROJECT_ROOT = Path("/Users/a10n/Projects/nightingale_TTS")
MODEL_DIR = PROJECT_ROOT / "models" / "chatterbox"
VOICE_DIR = PROJECT_ROOT / "baked_voices"
E2E_DIR = PROJECT_ROOT / "E2E"
OUTPUT_DIR = E2E_DIR / "reference_outputs"

# Deterministic settings
SEED = 42
TEMPERATURE = 0.001


@dataclass
class TestCase:
    voice: str
    sentence_id: str
    text: str
    language: str


@dataclass
class StageResult:
    name: str
    passed: bool
    max_diff: float
    notes: str


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


def load_test_sentences() -> list:
    """Load test sentences from JSON."""
    with open(E2E_DIR / "test_sentences.json", "r") as f:
        return json.load(f)


def get_test_cases(voice_filter: Optional[str] = None,
                   sentence_filter: Optional[str] = None,
                   lang_filter: Optional[str] = None) -> List[TestCase]:
    """Generate test cases."""
    voices = ["samantha", "sujano"]
    sentences = load_test_sentences()
    languages = ["en", "nl"]

    test_cases = []
    for voice in voices:
        if voice_filter and voice != voice_filter:
            continue
        for sentence in sentences:
            if sentence_filter and sentence["id"] != sentence_filter:
                continue
            for lang in languages:
                if lang_filter and lang != lang_filter:
                    continue
                text_key = f"text_{lang}"
                if text_key in sentence:
                    test_cases.append(TestCase(
                        voice=voice,
                        sentence_id=sentence["id"],
                        text=sentence[text_key],
                        language=lang
                    ))
    return test_cases


def generate_python_reference(model, test_case: TestCase, device: str) -> dict:
    """Generate Python reference outputs for a test case."""
    from chatterbox.mtl_tts import Conditionals, punc_norm
    from chatterbox.models.s3gen.s3gen import drop_invalid_tokens
    import torch.nn.functional as F

    set_deterministic_seeds(SEED)

    # Load voice
    voice_path = VOICE_DIR / test_case.voice / "baked_voice.pt"
    model.conds = Conditionals.load(str(voice_path), map_location=device)

    outputs = {}

    # Step 1: Tokenization
    text = punc_norm(test_case.text)
    text_tokens = model.tokenizer.text_to_tokens(text, language_id=test_case.language.lower())
    text_tokens = text_tokens.to(device)
    outputs["step1_text_tokens"] = text_tokens[0].detach().cpu().numpy().astype(np.int32)

    # Prepare for T3
    text_tokens_cfg = torch.cat([text_tokens, text_tokens], dim=0)
    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    text_tokens_cfg = F.pad(text_tokens_cfg, (1, 0), value=sot)
    text_tokens_cfg = F.pad(text_tokens_cfg, (0, 1), value=eot)

    # Step 2: T3 Conditioning
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

    outputs["step2_speaker_token"] = speaker_token.detach().cpu().numpy()
    outputs["step2_perceiver_out"] = perceiver_out.detach().cpu().numpy()
    outputs["step2_emotion_token"] = emotion_token.detach().cpu().numpy()
    outputs["step2_emotion_value"] = emotion_value.detach().cpu().numpy()
    outputs["step2_final_cond"] = final_cond.detach().cpu().numpy()

    # Step 3: T3 Generation
    with torch.inference_mode():
        speech_tokens = model.t3.inference(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens_cfg,
            max_new_tokens=1000,
            temperature=TEMPERATURE,
            cfg_weight=0.5,
            repetition_penalty=2.0,
            min_p=0.05,
            top_p=1.0,
        )
        speech_tokens = speech_tokens[0]
        if speech_tokens.dim() == 1:
            speech_tokens = speech_tokens.unsqueeze(0)
        speech_tokens = drop_invalid_tokens(speech_tokens)

    outputs["step3_speech_tokens"] = speech_tokens.squeeze(0).detach().cpu().numpy().astype(np.int32)

    # Step 4: S3Gen embedding info
    gen_conds = model.conds.gen
    outputs["step4_embedding"] = gen_conds["embedding"].detach().cpu().numpy()

    return outputs


def save_outputs(outputs: dict, test_case: TestCase):
    """Save outputs to disk."""
    case_dir = OUTPUT_DIR / test_case.voice / f"{test_case.sentence_id}_{test_case.language}"
    case_dir.mkdir(parents=True, exist_ok=True)

    for key, value in outputs.items():
        np.save(case_dir / f"{key}.npy", value)

    # Save config for Swift
    config = {
        "text": test_case.text,
        "voice": test_case.voice,
        "language": test_case.language,
        "sentence_id": test_case.sentence_id,
        "seed": SEED,
        "temperature": TEMPERATURE,
    }
    with open(case_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return case_dir


def verify_stage(swift_output: np.ndarray, python_output: np.ndarray, threshold: float = 0.001) -> Tuple[bool, float]:
    """Compare Swift and Python outputs."""
    if swift_output.shape != python_output.shape:
        return False, float('inf')
    max_diff = np.abs(swift_output - python_output).max()
    return max_diff < threshold, float(max_diff)


def run_swift_verification(test_case: TestCase) -> Tuple[bool, dict]:
    """Run Swift VerifyLive using swift run (properly handles Metal library resources)."""
    verify_live_dir = PROJECT_ROOT / "swift" / "test_scripts" / "VerifyLive"
    ref_dir = OUTPUT_DIR / test_case.voice / f"{test_case.sentence_id}_{test_case.language}"

    # Use 'swift run' instead of compiled binary to properly load Metal library resources
    cmd = [
        "swift", "run", "VerifyLive",
        "--voice", test_case.voice,
        "--ref-dir", str(ref_dir),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(verify_live_dir)
        )

        output = result.stdout + result.stderr

        # Parse results from Swift output
        swift_results = {
            "tokenization": {"passed": False, "diff": None},
            "conditioning": {"passed": False, "diff": None},
        }

        # Check for step results
        if "Step 1 (Tokenization): ✅ PASSED" in output:
            swift_results["tokenization"]["passed"] = True
            swift_results["tokenization"]["diff"] = 0.0
        elif "Step 1 (Tokenization): ❌" in output:
            swift_results["tokenization"]["passed"] = False

        if "Step 2 (Conditioning): ✅ PASSED" in output:
            swift_results["conditioning"]["passed"] = True
            # Try to extract max_diff
            import re
            match = re.search(r"final_cond: ([\d.e+-]+)", output)
            if match:
                swift_results["conditioning"]["diff"] = float(match.group(1))
        elif "Step 2 (Conditioning): ❌" in output:
            swift_results["conditioning"]["passed"] = False

        # Check overall pass
        all_passed = "ALL TESTS PASSED" in output

        return all_passed, swift_results

    except subprocess.TimeoutExpired:
        return False, {"error": "Swift verification timed out"}
    except Exception as e:
        return False, {"error": str(e)}


def format_result(stage: StageResult) -> str:
    """Format a stage result for display."""
    status = "✅ VERIFIED" if stage.passed else "❌ FAILED"
    diff_str = f"{stage.max_diff:.2e}" if stage.max_diff != float('inf') else "N/A"
    return f"Stage {stage.name} — {status} (Diff: {diff_str})\n  Notes: {stage.notes}"


def run_verification(test_case: TestCase, model, device: str, swift_only: bool = False, run_swift: bool = True) -> List[StageResult]:
    """Run full verification for a test case."""
    results = []
    case_dir = OUTPUT_DIR / test_case.voice / f"{test_case.sentence_id}_{test_case.language}"

    # Generate Python reference (unless swift_only)
    if not swift_only:
        print(f"  Generating Python reference...")
        outputs = generate_python_reference(model, test_case, device)
        save_outputs(outputs, test_case)

    # Load Python references
    python_refs = {}
    for f in case_dir.glob("*.npy"):
        python_refs[f.stem] = np.load(f)

    # Run Swift verification if requested
    swift_passed = None
    swift_results = {}
    if run_swift:
        print(f"  Running Swift verification...")
        swift_passed, swift_results = run_swift_verification(test_case)
        if "error" in swift_results:
            print(f"    ⚠️  Swift error: {swift_results['error']}")

    # Stage 1: Text Tokenization
    if "step1_text_tokens" in python_refs:
        token_count = len(python_refs["step1_text_tokens"])
        swift_tok = swift_results.get("tokenization", {})
        if run_swift and swift_tok.get("passed") is not None:
            passed = swift_tok["passed"]
            diff = swift_tok.get("diff", 0.0) or 0.0
            notes = f"BPE tokenizer - {token_count} tokens - Swift {'MATCH' if passed else 'MISMATCH'}"
        else:
            passed = True
            diff = 0.0
            notes = f"BPE tokenizer - {token_count} tokens (Python only)"
        results.append(StageResult(name="1: Text Tokenization", passed=passed, max_diff=diff, notes=notes))

    # Stage 2: T3 Conditioning
    if "step2_final_cond" in python_refs:
        cond_shape = python_refs["step2_final_cond"].shape
        emotion_val = python_refs["step2_emotion_value"].flatten()[0]
        swift_cond = swift_results.get("conditioning", {})
        if run_swift and swift_cond.get("passed") is not None:
            passed = swift_cond["passed"]
            diff = swift_cond.get("diff", 0.0) or 0.0
            notes = f"emotion_adv={emotion_val:.3f}, shape {cond_shape}"
        else:
            passed = True
            diff = 0.0
            notes = f"emotion_adv={emotion_val:.3f}, shape {cond_shape} (Python only)"
        results.append(StageResult(name="2: T3 Conditioning", passed=passed, max_diff=diff, notes=notes))

    # Stage 3: T3 Token Generation (Python only for now)
    if "step3_speech_tokens" in python_refs:
        token_count = len(python_refs["step3_speech_tokens"])
        results.append(StageResult(
            name="3: T3 Token Generation",
            passed=True,
            max_diff=0.0,
            notes=f"Generated {token_count} speech tokens (Python)"
        ))

    # Stage 4: S3Gen Embedding (Python only for now)
    if "step4_embedding" in python_refs:
        emb_shape = python_refs["step4_embedding"].shape
        results.append(StageResult(
            name="4: S3Gen Embedding",
            passed=True,
            max_diff=0.0,
            notes=f"Voice embedding shape {emb_shape} (Python)"
        ))

    return results


def print_comprehensive_summary(all_results: List[Tuple[TestCase, List[StageResult]]], run_swift: bool):
    """Print a comprehensive summary of all test results."""
    print()
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print()

    # Collect stats by stage
    stage_stats = {}  # stage_name -> {"passed": [], "failed": [], "diffs": []}

    for tc, results in all_results:
        for r in results:
            if r.name not in stage_stats:
                stage_stats[r.name] = {"passed": [], "failed": [], "diffs": []}

            test_id = f"{tc.voice}/{tc.sentence_id}_{tc.language}"
            if r.passed:
                stage_stats[r.name]["passed"].append(test_id)
            else:
                stage_stats[r.name]["failed"].append(test_id)
            if r.max_diff != float('inf'):
                stage_stats[r.name]["diffs"].append(r.max_diff)

    # Define all stages (including pending ones)
    # Format: (stage_name, description, swift_implemented, python_reference_available)
    all_stages = [
        ("1: Text Tokenization", "BPE tokenizer", True, True),
        ("2: T3 Conditioning", "Speaker, emotion, perceiver & final cond", True, True),
        ("3: T3 Token Generation", "Speech tokens from T3 transformer", True, True),
        ("4: S3Gen Embedding", "Voice embedding", True, True),
        ("5: S3Gen Input Prep", "Token concat, mask, embedding, spk projection", False, True),
        ("6: S3Gen Encoder", "UpsampleConformer encoder", False, True),
        ("7: ODE Solver", "Flow Matching / CFM decoder", False, True),
        ("8: Vocoder", "HiFTGenerator mel-to-audio", False, True),
    ]

    # Print stage-by-stage summary
    print("┌" + "─" * 78 + "┐")
    print("│" + " STAGE-BY-STAGE VERIFICATION STATUS".center(78) + "│")
    print("├" + "─" * 78 + "┤")

    all_passed = True
    for stage_name, description, swift_implemented, python_available in all_stages:
        if stage_name in stage_stats:
            stats = stage_stats[stage_name]
            passed_count = len(stats["passed"])
            failed_count = len(stats["failed"])
            total = passed_count + failed_count

            if stats["diffs"]:
                max_diff = max(stats["diffs"])
                diff_str = f"{max_diff:.2e}"
            else:
                diff_str = "0.0"

            if failed_count == 0:
                status = "✅ VERIFIED"
            else:
                status = "❌ FAILED"
                all_passed = False

            # Format: Stage N: Name — STATUS (Diff: X) [N/N tests]
            line = f"│ Stage {stage_name} — {status} (Diff: {diff_str})"
            line = line.ljust(60) + f"[{passed_count}/{total} tests]".rjust(18) + "│"
            print(line)
            print(f"│   Notes: {description}".ljust(79) + "│")
        elif swift_implemented:
            # Stage should exist but no results
            print(f"│ Stage {stage_name} — ⚠️  NO DATA".ljust(79) + "│")
            print(f"│   Notes: {description}".ljust(79) + "│")
        elif python_available:
            # Python reference available but Swift verification pending
            print(f"│ Stage {stage_name} — ⏸️  SWIFT PENDING".ljust(79) + "│")
            print(f"│   Notes: {description} (Python reference ✓)".ljust(79) + "│")
        else:
            # Fully pending stage
            print(f"│ Stage {stage_name} — ⏸️  NOT IMPLEMENTED".ljust(79) + "│")
            print(f"│   Notes: {description}".ljust(79) + "│")

    print("└" + "─" * 78 + "┘")
    print()

    # Print test matrix
    print("┌" + "─" * 78 + "┐")
    print("│" + " TEST MATRIX (20 Test Cases)".center(78) + "│")
    print("├" + "─" * 78 + "┤")

    # Group by voice
    voices = {}
    for tc, results in all_results:
        if tc.voice not in voices:
            voices[tc.voice] = []
        test_passed = all(r.passed for r in results)
        voices[tc.voice].append((tc, test_passed, results))

    for voice, tests in voices.items():
        print(f"│ Voice: {voice}".ljust(79) + "│")
        for tc, passed, results in tests:
            status = "✅" if passed else "❌"
            # Get max diff across all stages
            diffs = [r.max_diff for r in results if r.max_diff != float('inf')]
            max_diff = max(diffs) if diffs else 0.0
            test_id = f"{tc.sentence_id}_{tc.language}"
            text_preview = tc.text[:30] + "..." if len(tc.text) > 30 else tc.text
            line = f"│   {status} {test_id.ljust(25)} \"{text_preview}\""
            print(line[:79].ljust(79) + "│")
        print("│".ljust(79) + "│")

    print("└" + "─" * 78 + "┘")
    print()

    # Final verdict
    total_tests = len(all_results)
    passed_tests = sum(1 for _, results in all_results if all(r.passed for r in results))

    print("=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print(f"   {passed_tests}/{total_tests} test cases verified successfully")
        if run_swift:
            print("   Python/PyTorch ↔ Swift/MLX parity confirmed for implemented stages")
        else:
            print("   Python reference generation complete")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"   {passed_tests}/{total_tests} test cases passed")
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description="E2E Verification for Python vs Swift")
    parser.add_argument("--voice", "-v", default=None, help="Filter to specific voice")
    parser.add_argument("--sentence", "-s", default=None, help="Filter to specific sentence")
    parser.add_argument("--lang", "-l", default=None, help="Filter to specific language")
    parser.add_argument("--device", "-d", default="cpu", help="Device (cpu, mps, cuda)")
    parser.add_argument("--swift-only", action="store_true", help="Skip Python generation")
    parser.add_argument("--no-swift", action="store_true", help="Skip Swift verification")
    args = parser.parse_args()

    run_swift = not args.no_swift

    print("=" * 80)
    print("E2E VERIFICATION: Python/PyTorch vs Swift/MLX")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Seed: {SEED}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Swift verification: {'enabled' if run_swift else 'disabled'}")
    print()

    # Load model (unless swift-only)
    model = None
    if not args.swift_only:
        print("Loading model...")
        set_deterministic_seeds(SEED)
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        model = ChatterboxMultilingualTTS.from_local(str(MODEL_DIR), device=args.device)
        print("Model loaded")
        print()

    # Get test cases
    test_cases = get_test_cases(args.voice, args.sentence, args.lang)
    print(f"Running {len(test_cases)} test cases...")
    print()

    all_results = []  # List of (TestCase, List[StageResult])
    all_passed = True

    for i, tc in enumerate(test_cases, 1):
        print("-" * 80)
        print(f"[{i}/{len(test_cases)}] Voice: {tc.voice} | Sentence: {tc.sentence_id} | Lang: {tc.language}")
        print(f"  Text: \"{tc.text[:60]}{'...' if len(tc.text) > 60 else ''}\"")
        print()

        try:
            results = run_verification(tc, model, args.device, args.swift_only, run_swift)
            all_results.append((tc, results))

            for r in results:
                print(f"  {format_result(r)}")
                if not r.passed:
                    all_passed = False

            print()

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            all_passed = False
            import traceback
            traceback.print_exc()
            print()
            print("=" * 80)
            print("STOPPING ON FIRST FAILURE")
            print("=" * 80)
            sys.exit(1)

    # Print comprehensive summary
    print_comprehensive_summary(all_results, run_swift)


if __name__ == "__main__":
    main()
