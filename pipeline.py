"""
LLM-as-a-Judge — Iterative Prompt Refinement (POC)
===================================================
Paper implementation: textual gradient descent for judge prompt optimization.

Data splits (from 1000 balanced examples, 200 per class):
  - Gradient pool    : 500 examples (100/class)  — scored each iteration to find failures
  - Verification set : 200 examples  (40/class)  — accepts/rejects refined prompt
  - Test set         : 300 examples  (60/class)  — final eval only, never seen during training

How to run:
    pip install requests datasets scipy tqdm python-dotenv
    Set environment variables in .env (see .env.example), then:
    python pipeline.py
"""

import json
import math
import os
import random
import re
from datetime import datetime
from pathlib import Path

import requests
from datasets import load_dataset
from dotenv import load_dotenv
from scipy import stats
from tqdm import tqdm

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG  (edit here)
# ─────────────────────────────────────────────
SEED               = 42
SAMPLES_PER_CLASS  = 200   # 200 × 5 classes = 1000 total examples
TRAIN_PER_CLASS    = 140   # first 140 used for verification (40) + gradient pool (100)
VERIFY_PER_CLASS   = 40    # 40 × 5 = 200 verification examples
SUBSET_PER_ITER    = 100   # examples sampled from gradient pool each iteration
NUM_ITERATIONS     = 10
MAX_FAILED_SHOWN   = 50    # top-N worst failures sent to refiner

# Model settings — read from environment variables set in .env
MODEL    = os.environ["DEEPSEEK_MODEL_NAME"]
API_URL  = os.environ["DEEPSEEK_BASE_URL"] 
API_KEY  = os.environ["DEEPSEEK_API_KEY"]

JUDGE_TEMPERATURE   = 0.0
REFINER_TEMPERATURE = 0.8
MAX_TOKENS_JUDGE    = 512
MAX_TOKENS_REFINER  = 3000

JUDGE_PROMPT_PATH   = "prompts/judge.txt"
REFINER_PROMPT_PATH = "prompts/refiner.txt"
RESULTS_DIR         = "results"
# ─────────────────────────────────────────────


# ── 1. MODEL CALL ─────────────────────────────────────────────────────────────
def call_model(prompt_text: str, temperature: float, max_tokens: int) -> str:
    """Send a prompt and return the model's text response."""

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ── 2. DATA ────────────────────────────────────────────────────────────────────

def load_and_split(seed: int):
    """
    Load HelpSteer2, sample 200 examples per class (1000 total), then split:
      gradient_pool    — 100/class × 5 = 500  (find failures for refiner)
      verification_set —  40/class × 5 = 200  (accept/reject candidate prompts)
      test_set         —  60/class × 5 = 300  (final evaluation only)
    """
    print("Loading nvidia/HelpSteer2 ...")
    dataset = load_dataset("nvidia/HelpSteer2", split="train")

    # Group by helpfulness label (0–4)
    by_label = {i: [] for i in range(5)}
    for ex in dataset:
        label = ex.get("helpfulness")
        if label in by_label:
            by_label[label].append(ex)

    rng = random.Random(seed)
    gradient_pool, verification_set, test_set = [], [], []

    for label, examples in by_label.items():
        sampled = rng.sample(examples, SAMPLES_PER_CLASS)
        # normalize score: 0→0.0, 1→0.25, 2→0.5, 3→0.75, 4→1.0
        items = [{"prompt": e["prompt"], "response": e["response"],
                  "label": label, "score": label / 4} for e in sampled]
        verification_set.extend(items[:VERIFY_PER_CLASS])
        gradient_pool.extend(items[VERIFY_PER_CLASS:TRAIN_PER_CLASS])
        test_set.extend(items[TRAIN_PER_CLASS:])

    print(f"  gradient_pool={len(gradient_pool)}, "
          f"verification={len(verification_set)}, test={len(test_set)}")
    return gradient_pool, verification_set, test_set


def sample_subset(pool, size, seed, iteration):
    """Sample `size` examples from pool. Changes each iteration for variety."""
    rng = random.Random(seed + iteration)
    return rng.sample(pool, min(size, len(pool)))


# ── 3. JUDGE ───────────────────────────────────────────────────────────────────

VALID_SCORES = {0.0, 0.25, 0.5, 0.75, 1.0}
JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
SCORE_RE   = re.compile(r"SCORE:\s*([\d.]+)")


def judge_example(judge_prompt: str, example: dict) -> dict:
    """Run judge on one example. Returns result dict with predicted score."""
    filled = judge_prompt.replace("{USER_PROMPT}", example["prompt"]) \
                         .replace("{MODEL_RESPONSE}", example["response"])
    try:
        raw = call_model(filled, JUDGE_TEMPERATURE, MAX_TOKENS_JUDGE)
    except Exception as e:
        raw = f"ERROR: {e}"

    score = None
    reasoning = ""

    # Try JSON format first, then fallback to SCORE: text format
    text = (JSON_FENCE.search(raw) or type("", (), {"group": lambda s, n: raw})()).group(1) if JSON_FENCE.search(raw) else raw.strip()
    try:
        data = json.loads(text)
        v = float(data.get("score", -1))
        if v in VALID_SCORES:
            score = v
        reasoning = str(data.get("reason", data.get("reasoning", ""))).strip()
    except Exception:
        m = SCORE_RE.search(raw)
        if m:
            try:
                v = float(m.group(1))
                if v in VALID_SCORES:
                    score = v
            except ValueError:
                pass

    return {
        "prompt": example["prompt"],
        "response": example["response"],
        "true_score": example["score"],
        "predicted_score": score,
        "reasoning": reasoning,
        "abs_error": abs(score - example["score"]) if score is not None else None,
    }


def judge_batch(judge_prompt: str, examples: list) -> list:
    """Judge all examples with a progress bar."""
    results = []
    for ex in tqdm(examples, desc="Judging", unit="ex"):
        results.append(judge_example(judge_prompt, ex))
    parse_errors = sum(1 for r in results if r["abs_error"] is None)
    if parse_errors:
        print(f"  Warning: {parse_errors} parse errors")
    return results


# ── 4. METRICS ─────────────────────────────────────────────────────────────────

def compute_metrics(results: list) -> dict:
    """Compute MAE, RMSE, and Pearson r from judge results."""
    valid = [r for r in results if r["abs_error"] is not None]
    if not valid:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson_r": 0.0}
    preds = [r["predicted_score"] for r in valid]
    trues = [r["true_score"] for r in valid]
    mae  = sum(abs(p - t) for p, t in zip(preds, trues)) / len(valid)
    rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(preds, trues)) / len(valid))
    r, _ = (stats.pearsonr(preds, trues) if len(set(preds)) > 1 and len(set(trues)) > 1
            else (0.0, 1.0))
    return {"mae": mae, "rmse": rmse, "pearson_r": float(r), "n": len(valid)}


def build_failure_summary(results: list) -> str:
    """Statistical summary of all failures — sent to refiner as gradient signal."""
    valid  = [r for r in results if r["abs_error"] is not None]
    failed = [r for r in valid if r["abs_error"] > 0]
    if not failed:
        return "No failures — all predictions correct."

    over  = sum(1 for r in failed if r["predicted_score"] > r["true_score"])
    under = sum(1 for r in failed if r["predicted_score"] < r["true_score"])

    buckets = {}
    for r in failed:
        b = round(round(r["abs_error"] * 4) / 4, 2)
        buckets[b] = buckets.get(b, 0) + 1

    patterns = {}
    for r in failed:
        k = f"predicted={r['predicted_score']} → true={r['true_score']}"
        patterns[k] = patterns.get(k, 0) + 1
    top_patterns = sorted(patterns.items(), key=lambda x: -x[1])[:10]

    lines = [
        f"FAILURES: {len(failed)}/{len(valid)} ({100*len(failed)/len(valid):.1f}%)",
        f"Direction: over-rating={over}, under-rating={under}",
        "",
        "Error magnitude distribution:",
    ]
    for b in sorted(buckets): lines.append(f"  |error|={b:.2f}: {buckets[b]} cases")
    lines += ["", "Top error patterns (predicted → true):"]
    for pattern, count in top_patterns: lines.append(f"  {pattern}: {count} cases")

    return "\n".join(lines)


def format_examples(examples: list, label: str) -> str:
    """Format a list of examples for the refiner prompt."""
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"### {label} {i} (error={ex['abs_error']:.2f})")
        lines.append(f"True: {ex['true_score']:.2f} | Predicted: {ex['predicted_score']}")
        lines.append(f"Prompt: {ex['prompt'][:300]}")
        lines.append(f"Response: {ex['response'][:400]}")
        lines.append(f"Reasoning: {ex.get('reasoning','')[:200]}")
        lines.append("")
    return "\n".join(lines)


# ── 5. REFINER ─────────────────────────────────────────────────────────────────

REFINED_TAG = re.compile(r"<REFINED_PROMPT>(.*?)</REFINED_PROMPT>", re.DOTALL)


def run_refiner(refiner_template: str, current_prompt: str,
                results: list, iteration: int, mae: float, pearson_r: float) -> str:
    """
    Call meta-refiner to produce an improved judge prompt.
    Returns refined prompt string, or current_prompt if extraction fails.
    """
    failed_sorted = sorted(
        [r for r in results if r["abs_error"] is not None and r["abs_error"] > 0],
        key=lambda x: -x["abs_error"]
    )[:MAX_FAILED_SHOWN]

    message = (refiner_template
        .replace("{CURRENT_JUDGE_PROMPT}", current_prompt)
        .replace("{FAILURE_SUMMARY}",      build_failure_summary(results))
        .replace("{FAILING_EXAMPLES}",     format_examples(failed_sorted, "Failed Example"))
        .replace("{NUM_FAILED}",           str(len(failed_sorted)))
        .replace("{MAE}",                  f"{mae:.4f}")
        .replace("{PEARSON_R}",            f"{pearson_r:.4f}")
        .replace("{ITERATION}",            str(iteration))
    )

    print(f"  Running refiner (iter {iteration}, {len(failed_sorted)} failures)...")
    try:
        output = call_model(message, REFINER_TEMPERATURE, MAX_TOKENS_REFINER)
    except Exception as e:
        print(f"  Refiner failed: {e}")
        return current_prompt

    match = REFINED_TAG.search(output)
    if not match:
        print("  Could not extract <REFINED_PROMPT> tag — keeping current prompt.")
        return current_prompt

    refined = match.group(1).strip()
    if "{USER_PROMPT}" not in refined or "{MODEL_RESPONSE}" not in refined:
        print("  Refined prompt missing placeholders — keeping current prompt.")
        return current_prompt

    return refined


# ── 6. MAIN LOOP ───────────────────────────────────────────────────────────────

def main():
    # Setup
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    run_id  = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(RESULTS_DIR) / run_id
    out_dir.mkdir()

    judge_prompt    = Path(JUDGE_PROMPT_PATH).read_text()
    refiner_template = Path(REFINER_PROMPT_PATH).read_text()
    (out_dir / "prompt_iter_0_base.txt").write_text(judge_prompt)

    # Load data
    gradient_pool, verification_set, test_set = load_and_split(SEED)

    best_prompt  = judge_prompt
    best_pearson = -1.0
    best_iter    = 0
    log          = []

    print(f"\nRun: {run_id}")
    print(f"Gradient pool: {len(gradient_pool)} | Verification: {len(verification_set)} | Test: {len(test_set)}\n")

    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'='*55}")
        print(f"[Iteration {iteration + 1}/{NUM_ITERATIONS}]")
        print(f"{'='*55}")

        # Score gradient subset
        subset  = sample_subset(gradient_pool, SUBSET_PER_ITER, SEED, iteration)
        results = judge_batch(judge_prompt, subset)
        m       = compute_metrics(results)
        print(f"  Gradient pool → MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, r={m['pearson_r']:.4f}")

        # Track best by Pearson r
        if m["pearson_r"] > best_pearson:
            best_pearson = m["pearson_r"]
            best_prompt  = judge_prompt
            best_iter    = iteration + 1

        entry = {"iteration": iteration + 1, **m, "accepted": None}

        # Refine (skip on last iteration)
        if iteration < NUM_ITERATIONS - 1:
            candidate = run_refiner(
                refiner_template, judge_prompt, results,
                iteration, m["mae"], m["pearson_r"]
            )

            if candidate != judge_prompt:
                # Evaluate candidate on held-out verification set
                print(f"  Evaluating candidate on verification set ({len(verification_set)} examples)...")
                v_results = judge_batch(candidate, verification_set)
                vm = compute_metrics(v_results)

                # Also score current prompt on verification set for fair comparison
                print(f"  Evaluating current prompt on verification set...")
                c_results = judge_batch(judge_prompt, verification_set)
                cm = compute_metrics(c_results)

                accepted = vm["mae"] < cm["mae"]
                status   = "ACCEPTED" if accepted else "REJECTED"
                print(f"  Candidate:  MAE={vm['mae']:.4f}, r={vm['pearson_r']:.4f} [{status}]")
                print(f"  Current:    MAE={cm['mae']:.4f}, r={cm['pearson_r']:.4f}")

                if accepted:
                    judge_prompt = candidate

                entry["accepted"] = accepted
                suffix = "accepted" if accepted else "rejected"
                (out_dir / f"prompt_iter_{iteration + 1}_{suffix}.txt").write_text(candidate)
            else:
                print("  Refiner returned unchanged prompt.")
                entry["accepted"] = False

        log.append(entry)

    # ── Final evaluation on unseen test set ────────────────────────────────────
    print(f"\n{'#'*55}")
    print("FINAL EVALUATION on unseen test set")
    print(f"{'#'*55}")
    test_results = judge_batch(best_prompt, test_set)
    test_m = compute_metrics(test_results)
    print(f"  Test set → MAE={test_m['mae']:.4f}, RMSE={test_m['rmse']:.4f}, r={test_m['pearson_r']:.4f}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\nBest iteration: {best_iter}  (Pearson r={best_pearson:.4f})")
    print("\nProgression:")
    for e in log:
        acc = e.get("accepted")
        tag = "[ACCEPTED]" if acc is True else "[REJECTED]" if acc is False else "[last]"
        mark = " ← best" if e["iteration"] == best_iter else ""
        print(f"  Iter {e['iteration']:>2}: MAE={e['mae']:.4f}, r={e['pearson_r']:.4f}  {tag}{mark}")

    # Save results
    (out_dir / "prompt_best.txt").write_text(best_prompt)
    out_path = Path(RESULTS_DIR) / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump({
            "run_id": run_id,
            "config": {
                "model": MODEL, "seed": SEED,
                "gradient_pool": len(gradient_pool),
                "verification_set": len(verification_set),
                "test_set": len(test_set),
                "subset_per_iter": SUBSET_PER_ITER,
                "num_iterations": NUM_ITERATIONS,
            },
            "iterations": log,
            "best_iteration": best_iter,
            "best_pearson_r": best_pearson,
            "test_evaluation": test_m,
        }, f, indent=2)

    print(f"\nResults saved → {out_path}")
    print(f"Prompts saved → {out_dir}/")


if __name__ == "__main__":
    main()
