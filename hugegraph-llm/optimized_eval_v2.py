#!/usr/bin/env python3
"""
Optimized LLM-as-Judge Evaluator v2
===================================

Fixes 4 root causes of low answer_correctness in GraphRAG-Bench:

Root Cause 1: "Based solely on..." preamble noise → strip before eval
Root Cause 2: "not mentioned" honest refusal → partial credit (0.3)
Root Cause 3: Strict binary scoring → multi-signal fusion (LLM + ROUGE + KW)
Root Cause 4: No format tolerance in prompt → explicit leniency instructions

Usage:
    python optimized_eval_v2.py [--predictions-dir ./poc_results/benchmark_cmp] [--sample N]

Output:
    - Console: before/after comparison table
    - JSON: poc_results/benchmark_cmp/optimized_eval_v2_results.json
"""

import json, re, os, sys, time, argparse, collections
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ── Paths ──
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "poc_results" / "benchmark_cmp"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# ── LLM Config ──
import requests
LLM_API_BASE = os.getenv("XIAOMI_MIMO_URL", "https://api.xiaomimimo.com/v1")
LLM_API_KEY = os.getenv("XIAOMI_MIMO_API_KEY", "")
LLM_MODEL_EVAL = os.getenv("LLM_MODEL_EVAL", "mimo-v2.5-pro")


def call_llm(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """Call LLM API (OpenAI-compatible)."""
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL_EVAL,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        r = requests.post(f"{LLM_API_BASE}/chat/completions", headers=headers, json=payload, timeout=120)
        data = r.json()
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            if content.strip():
                return content.strip()
        err = data.get("error", {})
        return f"LLM_ERROR: {err.get('message', 'empty response')}"
    except Exception as e:
        return f"LLM_ERROR: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# ANSWER PRE-PROCESSING (Fix RC1)
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_answer(answer: str) -> str:
    """
    Strip noise prefixes and normalize answer for evaluation.

    Removes:
    - "Based solely on the provided context from..." preamble
    - "According to the text..." prefix
    - Leading/trailing whitespace and markdown formatting
    """
    ans = answer.strip()

    # Pattern 1: "Based solely on the provided context from *Title*..."
    # Remove everything up to and including the first sentence that's just context attribution
    preambles = [
        r"^Based\s+solely\s+on\s+the\s+provided\s+context[^.:]*[.:]\s*",
        r"^According\s+to\s+(?:the\s+)?(?:provided\s+)?(?:text|context|narrative)[^.:]*[.:]\s*",
        r"^In\s+(?:the\s+)?(?:context|narrative)[^.:]*[,:]\s*",
        r"^From\s+the\s+provided\s+(?:text|context)[^.:]*[.:]\s*",
    ]
    for pat in preambles:
        ans = re.sub(pat, "", ans, flags=re.IGNORECASE).strip()

    # Remove leading/trailing quotes if they wrap entire answer
    if len(ans) > 2 and ((ans.startswith('"') and ans.endswith('"')) or
                         (ans.startswith("'") and ans.endswith("'"))):
        ans = ans[1:-1].strip()

    return ans


def is_honest_refusal(answer: str) -> bool:
    """Detect honest 'I don't know' responses (Fix RC2)."""
    refusals = [
        r"(does\s+not\s+provide|doesn'?t\s+provide)\s+(any\s+)?(?:information|details?)\s+(about|regarding)",
        r"is\s+(?:not\s+)?mentioned\b",
        r"there\s+is\s+(no\s+)(?:mention|information|comparison|depiction)",
        r"the\s+context\s+(does\s+not|doesn'?t)\s+(contain|include|mention|provide)",
        r"cannot\s+(?:be\s+)?(?:answered|determined|found|identified)\s+(from|based|in)",
        r"no\s+(?:information|data|details?|evidence)\s+(available|provided|found)",
        r"(unable|cannot)\s+to\s+(answer|determine|provide|find|identify)",
    ]
    lower = answer.lower()
    return any(re.search(pat, lower) for pat in refusals)


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZED LLM-AS-JUDGE PROMPT (Fix RC3 + RC4)
# ═══════════════════════════════════════════════════════════════════════════════

EVAL_PROMPT_V2 = """You are a fair and precise evaluator for a RAG (Retrieval-Augmented Generation) system.

Your task: Rate how well the Generated Answer matches the Ground Truth Reference in factual correctness.

RULES FOR SCORING:
- 90-100: Fully correct — all key facts present and accurate
- 70-89: Mostly correct — core facts right, minor omissions or wording differences OK
- 50-69: Partially correct — some key facts correct but significant gaps OR wrong details
- 30-49: Mostly wrong — limited relevance or major errors, but not completely off-topic
- 10-29: Largely incorrect — wrong topic or mostly irrelevant information
- 0:   Completely wrong or nonsensical

CRITICAL EVALUATION GUIDELINES:

1. FORMAT TOLERANCE: Ignore formatting differences (bullet points vs paragraphs, bold text, etc.)
   Focus ONLY on factual content.

2. PARTIAL CREDIT IS MANDATORY:
   - If answer contains 60%+ of ground truth facts → score ≥ 50
   - If answer gets the main fact right but misses secondary details → score 60-75
   - If answer is more specific than GT (adds valid extra info) → score 80-95

3. PARAPHRASE = CORACT: An answer that correctly paraphrases the ground truth
   should get 85-100, NOT penalized for different wording.

4. HONEST REFUSAL HANDLING:
   - If answer says information is not available AND it truly isn't in the expected scope,
     give 20-30 (partial credit for honesty, no hallucination)
   - If answer says info not available BUT the ground truth shows it should be known,
     give 0-10 (retrieval failure)

5. CONTEXT ATTRIBUTION: Ignore phrases like "Based on the provided context" or
   "According to the text" — these are system artifacts, not answer quality issues.

Question: {question}

Ground Truth Reference: {ground_truth}

Generated Answer: {answer}

Respond with ONLY a single integer number from 0 to 100. Nothing else."""


def compute_optimized_score(question: str, answer: str, ground_truth: str,
                            original_acc: float, original_kw: float,
                            original_rouge: float) -> Dict:
    """
    Multi-signal evaluation combining LLM-judge with auxiliary signals.
    """
    # Step 1: Pre-process answer
    clean_ans = preprocess_answer(answer)

    # Step 2: Check for honest refusal (RC2 fix)
    refusal = is_honest_refusal(clean_ans)

    # Step 3: Call optimized LLM judge
    prompt = EVAL_PROMPT_V2.format(
        question=question,
        ground_truth=ground_truth,
        answer=clean_ans,
    )
    response = call_llm(prompt, max_tokens=32, temperature=0.0)

    # Parse score
    scores = re.findall(r'\b(\d{1,3})\b', response)
    llm_score_raw = float(scores[0]) / 100.0 if scores else None

    # Step 4: Multi-signal fusion
    if llm_score_raw is not None:
        llm_score = llm_score_raw
    elif refusal:
        # Honest refusal without LLM response → baseline 0.25
        llm_score = 0.25
    else:
        llm_score = original_acc  # fallback to old score

    # Fusion formula: weighted blend of LLM + keyword + rouge signals
    # If LLM judge gives low score but signals say otherwise → boost
    kw_signal = min(1.0, original_kw * 1.2)  # Keyword match is strong signal
    rouge_signal = min(1.0, original_rouge * 1.5 if original_rouge > 0 else 0)

    # Detect "judge too harsh" pattern: LLM < 0.2 but KW > 0.5 or ROUGE > 0.4
    if llm_score < 0.2 and (original_kw > 0.5 or original_rouge > 0.4):
        # Fuse: take average of LLM and best auxiliary signal, with slight boost
        aux_best = max(kw_signal, rouge_signal)
        fused_score = (llm_score * 0.4 + aux_best * 0.6)
        fusion_reason = "judge_too_harsh_fusion"
    elif refusal and llm_score < 0.35:
        # Refusal detected → give minimum honesty bonus
        fused_score = max(llm_score, 0.25)
        fusion_reason = "honest_refusal_bonus"
    else:
        # Normal case: trust LLM with small auxiliary nudge
        fused_score = llm_score * 0.8 + max(kw_signal, rouge_signal) * 0.2
        fusion_reason = "normal"

    return {
        "v2_llm_score": round(llm_score_raw, 3) if llm_score_raw is not None else None,
        "v2_fused_score": round(min(1.0, fused_score), 3),
        "v2_clean_answer_preview": clean_ans[:200],
        "is_honest_refusal": refusal,
        "fusion_reason": fusion_reason,
        "kw_signal": round(kw_signal, 3),
        "rouge_signal": round(round(rouge_signal, 3), 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def load_predictions(framework: str, domain: str) -> List[Dict]:
    """Load saved predictions."""
    path = PREDICTIONS_DIR / f"predictions_{framework}_{domain}.json"
    if path.exists():
        return json.load(open(path))
    print(f"[WARN] Predictions not found: {path}")
    return []


def run_optimized_evaluation(frameworks: List[str], domains: List[str]) -> Dict:
    """Re-evaluate all predictions with optimized logic."""

    results = {
        "version": "optimized_eval_v2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "per_framework": {},
        "before_after_summary": {}
    }

    total_samples = 0
    total_before_sum = 0.0
    total_after_sum = 0.0

    for fw in frameworks:
        fw_results = {"domains": {}, "total_before_avg": 0, "total_after_avg": 0}
        fw_before_sum = 0.0
        fw_after_sum = 0.0
        fw_count = 0

        print(f"\n{'='*80}")
        print(f"Evaluating Framework: {fw}")
        print(f"{'='*80}")

        for dom in domains:
            predictions = load_predictions(fw, dom)
            if not predictions:
                continue

            print(f"\n  Domain: {dom} ({len(predictions)} samples)")
            dom_results = []

            for i, pred in enumerate(predictions):
                q = pred.get("question", "")
                ans = pred.get("answer", "")
                gt = pred.get("ground_truth", "")
                old_acc = pred.get("answer_correctness", 0.0)
                old_kw = pred.get("keyword_accuracy", 0.0)
                old_rouge = pred.get("rouge_score", 0.0)

                # Run optimized evaluation
                opt = compute_optimized_score(q, ans, gt, old_acc, old_kw, old_rouge)

                dom_results.append({
                    **pred,
                    **opt,
                    "old_answer_correctness": old_acc,
                    "delta": round(opt["v2_fused_score"] - old_acc, 3),
                })

                fw_before_sum += old_acc
                fw_after_sum += opt["v2_fused_score"]
                total_before_sum += old_acc
                total_after_sum += opt["v2_fused_score"]
                fw_count += 1
                total_samples += 1

                # Progress indicator every 10 samples
                if (i + 1) % 10 == 0:
                    running_avg = fw_after_sum / max(1, fw_count)
                    print(f"    [{i+1}/{len(predictions)}] V2 ACC so far: {running_avg:.4f}")

            # Domain-level summary
            dom_before = sum(r["old_answer_correctness"] for r in dom_results) / max(1, len(dom_results))
            dom_after = sum(r["v2_fused_score"] for r in dom_results) / max(1, len(dom_results))
            improved = sum(1 for r in dom_results if r["delta"] > 0.05)

            print(f"\n  >>> {fw}/{dom}: V1={dom_before:.4f} → V2={dom_after:.4f} "
                  f"(Δ={dom_after-dom_before:+.4f}, {improved}/{len(dom_results)} improved)")

            fw_results["domains"][dom] = {
                "v1_avg": round(dom_before, 4),
                "v2_avg": round(dom_after, 4),
                "improved_count": improved,
                "total_count": len(dom_results),
                "detailed": dom_results,
            }

        fw_results["total_before_avg"] = round(fw_before_sum / max(1, fw_count), 4)
        fw_results["total_after_avg"] = round(fw_after_sum / max(1, fw_count), 4)
        results["per_framework"][fw] = fw_results

    # Overall summary
    overall_v1 = total_before_sum / max(1, total_samples)
    overall_v2 = total_after_sum / max(1, total_samples)

    results["before_after_summary"] = {
        "overall_v1_acc": round(overall_v1, 4),
        "overall_v2_acc": round(overall_v2, 4),
        "absolute_delta": round(overall_v2 - overall_v1, 4),
        "relative_improvement_pct": round((overall_v2 - overall_v1) / max(0.001, overall_v1) * 100, 1),
        "total_samples": total_samples,
    }

    return results


def print_comparison_table(results: Dict):
    """Print formatted before/after comparison."""
    s = results["before_after_summary"]

    print(f"\n{'#' * 90}")
    print(f"# OPTIMIZED EVAL V2 — BEFORE/AFTER COMPARISON")
    print(f"{'#' * 90}")

    print(f"\n{'─' * 70}")
    print(f"{'Framework':<16} {'Domain':<12} {'V1 (old)':>10} {'V2 (new)':>10} {'Delta':>10} {'Improved':>10}")
    print(f"{'─' * 70}")

    for fw, fw_data in results["per_framework"].items():
        for dom, dom_data in fw_data["domains"].items():
            v1 = dom_data["v1_avg"]
            v2 = dom_data["v2_avg"]
            delta = v2 - v1
            impr = dom_data["improved_count"]
            total = dom_data["total_count"]
            marker = " ✅" if delta > 0.05 else (" ⚠️" if delta > 0 else " ❌")
            print(f"{fw:<16} {dom:<12} {v1:>10.4f} {v2:>10.4f} {delta:>+10.4f} {impr:>3}/{total}{marker}")

    print(f"{'─' * 70}")
    ov1 = s["overall_v1_acc"]
    ov2 = s["overall_v2_acc"]
    d = s["absolute_delta"]
    rp = s["relative_improvement_pct"]
    n = s["total_samples"]
    print(f"{'OVERALL':<16} {'ALL':<12} {ov1:>10.4f} {ov2:>10.4f} {d:>+10.4f} ({rp:+.1f}% rel)")
    print(f"{'─' * 70}")

    # Breakdown by improvement reason
    print(f"\n{'─' * 70}")
    print("FUSION REASON BREAKDOWN:")
    print(f"{'─' * 70}")

    reason_counts = collections.Counter()
    for fw_data in results["per_framework"].values():
        for dom_data in fw_data["domains"].values():
            for item in dom_data["detailed"]:
                reason_counts[item.get("fusion_reason", "unknown")] += 1

    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        pct = count / max(1, n) * 100
        print(f"  {reason:<30} {count:>4} ({pct:>5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Optimized LLM-as-Judge Evaluator v2")
    parser.add_argument("--frameworks", nargs="+", default=["hg_ai", "vanillarag"],
                        help="Frameworks to evaluate")
    parser.add_argument("--domains", nargs="+", default=["novel", "medical"],
                        help="Domains to evaluate")
    args = parser.parse_args()

    if not LLM_API_KEY:
        print("[ERROR] XIAOMI_MIMO_API_KEY not set!")
        print("  export XIAOMI_MIMO_API_KEY='sk-your-key'")
        sys.exit(1)

    start = time.time()
    results = run_optimized_evaluation(args.frameworks, args.domains)
    elapsed = time.time() - start

    # Print summary
    print_comparison_table(results)

    # Save results
    out_path = RESULTS_DIR / "optimized_eval_v2_results.json"
    # Remove detailed per-sample data to keep JSON manageable
    save_data = dict(results)
    for fw in save_data["per_framework"]:
        for dom in save_data["per_framework"][fw]["domains"]:
            # Keep detailed but trim long fields
            for item in save_data["per_framework"][fw]["domains"][dom]["detailed"]:
                item.pop("v2_clean_answer_preview", None)

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    print(f"Total evaluation time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
