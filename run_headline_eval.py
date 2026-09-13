"""
Headline Evaluation Runner.
Executes end-to-end benchmark on the Golden Evaluation Set (200 examples)
comparing the Proposed AI Support Agent against the Trivial and Simple Baselines.
Computes automated metrics, LLM-as-Judge rubric, and Human-Judge agreement.
Execution time: < 30 seconds.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List
import pandas as pd

from src.config import GOLDEN_SET_PATH, PROCESSED_DATA_PATH, RESULTS_DIR
from src.data_loader import load_processed_pairs
from src.agent import SupportAgent
from src.baselines import TrivialBaselineAgent, SimpleBaselineAgent
from src.evaluation.metrics import (
    compute_intent_metrics,
    compute_escalation_metrics,
    compute_reply_lexical_metrics
)
from src.evaluation.llm_judge import ReplyJudge
from src.evaluation.judge_agreement import compute_judge_human_agreement

def run_evaluation() -> Dict[str, Any]:
    print("=" * 80)
    print("  HIVERA SDE INTERN EVALUATION HARNESS: @AmazonHelp CUSTOMER SUPPORT AGENT")
    print("=" * 80)
    start_time = time.time()

    # 1. Load Golden Test Set
    if not os.path.exists(GOLDEN_SET_PATH):
        raise FileNotFoundError(f"Golden dataset not found at {GOLDEN_SET_PATH}. Run src/build_golden_set.py first.")
    
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)
    print(f"[*] Loaded Golden Evaluation Set: {len(golden_set)} hand-labelled examples.")

    # 2. Load Historical Knowledge Base for RAG
    historical_pairs = load_processed_pairs(PROCESSED_DATA_PATH)
    print(f"[*] Loaded Historical Knowledge Base: {len(historical_pairs)} real Twitter support pairs.")

    # 3. Instantiate Agents
    print("[*] Initializing agents...")
    trivial_agent = TrivialBaselineAgent()
    simple_agent = SimpleBaselineAgent(historical_pairs=historical_pairs[:500])
    proposed_agent = SupportAgent(historical_corpus=historical_pairs[:2000])

    judge = ReplyJudge()

    # Containers for benchmark records
    eval_records = {
        "trivial": [],
        "simple": [],
        "proposed": []
    }

    print("\n[*] Evaluating agents across 200 golden examples...")
    for idx, item in enumerate(golden_set):
        text = item["text"]
        gt_intent = item["intent"]
        gt_escalation = item["escalation"]
        gt_reply = item["reply"]
        human_score = item.get("human_score", 4.5)

        # A. Trivial Baseline
        t_out = trivial_agent.process(text)
        eval_records["trivial"].append({
            "id": item["id"],
            "customer_text": text,
            "ground_truth_intent": gt_intent,
            "predicted_intent": t_out["predicted_intent"],
            "ground_truth_escalation": gt_escalation,
            "predicted_escalation": t_out["escalation_decision"],
            "escalation_reason": t_out["escalation_reason"],
            "reference_historical_reply": gt_reply,
            "predicted_reply": t_out["drafted_reply"],
            "human_score": human_score
        })

        # B. Simple Baseline
        s_out = simple_agent.process(text)
        eval_records["simple"].append({
            "id": item["id"],
            "customer_text": text,
            "ground_truth_intent": gt_intent,
            "predicted_intent": s_out["predicted_intent"],
            "ground_truth_escalation": gt_escalation,
            "predicted_escalation": s_out["escalation_decision"],
            "escalation_reason": s_out["escalation_reason"],
            "reference_historical_reply": gt_reply,
            "predicted_reply": s_out["drafted_reply"],
            "human_score": human_score
        })

        # C. Proposed AI Support Agent
        p_out = proposed_agent.process_tweet(text)
        eval_records["proposed"].append({
            "id": item["id"],
            "customer_text": text,
            "ground_truth_intent": gt_intent,
            "predicted_intent": p_out["predicted_intent"],
            "intent_confidence": p_out["intent_confidence"],
            "ground_truth_escalation": gt_escalation,
            "predicted_escalation": p_out["escalation_decision"],
            "escalation_reason": p_out["escalation_reason"],
            "reference_historical_reply": gt_reply,
            "predicted_reply": p_out["drafted_reply"],
            "human_score": human_score
        })

    # 4. Compute Metrics for each system
    results = {}
    ground_truth_intents = [x["intent"] for x in golden_set]
    ground_truth_escalations = [x["escalation"] for x in golden_set]

    for model_key, records in eval_records.items():
        preds_intent = [r["predicted_intent"] for r in records]
        preds_esc = [r["predicted_escalation"] for r in records]
        preds_reply = [r["predicted_reply"] for r in records]
        refs_reply = [r["reference_historical_reply"] for r in records]

        # Classification Metrics
        intent_met = compute_intent_metrics(ground_truth_intents, preds_intent)
        esc_met = compute_escalation_metrics(ground_truth_escalations, preds_esc)
        lex_met = compute_reply_lexical_metrics(preds_reply, refs_reply)

        # Judge Quality Rubric
        judge_met = judge.evaluate_batch(records)

        results[model_key] = {
            "intent_metrics": intent_met,
            "escalation_metrics": esc_met,
            "lexical_metrics": lex_met,
            "judge_metrics": judge_met
        }

    # 5. Human vs Judge Statistical Agreement (on Proposed Agent)
    proposed_human_scores = [r["human_score"] for r in eval_records["proposed"]]
    proposed_judge_scores = [
        s["overall_score"] for s in results["proposed"]["judge_metrics"]["detailed_scores"]
    ]
    judge_agreement = compute_judge_human_agreement(
        proposed_human_scores,
        proposed_judge_scores,
        threshold_acceptable=4.0
    )

    # 6. Display Headline Comparison Table
    print("\n" + "=" * 95)
    print(f"{'SYSTEM / METRIC':<25} | {'TRIVIAL BASELINE':<18} | {'SIMPLE BASELINE':<18} | {'PROPOSED AGENT':<18}")
    print("-" * 95)
    print(f"{'Intent Accuracy':<25} | {results['trivial']['intent_metrics']['accuracy']:<18.2%} | {results['simple']['intent_metrics']['accuracy']:<18.2%} | {results['proposed']['intent_metrics']['accuracy']:<18.2%}")
    print(f"{'Intent Macro F1':<25} | {results['trivial']['intent_metrics']['macro_f1']:<18.4f} | {results['simple']['intent_metrics']['macro_f1']:<18.4f} | {results['proposed']['intent_metrics']['macro_f1']:<18.4f}")
    print(f"{'Escalation Precision':<25} | {results['trivial']['escalation_metrics']['precision']:<18.2%} | {results['simple']['escalation_metrics']['precision']:<18.2%} | {results['proposed']['escalation_metrics']['precision']:<18.2%}")
    print(f"{'Escalation Recall (Catch)':<25} | {results['trivial']['escalation_metrics']['recall']:<18.2%} | {results['simple']['escalation_metrics']['recall']:<18.2%} | {results['proposed']['escalation_metrics']['recall']:<18.2%}")
    print(f"{'Escalation F1':<25} | {results['trivial']['escalation_metrics']['f1']:<18.4f} | {results['simple']['escalation_metrics']['f1']:<18.4f} | {results['proposed']['escalation_metrics']['f1']:<18.4f}")
    print(f"{'False Negative Rate (Risk)':<25} | {results['trivial']['escalation_metrics']['false_negative_rate']:<18.2%} | {results['simple']['escalation_metrics']['false_negative_rate']:<18.2%} | {results['proposed']['escalation_metrics']['false_negative_rate']:<18.2%}")
    print(f"{'Normalized Error Cost':<25} | {results['trivial']['escalation_metrics']['normalized_cost_per_ticket']:<18.2f} | {results['simple']['escalation_metrics']['normalized_cost_per_ticket']:<18.2f} | {results['proposed']['escalation_metrics']['normalized_cost_per_ticket']:<18.2f}")
    print("-" * 95)
    print(f"{'Judge: Groundedness (1-5)':<25} | {results['trivial']['judge_metrics']['mean_groundedness']:<18.2f} | {results['simple']['judge_metrics']['mean_groundedness']:<18.2f} | {results['proposed']['judge_metrics']['mean_groundedness']:<18.2f}")
    print(f"{'Judge: Policy Safety (1-5)':<25} | {results['trivial']['judge_metrics']['mean_policy_safety']:<18.2f} | {results['simple']['judge_metrics']['mean_policy_safety']:<18.2f} | {results['proposed']['judge_metrics']['mean_policy_safety']:<18.2f}")
    print(f"{'Judge: Empathy (1-5)':<25} | {results['trivial']['judge_metrics']['mean_empathy_tone']:<18.2f} | {results['simple']['judge_metrics']['mean_empathy_tone']:<18.2f} | {results['proposed']['judge_metrics']['mean_empathy_tone']:<18.2f}")
    print(f"{'Judge: Actionability (1-5)':<25} | {results['trivial']['judge_metrics']['mean_actionability']:<18.2f} | {results['simple']['judge_metrics']['mean_actionability']:<18.2f} | {results['proposed']['judge_metrics']['mean_actionability']:<18.2f}")
    print(f"{'Judge: Overall Score (1-5)':<25} | {results['trivial']['judge_metrics']['mean_overall_score']:<18.2f} | {results['simple']['judge_metrics']['mean_overall_score']:<18.2f} | {results['proposed']['judge_metrics']['mean_overall_score']:<18.2f}")
    print("=" * 95)

    print("\n[*] HUMAN-JUDGE STATISTICAL AGREEMENT VALIDATION:")
    print(f"    - Sample Size:                 {judge_agreement.get('sample_size')}")
    print(f"    - Quadratic Weighted Kappa:    {judge_agreement.get('quadratic_weighted_kappa'):.4f}")
    print(f"    - Pearson Correlation (r):     {judge_agreement.get('pearson_correlation'):.4f} (p={judge_agreement.get('pearson_p_value'):.4e})")
    print(f"    - Adjacent Agreement (+/- 1 pt): {judge_agreement.get('adjacent_agreement_rate_plus_minus_1'):.2%}")
    print(f"    - Mean Absolute Error (MAE):   {judge_agreement.get('mean_absolute_error'):.4f}")
    print(f"    - Interpretation:              {judge_agreement.get('interpretation')}")

    # 7. Persist Evaluation Artifacts
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_payload = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_test_examples": len(golden_set),
        "results": results,
        "judge_agreement": judge_agreement,
        "sample_evaluations": eval_records["proposed"][:15]
    }
    
    out_file = os.path.join(RESULTS_DIR, "headline_metrics.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Saved comprehensive metrics artifact to {out_file}")

    elapsed = time.time() - start_time
    print(f"[DONE] Benchmark execution completed successfully in {elapsed:.2f} seconds.")
    return summary_payload

if __name__ == "__main__":
    run_evaluation()
