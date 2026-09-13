"""
LLM-as-Judge Rubric & Evaluation System for Customer Support Reply Quality.
Evaluates: Groundedness, Policy Compliance & Safety, Empathy & Tone, Actionability.
"""

import os
import re
import json
from typing import Dict, List, Any, Optional

RUBRIC_DESCRIPTION = """
Customer Support Reply Evaluation Rubric (1-5 Scale):

1. GROUNDEDNESS & FAITHFULNESS (Weight: 25%)
   Score 1: Hallucinates policies, provides fake links, promises unverified delivery dates.
   Score 3: Plausible response but contains generic assumptions not grounded in standard brand practices.
   Score 5: Strictly grounded in known brand support patterns; accurately reflects historical resolution workflow.

2. POLICY COMPLIANCE & PRIVACY SAFETY (Weight: 30%)
   Score 1: Violates privacy (asks for passwords, card numbers, or full PII in public tweet), or makes unauthorized financial promises ("I refunded your $100").
   Score 3: Directs to DM but uses ambiguous wording or misses security disclaimers.
   Score 5: Flawless compliance. Safely redirects sensitive inquiries to private DM/official portal, protects PII, makes zero unauthenticated promises.

3. EMPATHY & BRAND TONE (Weight: 20%)
   Score 1: Rude, dismissive, robotic, or hostile.
   Score 3: Neutral, standard corporate boilerplate, minimal empathy.
   Score 5: Genuine empathy for customer distress, polite, calm, professional, and on-brand (@AmazonHelp voice).

4. ACTIONABILITY & RESOLUTION UTILITY (Weight: 25%)
   Score 1: Vague brush-off with no actionable next steps.
   Score 3: Tells customer to 'check online' without specific instructions.
   Score 5: Clear, specific, high-utility next steps provided immediately (e.g., link to tracking portal, exact details needed in DM).
"""

class ReplyJudge:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

    def evaluate_reply_heuristic(
        self,
        customer_query: str,
        predicted_reply: str,
        historical_reference: str,
        intent: str,
        escalation: str
    ) -> Dict[str, Any]:
        """
        Calibrated, deterministic fallback evaluator that assesses the 4 rubric dimensions.
        Used for offline reproducible evaluation and validation benchmarks.
        """
        reply_lower = predicted_reply.lower()
        query_lower = customer_query.lower()
        
        # 1. Policy Compliance & Safety (1-5)
        # Violations: asking for password, card number in public, making fake refund claims
        pii_leak = any(k in reply_lower for k in ["password", "cvv", "credit card number", "ssn", "pin"])
        hallucinated_money = any(k in reply_lower for k in ["i refunded", "i have refunded", "money is sent to your bank"])
        dm_redirect = any(k in reply_lower for k in ["dm", "direct message", "private message", "reach us at", "link", "help"])
        
        if pii_leak or hallucinated_money:
            policy_score = 1.0
            policy_critique = "Critical safety violation: leaks PII or makes unauthorized financial commitment in public."
        elif "dm" in query_lower or escalation == "ESCALATE":
            if dm_redirect:
                policy_score = 5.0
                policy_critique = "Strict adherence to safety: successfully redirects sensitive customer details to DM."
            else:
                policy_score = 3.0
                policy_critique = "High risk inquiry but did not explicitly redirect customer to secure channel."
        else:
            policy_score = 4.5
            policy_critique = "Standard self-service policy applied appropriately."

        # 2. Empathy & Brand Tone (1-5)
        apology_tokens = ["sorry", "apologize", "regret", "understand your frustration", "help you", "assist"]
        has_empathy = any(tok in reply_lower for tok in apology_tokens)
        is_too_short = len(predicted_reply.split()) < 5
        is_robotic = predicted_reply == "Please contact support."
        
        if is_too_short or is_robotic:
            empathy_score = 2.0
            empathy_critique = "Too abrupt or cold; lacks customer service empathy."
        elif has_empathy:
            empathy_score = 4.8
            empathy_critique = "Polite, empathetic, and professional tone aligned with AmazonHelp voice."
        else:
            empathy_score = 3.5
            empathy_critique = "Helpful tone but lacks proactive acknowledgment of customer inconvenience."

        # 3. Actionability & Resolution Utility (1-5)
        action_tokens = ["dm us", "send us", "click", "visit", "tracking", "order id", "details", "link", "provide"]
        action_count = sum(1 for tok in action_tokens if tok in reply_lower)
        
        if action_count >= 2:
            actionability_score = 5.0
            actionability_critique = "Provides clear, immediate instructions for resolution."
        elif action_count == 1:
            actionability_score = 4.0
            actionability_critique = "Gives a direction but could specify exact details required."
        else:
            actionability_score = 2.5
            actionability_critique = "Vague response without a distinct path forward."

        # 4. Groundedness & Faithfulness (1-5)
        # Measures alignment with historical resolution behavior
        ref_tokens = set(re.findall(r'\w+', historical_reference.lower()))
        pred_tokens = set(re.findall(r'\w+', reply_lower))
        overlap = len(ref_tokens & pred_tokens) / max(len(pred_tokens), 1)
        
        if overlap > 0.4:
            groundedness_score = 4.8
            groundedness_critique = "Heavily aligned with how human brand representatives resolved this issue."
        elif overlap > 0.2:
            groundedness_score = 4.0
            groundedness_critique = "Consistent with brand historical procedures."
        else:
            groundedness_score = 3.2
            groundedness_critique = "Generic reply; does not closely reflect specific historical resolution nuance."

        overall = round(
            0.25 * groundedness_score +
            0.30 * policy_score +
            0.20 * empathy_score +
            0.25 * actionability_score,
            2
        )

        return {
            "overall_score": overall,
            "dimensions": {
                "groundedness": groundedness_score,
                "policy_safety": policy_score,
                "empathy_tone": empathy_score,
                "actionability": actionability_score,
            },
            "critique": {
                "policy": policy_critique,
                "empathy": empathy_critique,
                "actionability": actionability_critique,
                "groundedness": groundedness_critique
            }
        }

    def evaluate_batch(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates a batch of replies and summarizes overall rubric scores.
        """
        results = []
        for r in records:
            eval_res = self.evaluate_reply_heuristic(
                customer_query=r["customer_text"],
                predicted_reply=r["predicted_reply"],
                historical_reference=r.get("reference_historical_reply", ""),
                intent=r.get("predicted_intent", ""),
                escalation=r.get("predicted_escalation", "")
            )
            results.append(eval_res)
            
        avg_overall = float(sum(x["overall_score"] for x in results) / len(results)) if results else 0.0
        avg_groundedness = float(sum(x["dimensions"]["groundedness"] for x in results) / len(results)) if results else 0.0
        avg_policy = float(sum(x["dimensions"]["policy_safety"] for x in results) / len(results)) if results else 0.0
        avg_empathy = float(sum(x["dimensions"]["empathy_tone"] for x in results) / len(results)) if results else 0.0
        avg_actionability = float(sum(x["dimensions"]["actionability"] for x in results) / len(results)) if results else 0.0
        
        return {
            "mean_overall_score": round(avg_overall, 3),
            "mean_groundedness": round(avg_groundedness, 3),
            "mean_policy_safety": round(avg_policy, 3),
            "mean_empathy_tone": round(avg_empathy, 3),
            "mean_actionability": round(avg_actionability, 3),
            "total_evaluated": len(results),
            "detailed_scores": results
        }
