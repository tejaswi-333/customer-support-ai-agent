"""
Baseline Customer Support Agents for Benchmark Comparison.
Includes:
1. Trivial Baseline: Majority class intent + static canned template reply + naive escalation.
2. Simple Baseline: Unigram keyword matching + raw 1-NN verbatim historical reply + heuristic escalation.
"""

from typing import Dict, Any, List
import re
from src.config import Intent, EscalationAction

class TrivialBaselineAgent:
    """
    Trivial Baseline:
    - Intent: Always predicts the most common class (ORDER_TRACKING_DELIVERY).
    - Escalation: Always escalates (ESCALATE) to human.
    - Reply: Static canned template message.
    """
    def __init__(self, default_intent: str = Intent.ORDER_TRACKING_DELIVERY.value):
        self.default_intent = default_intent
        self.canned_reply = (
            "Thank you for reaching out to customer support. We are sorry for the inconvenience. "
            "Please send us a direct message with your details so we can assist you."
        )

    def process(self, customer_text: str) -> Dict[str, Any]:
        return {
            "predicted_intent": self.default_intent,
            "intent_confidence": 0.20,
            "escalation_decision": EscalationAction.ESCALATE.value,
            "escalation_reason": "Trivial Baseline: Defaulting to human escalation for all tickets.",
            "drafted_reply": self.canned_reply,
            "grounding_confidence": 0.0
        }

    def process_batch(self, customer_texts: List[str]) -> List[Dict[str, Any]]:
        return [self.process(t) for t in customer_texts]


class SimpleBaselineAgent:
    """
    Simple Baseline:
    - Intent: Naive keyword search over fixed intent keywords.
    - Escalation: Heuristic rule (if query contains '!' or 'urgent' or 'not' -> ESCALATE, else AUTO_HANDLE).
    - Reply: Verbatim 1-NN retrieval from historical replies (unadapted, unverified).
    """
    def __init__(self, historical_pairs: List[Dict[str, str]] = None):
        self.historical_pairs = historical_pairs or []
        self.intent_keywords = {
            Intent.ORDER_TRACKING_DELIVERY.value: ["where", "package", "tracking", "delivery", "arrive", "shipped", "carrier"],
            Intent.REFUND_CANCELLATION.value: ["refund", "cancel", "return", "cancellation", "credited", "money back"],
            Intent.PRODUCT_DEFECT_WRONG_ITEM.value: ["broken", "damaged", "wrong", "defective", "missing", "faulty", "scratch"],
            Intent.ACCOUNT_SECURITY_LOGIN.value: ["login", "password", "otp", "hacked", "account", "locked", "verification"],
            Intent.PAYMENT_BILLING.value: ["charge", "card", "billing", "prime", "overcharged", "fee", "payment", "bank"],
            Intent.GENERAL_INQUIRY_FEEDBACK.value: ["feedback", "app", "website", "driver", "service", "store", "locker"],
            Intent.URGENT_SAFETY_LEGAL.value: ["lawyer", "fire", "exploded", "sue", "legal", "police", "hazard", "court"]
        }

    def _match_intent(self, text: str) -> str:
        text_lower = text.lower()
        best_intent = Intent.ORDER_TRACKING_DELIVERY.value
        max_matches = -1
        
        for intent, kw_list in self.intent_keywords.items():
            matches = sum(1 for kw in kw_list if kw in text_lower)
            if matches > max_matches:
                max_matches = matches
                best_intent = intent
                
        return best_intent

    def _heuristic_escalate(self, text: str) -> str:
        text_lower = text.lower()
        escalate_cues = ["urgent", "immediately", "worst", "ridiculous", "never", "!", "refund", "stolen", "terrible"]
        if any(c in text_lower for c in escalate_cues):
            return EscalationAction.ESCALATE.value
        return EscalationAction.AUTO_HANDLE.value

    def process(self, customer_text: str) -> Dict[str, Any]:
        intent = self._match_intent(customer_text)
        escalation = self._heuristic_escalate(customer_text)
        
        # Simple verbatim retrieval fallback
        raw_reply = (
            "We'd like to look into this. Please send us your order number via private message so we can check."
        )
        if self.historical_pairs:
            # Pick first pair that shares a token
            for pair in self.historical_pairs[:50]:
                if any(w in pair.get("customer_query", "").lower() for w in customer_text.lower().split()[:3]):
                    raw_reply = pair.get("agent_reply", raw_reply)
                    break

        return {
            "predicted_intent": intent,
            "intent_confidence": 0.50,
            "escalation_decision": escalation,
            "escalation_reason": "Simple Baseline: Heuristic keyword escalation.",
            "drafted_reply": raw_reply,
            "grounding_confidence": 0.35
        }

    def process_batch(self, customer_texts: List[str]) -> List[Dict[str, Any]]:
        return [self.process(t) for t in customer_texts]
