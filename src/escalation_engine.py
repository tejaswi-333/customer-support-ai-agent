"""
Escalation Engine for Customer Support AI Agent.
Determines whether incoming inquiries should be auto-handled or escalated to a human agent,
along with a transparent, stated reason and risk score.
"""

import re
from typing import Dict, Any, List, Optional
from src.config import (
    Intent,
    EscalationAction,
    CONFIDENCE_THRESHOLD,
    RISK_SCORE_THRESHOLD,
    MANDATORY_ESCALATION_INTENTS,
    HIGH_RISK_KEYWORDS,
)

class EscalationDecision:
    def __init__(
        self,
        action: EscalationAction,
        reason: str,
        risk_score: float,
        policy_triggers: List[str]
    ):
        self.action = action
        self.reason = reason
        self.risk_score = risk_score
        self.policy_triggers = policy_triggers

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "risk_score": round(self.risk_score, 3),
            "policy_triggers": self.policy_triggers
        }

class EscalationEngine:
    """
    Evaluates incoming customer tweets against safety policies, business rules,
    sentiment distress, and model uncertainty to decide between AUTO_HANDLE and ESCALATE.
    """

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        risk_threshold: float = RISK_SCORE_THRESHOLD
    ):
        self.confidence_threshold = confidence_threshold
        self.risk_threshold = risk_threshold
        self.high_risk_patterns = [
            re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
            for kw in HIGH_RISK_KEYWORDS
        ]

    def evaluate(
        self,
        customer_text: str,
        intent: Intent,
        confidence: float
    ) -> EscalationDecision:
        triggers = []
        risk_score = 0.0
        text_lower = customer_text.lower()

        # 1. Check for High-Risk & Legal/Safety Keywords
        matched_keywords = []
        for kw, pattern in zip(HIGH_RISK_KEYWORDS, self.high_risk_patterns):
            if pattern.search(text_lower):
                matched_keywords.append(kw)
        
        if matched_keywords:
            triggers.append(f"High-risk keyword detected: {', '.join(matched_keywords[:3])}")
            risk_score += 0.60

        # 2. Check Intent-Specific Mandatory Escalations
        if intent in MANDATORY_ESCALATION_INTENTS:
            if intent == Intent.URGENT_SAFETY_LEGAL:
                triggers.append("Mandatory policy: Urgent safety, legal threat, or regulatory complaint.")
                risk_score += 0.85
            elif intent == Intent.ACCOUNT_SECURITY_LOGIN:
                triggers.append("Mandatory policy: Account security or authentication compromise requires human verification.")
                risk_score += 0.70

        # 3. Model Confidence Check (Ambiguity Escalation)
        if confidence < self.confidence_threshold:
            triggers.append(
                f"Model uncertainty: Intent confidence ({confidence:.2f}) below threshold ({self.confidence_threshold:.2f})."
            )
            risk_score += 0.40

        # 4. Financial & Disputed Billing Policy
        billing_escalation_cues = [
            "charged twice", "unrecognized charge", "unexpected charge", "why was my credit card charged",
            "cancelled my prime", "charged $", "money back", "deducted a restocking fee", "haven't received my refund",
            "where is my refund", "refund hasn't been credited", "stole my money", "unauthorized charge"
        ]
        if any(c in text_lower for c in billing_escalation_cues):
            triggers.append("Financial policy: Disputed transaction or delayed refund requires verified billing agent lookup.")
            risk_score += 0.55

        # 5. Delivery Discrepancies & Disputed Signatures
        delivery_dispute_cues = [
            "says delivered", "marked delivered", "delivered to resident", "nobody knocked",
            "nowhere to be found", "delivered to the roof", "driver left the package right on the sidewalk",
            "in the middle of our open driveway", "pouring rain", "driver threw", "almost hit my dog",
            "broken flower pot", "stuck on 'label created'"
        ]
        if any(c in text_lower for c in delivery_dispute_cues):
            triggers.append("Logistics policy: Disputed delivery scan or delivery driver property damage requires carrier investigation.")
            risk_score += 0.55

        # 6. Physical Damage & Fraudulent Substitution
        damage_cues = [
            "shattered", "filled with bars of soap", "empty box", "safety foil was torn",
            "expired food", "counterfeit", "unsealed with fingerprints", "severely damaged"
        ]
        if any(c in text_lower for c in damage_cues):
            triggers.append("Product safety policy: Severe product damage, tamper evident, or suspected counterfeit.")
            risk_score += 0.55

        # 7. Repeated Customer Frustration Markers
        frustration_markers = ["unacceptable", "worst customer service", "ridiculous", "days waiting", "never buying again", "fix your bot"]
        frustration_hits = [m for m in frustration_markers if m in text_lower]
        if frustration_hits:
            triggers.append(f"Customer frustration marker detected: {', '.join(frustration_hits)}")
            risk_score += 0.25

        # Cap risk score at 1.0
        risk_score = min(1.0, risk_score)

        # Final Escalation Decisions
        if risk_score >= self.risk_threshold or len(matched_keywords) > 0 or intent in MANDATORY_ESCALATION_INTENTS:
            primary_reason = triggers[0] if triggers else "Accumulated risk score exceeded safety threshold."
            return EscalationDecision(
                action=EscalationAction.ESCALATE,
                reason=f"Escalated to Human Agent: {primary_reason}",
                risk_score=risk_score,
                policy_triggers=triggers
            )

        if confidence < self.confidence_threshold:
            return EscalationDecision(
                action=EscalationAction.ESCALATE,
                reason=f"Escalated to Human Agent: Low intent classification confidence ({confidence:.2f}).",
                risk_score=risk_score,
                policy_triggers=triggers
            )

        # Default Auto-Handle Reasons
        reason_map = {
            Intent.ORDER_TRACKING_DELIVERY: "Auto-handled: Providing self-service carrier tracking link and delivery timeline guidance.",
            Intent.REFUND_CANCELLATION: "Auto-handled: Directing customer to Online Return Center and return window policy.",
            Intent.PRODUCT_DEFECT_WRONG_ITEM: "Auto-handled: Providing instant replacement exchange portal link and instructions.",
            Intent.PAYMENT_BILLING: "Auto-handled: Providing link to manage saved payment methods and subscription billing history.",
            Intent.GENERAL_INQUIRY_FEEDBACK: "Auto-handled: Acknowledging customer inquiry and providing official Help Center resources."
        }
        auto_reason = reason_map.get(intent, "Auto-handled: Routine query safely resolvable through self-service portals.")

        return EscalationDecision(
            action=EscalationAction.AUTO_HANDLE,
            reason=auto_reason,
            risk_score=risk_score,
            policy_triggers=triggers
        )
