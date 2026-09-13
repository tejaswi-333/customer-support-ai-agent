"""
Main AI Customer Support Agent Orchestrator.
Unifies Intent Classification, Escalation Decision Engine, and Historical RAG Reply Generation.
"""

from typing import Dict, Any, List, Optional
from src.config import Intent, EscalationAction
from src.intent_classifier import IntentClassifier
from src.escalation_engine import EscalationEngine
from src.reply_generator import ReplyGenerator, HistoricalResolutionStore

class SupportAgent:
    """
    Production-grade AI Support Agent for Twitter customer inquiries.
    """
    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        escalation_engine: Optional[EscalationEngine] = None,
        reply_generator: Optional[ReplyGenerator] = None,
        historical_corpus: Optional[List[Dict[str, str]]] = None
    ):
        self.classifier = classifier or IntentClassifier()
        self.escalation_engine = escalation_engine or EscalationEngine()
        
        if reply_generator is None:
            store = HistoricalResolutionStore()
            if historical_corpus:
                store.fit(historical_corpus)
            self.reply_generator = ReplyGenerator(store)
        else:
            self.reply_generator = reply_generator

    def process_tweet(self, customer_text: str) -> Dict[str, Any]:
        """
        End-to-end processing of an incoming customer tweet:
        1. Intent classification + confidence score
        2. Escalation decision + stated reason + risk score
        3. Grounded reply generation
        """
        # 1. Intent Classification
        clf_result = self.classifier.predict(customer_text)
        predicted_intent_str = clf_result["predicted_intent"]
        confidence = clf_result["confidence"]
        intent_enum = Intent(predicted_intent_str)

        # 2. Escalation Decision Engine
        esc_decision = self.escalation_engine.evaluate(
            customer_text=customer_text,
            intent=intent_enum,
            confidence=confidence
        )
        esc_dict = esc_decision.to_dict()

        # 3. Grounded Reply Drafting
        reply_dict = self.reply_generator.draft_reply(
            customer_text=customer_text,
            intent=intent_enum,
            escalation_decision=esc_dict
        )

        return {
            "customer_text": customer_text,
            "predicted_intent": predicted_intent_str,
            "intent_confidence": confidence,
            "all_intent_scores": clf_result["all_scores"],
            "escalation_decision": esc_dict["action"],
            "escalation_reason": esc_dict["reason"],
            "risk_score": esc_dict["risk_score"],
            "policy_triggers": esc_dict["policy_triggers"],
            "drafted_reply": reply_dict["drafted_reply"],
            "grounding_confidence": reply_dict["grounding_confidence"],
            "retrieved_historical_context": reply_dict["retrieved_historical_context"]
        }

    def process_batch(self, customer_texts: List[str]) -> List[Dict[str, Any]]:
        return [self.process_tweet(t) for t in customer_texts]
