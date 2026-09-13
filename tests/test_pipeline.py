import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Intent, EscalationAction
from src.intent_classifier import IntentClassifier
from src.escalation_engine import EscalationEngine
from src.reply_generator import ReplyGenerator, HistoricalResolutionStore
from src.agent import SupportAgent
from src.evaluation.metrics import compute_intent_metrics, compute_escalation_metrics

class TestSupportPipeline(unittest.TestCase):
    def test_intent_classifier(self):
        clf = IntentClassifier()
        res = clf.predict("Where is my package? The tracking has not updated in two days.")
        self.assertIn("predicted_intent", res)
        self.assertIn(res["predicted_intent"], [i.value for i in Intent])
        self.assertTrue(0.0 <= res["confidence"] <= 1.0)
        self.assertEqual(res["predicted_intent"], Intent.ORDER_TRACKING_DELIVERY.value)

    def test_mandatory_escalation(self):
        engine = EscalationEngine()
        # Safety / legal hazard
        decision = engine.evaluate(
            customer_text="The battery exploded and scorched my living room carpet. I am suing Amazon!",
            intent=Intent.URGENT_SAFETY_LEGAL,
            confidence=0.95
        )
        self.assertEqual(decision.action, EscalationAction.ESCALATE)
        self.assertTrue("safety" in decision.reason.lower() or "legal" in decision.reason.lower())
        self.assertGreaterEqual(decision.risk_score, 0.50)

    def test_auto_handle_standard_query(self):
        engine = EscalationEngine()
        decision = engine.evaluate(
            customer_text="Where is my package? How do I check tracking?",
            intent=Intent.ORDER_TRACKING_DELIVERY,
            confidence=0.85
        )
        self.assertEqual(decision.action, EscalationAction.AUTO_HANDLE)
        self.assertTrue("tracking" in decision.reason.lower())

    def test_low_confidence_ambiguity_escalation(self):
        engine = EscalationEngine()
        decision = engine.evaluate(
            customer_text="hello why is this happening??",
            intent=Intent.GENERAL_INQUIRY_FEEDBACK,
            confidence=0.15  # below threshold
        )
        self.assertEqual(decision.action, EscalationAction.ESCALATE)
        self.assertTrue("confidence" in decision.reason.lower() or "ambiguous" in decision.reason.lower())

    def test_reply_generator_safety(self):
        store = HistoricalResolutionStore()
        generator = ReplyGenerator(store)
        
        # Test escalation draft
        res = generator.draft_reply(
            customer_text="Someone hacked my account and ordered gift cards!",
            intent=Intent.ACCOUNT_SECURITY_LOGIN,
            escalation_decision={"action": EscalationAction.ESCALATE.value, "reason": "Account security breach"}
        )
        reply = res["drafted_reply"]
        self.assertTrue("password" not in reply.lower() or "never share" in reply.lower())
        self.assertTrue("dm" in reply.lower() or "help" in reply.lower())

    def test_end_to_end_agent(self):
        agent = SupportAgent()
        output = agent.process_tweet("I received the wrong item in my delivery, ordered a laptop and got a book.")
        self.assertEqual(output["predicted_intent"], Intent.PRODUCT_DEFECT_WRONG_ITEM.value)
        self.assertIn(output["escalation_decision"], [EscalationAction.AUTO_HANDLE.value, EscalationAction.ESCALATE.value])
        self.assertGreater(len(output["escalation_reason"]), 10)
        self.assertGreater(len(output["drafted_reply"]), 20)

    def test_evaluation_metrics(self):
        y_true = ["ORDER_TRACKING_DELIVERY", "REFUND_CANCELLATION", "ACCOUNT_SECURITY_LOGIN"]
        y_pred = ["ORDER_TRACKING_DELIVERY", "REFUND_CANCELLATION", "GENERAL_INQUIRY_FEEDBACK"]
        
        intent_metrics = compute_intent_metrics(y_true, y_pred)
        self.assertAlmostEqual(intent_metrics["accuracy"], 2/3, places=2)

        esc_true = ["ESCALATE", "AUTO_HANDLE", "ESCALATE"]
        esc_pred = ["ESCALATE", "AUTO_HANDLE", "AUTO_HANDLE"]
        esc_metrics = compute_escalation_metrics(esc_true, esc_pred, cost_fn=5.0, cost_fp=1.0)
        self.assertEqual(esc_metrics["false_negatives"], 1)
        self.assertEqual(esc_metrics["false_positives"], 0)
        self.assertEqual(esc_metrics["total_cost"], 5.0)

if __name__ == "__main__":
    unittest.main()
