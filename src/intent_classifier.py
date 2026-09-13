"""
Intent Classification Module.
Provides calibrated intent classification across the 7 defined customer support intents.
Uses a hybrid semantic pipeline: TF-IDF with word and character n-grams + Logistic Regression
trained on rich customer support patterns with normalized confidence scoring.
"""

import os
import re
import json
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.config import Intent, INTENT_DESCRIPTIONS

class IntentClassifier:
    """
    Classifies incoming customer messages into one of the 7 brand intents
    with calibrated confidence scores.
    """

    def __init__(self):
        self.pipeline: Pipeline = None
        self.labels: List[str] = [i.value for i in Intent]
        self._initialize_and_train()

    def _initialize_and_train(self):
        training_texts = []
        training_labels = []

        # 1. Intent descriptions & definitions
        for intent in Intent:
            desc = INTENT_DESCRIPTIONS[intent]
            training_texts.append(desc)
            training_labels.append(intent.value)

        # 2. Comprehensive domain pattern seeds
        patterns = {
            Intent.ORDER_TRACKING_DELIVERY.value: [
                "where is my package", "order tracking status update", "delivery is delayed late",
                "carrier has not scanned package", "marked delivered but not received missing",
                "when will my order arrive", "shipping tracking updates", "address change before dispatch",
                "driver left package outside", "tracking number not found invalid", "delivery date postponed",
                "late shipment prime guaranteed delivery", "has my order dispatched", "carrier courier delay",
                "package said delivered resident nobody came", "package delivered wrong house porch",
                "delivery address update", "carrier transit scan"
            ],
            Intent.REFUND_CANCELLATION.value: [
                "cancel my order immediately", "return this item how do i return", "haven't received my refund yet",
                "return window expired exception", "charged return shipping postage fee", "drop off return package ups",
                "refund status on bank credit card", "stop subscription cancellation", "accidental purchase return kindle",
                "return label qr code printing", "money not credited back ledger", "return item damaged box",
                "dropped off package at courier refund not issued", "cancel order before shipping",
                "request full refund", "refund pending timeline"
            ],
            Intent.PRODUCT_DEFECT_WRONG_ITEM.value: [
                "item arrived damaged broken shattered", "received completely wrong product item", "missing parts pieces in box",
                "package was empty inside box", "box arrived crushed opened torn", "product stopped working defective",
                "ordered blue received red", "counterfeit fake product received", "expired food medicine delivered",
                "defective electronic device won't turn on", "damaged screen on delivery cracked",
                "ceramic mug broken handle", "wrong size shoes delivered", "faulty item replacement"
            ],
            Intent.ACCOUNT_SECURITY_LOGIN.value: [
                "cannot log into my account login failed", "otp code not arriving on phone sms",
                "unauthorized purchase charges hacked account", "password reset email not received link",
                "account suspended locked for security verification", "two factor authentication 2fa failed",
                "someone changed my account email password", "suspicious login alert unauthorized access",
                "recover my account credentials government id", "account takeover fraudulent order"
            ],
            Intent.PAYMENT_BILLING.value: [
                "unrecognized charge on my credit card bank", "charged twice for the same order duplicate",
                "unexpected prime membership fee charged", "gift card balance not showing applying",
                "payment method declined bank card", "billing discrepancy invoice overcharged",
                "overcharged sales tax", "monthly subscription charge dispute",
                "card charged after cancelling subscription", "update payment expiration date wallet"
            ],
            Intent.GENERAL_INQUIRY_FEEDBACK.value: [
                "amazon app crashes on checkout freeze", "delivery driver was very courteous polite safe",
                "feedback about website navigation layout", "phone customer service representative was rude",
                "is amazon locker free for prime members pickup", "general inquiry feedback question",
                "driver sped down private residential driveway", "how do i leave seller review rating",
                "search filter bug on website", "prime day deals promotional dates"
            ],
            Intent.URGENT_SAFETY_LEGAL.value: [
                "phone charger exploded and caught fire scorched wall", "calling my lawyer filing lawsuit sue",
                "reporting amazon to federal trade commission ftc", "battery started smoking burning hazardous fire",
                "product caused severe allergic injury hospital medical", "filing police report fraud scam theft",
                "attorney general complaint regulatory notice", "consumer protection court legal notice",
                "dangerous safety hazard toxic burn"
            ]
        }

        for intent_name, text_list in patterns.items():
            for text in text_list:
                training_texts.append(text)
                training_labels.append(intent_name)
                # Augmented variations
                training_texts.append(f"Hello support, {text}, please assist.")
                training_labels.append(intent_name)

        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            token_pattern=r'(?u)\b\w+\b'
        )
        base_clf = LogisticRegression(C=10.0, max_iter=1000, class_weight="balanced")
        
        self.pipeline = Pipeline([
            ("tfidf", tfidf),
            ("clf", base_clf)
        ])
        
        self.pipeline.fit(training_texts, training_labels)

    def predict(self, text: str) -> Dict[str, Any]:
        text_clean = text.strip()
        probs = self.pipeline.predict_proba([text_clean])[0]
        classes = self.pipeline.classes_
        
        best_idx = np.argmax(probs)
        predicted_label = classes[best_idx]
        raw_prob = float(probs[best_idx])
        
        # Normalized confidence: scale relative to uniform baseline (1/K)
        # where 1/7 = 0.1428 -> normalized = (raw - 0.1428) / (1 - 0.1428)
        k = len(classes)
        uniform_baseline = 1.0 / k
        normalized_conf = max(0.0, (raw_prob - uniform_baseline) / (1.0 - uniform_baseline))
        
        scores = {cls: round(float(probs[i]), 4) for i, cls in enumerate(classes)}

        return {
            "predicted_intent": predicted_label,
            "confidence": round(normalized_conf, 4),
            "raw_probability": round(raw_prob, 4),
            "all_scores": scores
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict(t) for t in texts]
