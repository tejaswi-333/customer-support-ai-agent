"""
Configuration module for the Customer Support AI Agent.
Brand: @AmazonHelp
"""

import os
from enum import Enum
from typing import Dict, List, Any

# Brand identifier and characteristics
BRAND_HANDLE = "@AmazonHelp"
BRAND_NAME = "Amazon Customer Service"

class Intent(str, Enum):
    ORDER_TRACKING_DELIVERY = "ORDER_TRACKING_DELIVERY"
    REFUND_CANCELLATION = "REFUND_CANCELLATION"
    PRODUCT_DEFECT_WRONG_ITEM = "PRODUCT_DEFECT_WRONG_ITEM"
    ACCOUNT_SECURITY_LOGIN = "ACCOUNT_SECURITY_LOGIN"
    PAYMENT_BILLING = "PAYMENT_BILLING"
    GENERAL_INQUIRY_FEEDBACK = "GENERAL_INQUIRY_FEEDBACK"
    URGENT_SAFETY_LEGAL = "URGENT_SAFETY_LEGAL"

INTENT_DESCRIPTIONS: Dict[Intent, str] = {
    Intent.ORDER_TRACKING_DELIVERY: (
        "Inquiries about package delivery status, late shipments, carrier tracking updates, "
        "or packages marked delivered but not received."
    ),
    Intent.REFUND_CANCELLATION: (
        "Requests to cancel an order, return an item, check refund status, or disputes over "
        "return policy / restocking fees."
    ),
    Intent.PRODUCT_DEFECT_WRONG_ITEM: (
        "Reports of damaged goods, broken packaging, incorrect items sent, missing parts, "
        "or counterfeit/expired goods."
    ),
    Intent.ACCOUNT_SECURITY_LOGIN: (
        "Trouble logging in, password reset failures, two-factor authentication (OTP) issues, "
        "unauthorized account changes, or suspected compromised accounts."
    ),
    Intent.PAYMENT_BILLING: (
        "Unrecognized credit card charges, unexpected Amazon Prime membership charges, "
        "gift card redemption issues, or payment method declines."
    ),
    Intent.GENERAL_INQUIRY_FEEDBACK: (
        "Feedback regarding website/app navigation, customer service experience, delivery driver behavior, "
        "or general inquiries not tied to a specific order issue."
    ),
    Intent.URGENT_SAFETY_LEGAL: (
        "Severe situations: battery explosion/fire hazard, injury caused by a product, threats of "
        "lawsuits, regulatory complaints (FTC/Better Business Bureau), or criminal fraud accusations."
    ),
}

class EscalationAction(str, Enum):
    AUTO_HANDLE = "AUTO_HANDLE"
    ESCALATE = "ESCALATE"

# Escalation Risk Thresholds
CONFIDENCE_THRESHOLD = 0.35  # Normalized confidence threshold (relative to uniform random chance)
RISK_SCORE_THRESHOLD = 0.50  # If risk score >= 0.50, escalate to human

# Intents that mandate human escalation under standard policy
MANDATORY_ESCALATION_INTENTS = {
    Intent.URGENT_SAFETY_LEGAL,
    Intent.ACCOUNT_SECURITY_LOGIN,
}

# High-risk triggers requiring immediate human escalation
HIGH_RISK_KEYWORDS = [
    "lawyer", "attorney", "sue", "lawsuit", "court", "police", "legal action",
    "fraud", "scam", "stolen", "thief", "fire", "explosion", "burn", "hazard",
    "injury", "hospital", "poison", "chargeback", "better business bureau", "bbb",
    "consumer court", "hacked", "unauthorized access", "breach"
]

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "dataset.parquet")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "amazon_help_pairs.jsonl")
GOLDEN_SET_PATH = os.path.join(DATA_DIR, "golden_set.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
