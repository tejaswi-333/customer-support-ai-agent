"""
Evaluation metrics for Intent Classification, Escalation Decisions, and Reply Quality.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

def compute_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Computes standard classification metrics for intent classification:
    Accuracy, Macro Precision/Recall/F1, and per-class metrics.
    """
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    p_class, r_class, f1_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": round(float(p_class[i]), 4),
            "recall": round(float(r_class[i]), 4),
            "f1": round(float(f1_class[i]), 4),
            "support": int(support[i])
        }
        
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    
    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": cm
    }

def compute_escalation_metrics(
    y_true: List[str],
    y_pred: List[str],
    cost_fn: float = 5.0,
    cost_fp: float = 1.0
) -> Dict[str, Any]:
    """
    Computes metrics for binary escalation decision (AUTO_HANDLE vs ESCALATE).
    Positive class = ESCALATE.
    
    In customer service operations, False Negatives (failing to escalate a high-risk
    or frustrated customer to a human) are significantly more damaging than False Positives
    (unnecessarily handing off a resolvable issue to a human agent).
    """
    pos_label = "ESCALATE"
    neg_label = "AUTO_HANDLE"
    
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp == pos_label)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == neg_label and yp == neg_label)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == neg_label and yp == pos_label)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp == neg_label)
    
    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Asymmetric business cost: Cost = c_fn * FN + c_fp * FP
    total_cost = (cost_fn * fn) + (cost_fp * fp)
    normalized_cost = total_cost / total if total > 0 else 0.0
    
    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_negative_rate": round(fnr, 4),
        "false_positive_rate": round(fpr, 4),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "total_cost": round(total_cost, 2),
        "normalized_cost_per_ticket": round(normalized_cost, 4)
    }

def compute_reply_lexical_metrics(
    predictions: List[str],
    references: List[str]
) -> Dict[str, Any]:
    """
    Computes lexical similarity (token overlap, Jaccard, length ratio)
    between generated replies and ground truth historical agent resolutions.
    """
    if not predictions or not references:
        return {"avg_jaccard": 0.0, "avg_pred_len": 0, "avg_ref_len": 0}
        
    jaccards = []
    pred_lens = []
    ref_lens = []
    
    for pred, ref in zip(predictions, references):
        p_tokens = set(pred.lower().split())
        r_tokens = set(ref.lower().split())
        
        pred_lens.append(len(pred.split()))
        ref_lens.append(len(ref.split()))
        
        intersection = len(p_tokens & r_tokens)
        union = len(p_tokens | r_tokens)
        jaccard = intersection / union if union > 0 else 0.0
        jaccards.append(jaccard)
        
    return {
        "avg_jaccard_similarity": round(float(np.mean(jaccards)), 4),
        "avg_pred_word_count": round(float(np.mean(pred_lens)), 1),
        "avg_ref_word_count": round(float(np.mean(ref_lens)), 1),
    }
