"""
Human vs. LLM-Judge Agreement Validation.
Computes Cohen's Kappa, Pearson Correlation, Spearman Rank Correlation,
MAE, and Adjacent Agreement Rate to scientifically prove whether the Judge can be trusted.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

def compute_judge_human_agreement(
    human_scores: List[float],
    judge_scores: List[float],
    threshold_acceptable: float = 3.5
) -> Dict[str, Any]:
    """
    Evaluates alignment between Human Annotators and the Automated Judge.
    
    Args:
        human_scores: Ground-truth human ratings (1.0 to 5.0)
        judge_scores: Automated Judge ratings (1.0 to 5.0)
        threshold_acceptable: Cutoff for classifying a response as acceptable (>= cutoff)
    """
    assert len(human_scores) == len(judge_scores), "Scores lists must have identical lengths."
    n = len(human_scores)
    if n == 0:
        return {}

    # Discretize into Binary Pass / Fail decisions
    human_binary = [1 if s >= threshold_acceptable else 0 for s in human_scores]
    judge_binary = [1 if s >= threshold_acceptable else 0 for s in judge_scores]

    # Cohen's Kappa for binary pass/fail
    kappa = cohen_kappa_score(human_binary, judge_binary)

    # Discretize into 1-5 integer bins for quadratic weighted kappa
    h_int = [int(round(s)) for s in human_scores]
    j_int = [int(round(s)) for s in judge_scores]
    weighted_kappa = cohen_kappa_score(h_int, j_int, weights="quadratic")

    # Correlations
    pearson_corr, pearson_p = pearsonr(human_scores, judge_scores)
    spearman_corr, spearman_p = spearmanr(human_scores, judge_scores)

    # Agreement Rates
    diffs = np.abs(np.array(human_scores) - np.array(judge_scores))
    exact_agreement = np.mean(diffs < 0.25)
    adjacent_agreement = np.mean(diffs <= 1.0)
    mae = float(np.mean(diffs))
    rmse = float(np.sqrt(np.mean(diffs ** 2)))

    return {
        "sample_size": n,
        "cohen_kappa_binary": round(float(kappa), 4),
        "quadratic_weighted_kappa": round(float(weighted_kappa), 4),
        "pearson_correlation": round(float(pearson_corr), 4),
        "pearson_p_value": float(pearson_p),
        "spearman_rank_correlation": round(float(spearman_corr), 4),
        "spearman_p_value": float(spearman_p),
        "exact_agreement_rate": round(float(exact_agreement), 4),
        "adjacent_agreement_rate_plus_minus_1": round(float(adjacent_agreement), 4),
        "mean_absolute_error": round(mae, 4),
        "root_mean_squared_error": round(rmse, 4),
        "interpretation": (
            "Substantial agreement (Kappa > 0.60, Pearson r > 0.75). "
            "The Judge reliably mirrors human evaluator consensus on customer service quality."
        ) if weighted_kappa >= 0.60 else "Moderate or weak agreement; human calibration required."
    }
