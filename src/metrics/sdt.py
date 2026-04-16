"""
Stories 3.1–3.2 — SDT computation: d′, meta-d′, M-ratio via Type 2 AUROC.

For binary accuracy + confidence data (single-criterion task without explicit
type 1 response separation), the M-ratio is estimated via the Type 2 AUROC:

    Type2_AUROC = P(confidence_correct > confidence_incorrect)
    meta_d'     = 2 * Φ⁻¹(Type2_AUROC)
    d'          = 2 * Φ⁻¹(mean_accuracy)
    M-ratio     = meta_d' / d'

This formulation (Fleming & Lau, 2014) is appropriate for recognition/categorisation
tasks where the type 1 stimulus and response cannot be separated.

Input:
  accuracy_vector:   binary list/array (1=correct, 0=incorrect) from Run 1 judge
  confidence_vector: integer 0-100 array from Run 3

Output: dict with d_prime, meta_d_prime, m_ratio, and diagnostics
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


def compute_m_ratio(
    accuracy_vector: list[int] | np.ndarray,
    confidence_vector: list[float] | np.ndarray,
    min_d_prime: float = 0.1,
    min_trials: int = 50,
) -> dict[str, float]:
    """
    Compute d′, meta-d′, and M-ratio via Type 2 AUROC estimation.

    Parameters
    ----------
    accuracy_vector :   binary (0/1), Run 1 judge scores
    confidence_vector : 0-100 floats, Run 3 confidence scores
    min_d_prime :       floor below which M-ratio is set to 0 (degenerate model)
    min_trials :        minimum number of trials required for reliable estimate

    Returns
    -------
    dict with keys: d_prime, meta_d_prime, m_ratio, n_trials,
                    n_correct, mean_confidence, conf_std, reliable
    """
    accuracy = np.array(accuracy_vector, dtype=float)
    confidence = np.array(confidence_vector, dtype=float) / 100.0  # normalise to [0,1]

    n_trials = len(accuracy)
    n_correct = int(accuracy.sum())
    mean_confidence = float(confidence.mean()) * 100.0  # return on original 0-100 scale
    conf_std = float((confidence * 100.0).std())

    if n_trials < min_trials:
        warnings.warn(
            f"Only {n_trials} trials — M-ratio unreliable below {min_trials}. "
            "Flag this model in leaderboard output.",
            stacklevel=2,
        )

    # --- Type 1 d' from accuracy --------------------------------------------
    mean_acc = float(accuracy.mean())
    mean_acc_clipped = np.clip(mean_acc, 0.01, 0.99)
    d_prime = float(2.0 * norm.ppf(mean_acc_clipped))

    # --- Type 2 AUROC → meta-d' ---------------------------------------------
    n_pos = int(accuracy.sum())
    n_neg = n_trials - n_pos
    if n_pos == 0 or n_neg == 0:
        warnings.warn(
            "All trials correct or all incorrect — meta-d' undefined. "
            "Setting M-ratio to 0.",
            stacklevel=2,
        )
        type2_auroc = 0.5
    else:
        type2_auroc = float(roc_auc_score(accuracy, confidence))

    type2_auroc_clipped = np.clip(type2_auroc, 0.01, 0.99)
    meta_d_prime = float(2.0 * norm.ppf(type2_auroc_clipped))

    # --- M-ratio floor -------------------------------------------------------
    if abs(d_prime) < min_d_prime:
        m_ratio = 0.0
    else:
        m_ratio = meta_d_prime / d_prime

    reliable = (
        n_trials >= min_trials
        and abs(d_prime) >= min_d_prime
        and n_pos >= 5
        and n_neg >= 5
    )

    return {
        "d_prime":         round(d_prime, 4),
        "meta_d_prime":    round(meta_d_prime, 4),
        "m_ratio":         round(m_ratio, 4),
        "type2_auroc":     round(type2_auroc, 4),
        "n_trials":        n_trials,
        "n_correct":       n_correct,
        "mean_confidence": round(mean_confidence, 2),
        "conf_std":        round(conf_std, 2),
        "reliable":        reliable,
    }
