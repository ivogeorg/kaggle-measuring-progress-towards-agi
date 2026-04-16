"""
Story 3.5 — Metacognitive Capability Index (MCI).

MCI = α · max(0, M-ratio) + β · SRS − γ · ECE

α = 0.65  (primacy of introspective accuracy)
β = 0.35  (outward attribution capability)
γ = dynamic (penalises near-constant confidence output)
"""

from __future__ import annotations

import numpy as np

from src.metrics.ece import compute_dynamic_gamma, compute_ece
from src.metrics.sdt import compute_m_ratio

ALPHA = 0.65
BETA = 0.35


def compute_mci(
    accuracy_vector: list[int],
    confidence_vector: list[float],
    srs: float,
) -> dict[str, float]:
    """
    Compute the full MCI score for a single model.

    Parameters
    ----------
    accuracy_vector :   binary (0/1) from Run 1 judge, length N
    confidence_vector : 0-100 from Run 3, length N (aligned with accuracy_vector)
    srs :               Self-Recognition Score from Run 2 (float 0.0–1.0)

    Returns
    -------
    dict with mci, m_ratio, d_prime, meta_d_prime, srs, ece, gamma, and diagnostics
    """
    sdt = compute_m_ratio(accuracy_vector, confidence_vector)
    ece = compute_ece(confidence_vector, accuracy_vector)
    gamma = compute_dynamic_gamma(confidence_vector)

    m_ratio = sdt["m_ratio"]
    mci = ALPHA * max(0.0, m_ratio) + BETA * srs - gamma * ece

    return {
        "mci":           round(mci, 4),
        "m_ratio":       sdt["m_ratio"],
        "d_prime":       sdt["d_prime"],
        "meta_d_prime":  sdt["meta_d_prime"],
        "type2_auroc":   sdt["type2_auroc"],
        "srs":           round(srs, 4),
        "ece":           round(ece, 6),
        "gamma":         gamma,
        "n_trials":      sdt["n_trials"],
        "n_correct":     sdt["n_correct"],
        "mean_confidence": sdt["mean_confidence"],
        "conf_std":      sdt["conf_std"],
        "reliable":      sdt["reliable"],
    }


def build_leaderboard(results_by_model: dict[str, dict]) -> "pd.DataFrame":
    """
    Build a sorted leaderboard DataFrame from per-model MCI dicts.

    Parameters
    ----------
    results_by_model : { model_name: compute_mci() output dict }
    """
    import pandas as pd

    rows = []
    for model_name, metrics in results_by_model.items():
        row = {"model": model_name, **metrics}
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("mci", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-based rank
    df.index.name = "rank"
    return df
