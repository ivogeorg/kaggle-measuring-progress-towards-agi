"""
Story 3.4 — ECE computation with dynamic gamma.

ECE = average |mean_confidence_in_bin - accuracy_in_bin| weighted by bin size.
Dynamic gamma: exponentially penalises models with near-constant confidence output.
"""

from __future__ import annotations

import numpy as np


def compute_ece(
    confidence_scores: list[float] | np.ndarray,
    accuracy_scores: list[int] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Standard equal-width ECE on [0,1] normalised confidence."""
    conf = np.array(confidence_scores, dtype=float) / 100.0  # normalise to [0,1]
    acc = np.array(accuracy_scores, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)

    for i in range(n_bins):
        in_bin = (conf >= bin_edges[i]) & (conf < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin |= (conf == 1.0)  # include upper edge in last bin
        n_bin = in_bin.sum()
        if n_bin == 0:
            continue
        bin_conf = conf[in_bin].mean()
        bin_acc = acc[in_bin].mean()
        ece += (n_bin / n) * abs(bin_conf - bin_acc)

    return float(round(ece, 6))


def compute_dynamic_gamma(confidence_scores: list[float] | np.ndarray) -> float:
    """
    Scale gamma exponentially when confidence variance is near-zero.

    Rationale: a model that always outputs ~100% confidence has zero discriminative
    metacognitive signal. ECE alone would reward it if it happens to be right often.
    Dynamic gamma prevents this by amplifying the ECE penalty.

    Returns gamma in range [0.05, 0.30].
    """
    conf_std = float(np.std(confidence_scores))
    VARIANCE_THRESHOLD = 5.0   # std below this triggers penalty

    if conf_std < VARIANCE_THRESHOLD:
        gamma = 0.30 * np.exp(-conf_std / VARIANCE_THRESHOLD)
        gamma = max(gamma, 0.05)  # floor
    else:
        gamma = 0.05  # standard minor penalty for well-calibrated models

    return round(float(gamma), 4)
