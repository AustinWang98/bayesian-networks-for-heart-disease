"""Evaluation: discrimination, calibration, probabilistic scoring.

Owner: Member 4 (Evaluation & Uncertainty)

In medical risk modeling, *probability quality* matters as much as
classification accuracy. We therefore report:

* **Discrimination**: accuracy, ROC-AUC, average precision, F1.
* **Calibration**: Brier score, expected calibration error (ECE),
  reliability diagrams.
* **Probabilistic fit**: negative log-likelihood (log loss).

Everything below operates on a 1-D array of P(target=1) plus a 1-D
array of ground-truth labels ∈ {0,1}.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------
def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error (Guo et al., 2017)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_prob)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Return a per-bin reliability table (for reliability diagrams)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append(
                dict(bin_lo=lo, bin_hi=hi, count=0, avg_predicted=np.nan, frac_positive=np.nan)
            )
            continue
        rows.append(
            dict(
                bin_lo=lo,
                bin_hi=hi,
                count=n,
                avg_predicted=float(y_prob[mask].mean()),
                frac_positive=float(y_true[mask].mean()),
            )
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Aggregate report
# ----------------------------------------------------------------------------
@dataclass
class MetricBundle:
    accuracy: float
    f1: float
    roc_auc: float
    avg_precision: float
    brier: float
    log_loss: float
    ece: float

    def as_series(self) -> pd.Series:
        return pd.Series(asdict(self))


def score(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> MetricBundle:
    """Compute the full metric bundle for a single model."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob), 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)
    return MetricBundle(
        accuracy=accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_prob),
        avg_precision=average_precision_score(y_true, y_prob),
        brier=brier_score_loss(y_true, y_prob),
        log_loss=log_loss(y_true, y_prob),
        ece=expected_calibration_error(y_true, y_prob),
    )


def benchmark(
    y_true: np.ndarray, probas: dict[str, np.ndarray], threshold: float = 0.5
) -> pd.DataFrame:
    """Return a side-by-side benchmark of all models in ``probas``."""
    rows = {name: score(y_true, p, threshold).as_series() for name, p in probas.items()}
    df = pd.DataFrame(rows).T
    df = df.sort_values("roc_auc", ascending=False)
    return df
