"""Epistemic uncertainty via posterior sampling over CPDs.

Owner: Jingyuan Wang (Evaluation & Uncertainty)

With a Bayesian estimator and a Dirichlet prior, the posterior over
each CPD column is itself a Dirichlet. We can sample several plausible
parameterizations of the network from this posterior, run inference
with each, and obtain a *distribution* over P(target=1) for every
patient. The width of that distribution captures epistemic uncertainty
(uncertainty about parameters), which is distinct from the inherent
aleatoric uncertainty captured by the predicted probability itself.

For the proof-of-concept here, we use a parametric bootstrap of the
training data (a simple, well-known approximation to posterior draws
under a flat prior) so we don't have to manually reimplement Dirichlet
posterior sampling for every CPD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import networkx as nx
import numpy as np
import pandas as pd

from .inference import predict_proba
from .parameter_learning import ParameterFitConfig, fit_parameters


@dataclass
class UncertaintyConfig:
    """Configuration for posterior-sample uncertainty."""

    n_posterior_samples: int = 25
    bootstrap_size: int | None = None  # default = len(train)
    seed: int = 42
    fit_cfg: ParameterFitConfig | None = None


def posterior_predictive(
    dag: nx.DiGraph,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str = "target",
    state_names: Mapping[str, list[str]] | None = None,
    cfg: UncertaintyConfig | None = None,
) -> dict[str, np.ndarray]:
    """Compute mean and per-patient credible intervals for P(target=1).

    Returns
    -------
    dict
        Keys: ``mean``, ``std``, ``ci_low``, ``ci_high`` (all 1-D arrays
        of length ``len(test)``), and ``samples`` (n_samples × len(test)
        matrix of P(target = positive_state) values).
    """
    cfg = cfg or UncertaintyConfig()
    fit_cfg = cfg.fit_cfg or ParameterFitConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.bootstrap_size or len(train)

    sample_probas: list[np.ndarray] = []
    for _ in range(cfg.n_posterior_samples):
        idx = rng.integers(0, len(train), size=n)
        boot = train.iloc[idx].reset_index(drop=True)
        bn = fit_parameters(dag, boot, cfg=fit_cfg, state_names=state_names)
        proba_df = predict_proba(bn, test, target=target, show_progress=False)
        positive_state = sorted(bn.get_cpds(target).state_names[target])[-1]
        sample_probas.append(proba_df[positive_state].values)

    samples = np.vstack(sample_probas)  # (n_samples, n_test)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    ci_low = np.quantile(samples, 0.025, axis=0)
    ci_high = np.quantile(samples, 0.975, axis=0)

    return {
        "samples": samples,
        "mean": mean,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


# ----------------------------------------------------------------------------
# Decision theory
# ----------------------------------------------------------------------------
@dataclass
class UtilityMatrix:
    """Asymmetric cost matrix for medical decisions.

    For heart disease, a false negative (missed disease) is far more
    costly than a false positive (extra cardiology referral). We
    encode this as a utility matrix and find the Bayes-optimal
    threshold by minimizing expected cost over the test set.
    """

    cost_fp: float = 1.0   # cost of unnecessary referral
    cost_fn: float = 10.0  # cost of missing real disease
    cost_tp: float = 0.0
    cost_tn: float = 0.0

    def expected_cost(self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
        """Empirical expected cost on a held-out set at the given threshold."""
        y_pred = (y_prob >= threshold).astype(int)
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        return (
            fp * self.cost_fp
            + fn * self.cost_fn
            + tp * self.cost_tp
            + tn * self.cost_tn
        ) / len(y_true)


def optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    utility: UtilityMatrix | None = None,
    grid: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """Search a grid of thresholds for the one that minimizes expected cost.

    Also returns the full grid for plotting.
    """
    utility = utility or UtilityMatrix()
    # Use a fine grid by default so the optimum is less likely to be
    # artificially pinned to a coarse boundary such as 0.05.
    grid = grid if grid is not None else np.linspace(0.01, 0.99, 99)
    rows = []
    for t in grid:
        rows.append(
            dict(threshold=float(t), expected_cost=utility.expected_cost(y_true, y_prob, t))
        )
    df = pd.DataFrame(rows)
    best = float(df.loc[df["expected_cost"].idxmin(), "threshold"])
    return best, df
