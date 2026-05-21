"""Classical ML baselines for benchmark comparison.

Owner: Jingyuan Wang (Evaluation & Uncertainty)

We compare the Bayesian Network against three discriminative baselines
trained on one-hot-encoded discrete features (so all models see the
same input representation):

* Logistic Regression  — strong interpretable baseline.
* Random Forest        — non-linear, lightly regularized.
* XGBoost              — gradient boosting, usually the SOTA on UCI.

Each baseline exposes a sklearn-style ``.predict_proba``. We expose a
single ``train_baselines`` entry point that returns a dict of fitted
``Pipeline`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class BaselineConfig:
    random_state: int = 42


def _make_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), feature_cols)]
    )


def train_baselines(
    train: pd.DataFrame,
    target: str = "target",
    cfg: BaselineConfig | None = None,
) -> dict[str, Pipeline]:
    """Fit and return three sklearn baselines + (optionally) XGBoost."""
    cfg = cfg or BaselineConfig()
    feature_cols = [c for c in train.columns if c != target]
    X = train[feature_cols]
    y = train[target].astype(int) if train[target].dtype != int else train[target]

    pre = _make_preprocessor(feature_cols)
    models: dict[str, Pipeline] = {
        "LogisticRegression": Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        C=1.0,
                        random_state=cfg.random_state,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("pre", _make_preprocessor(feature_cols)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=None,
                        min_samples_leaf=2,
                        random_state=cfg.random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier  # may fail at import OR at first call

        xgb_pipe = Pipeline(
            [
                ("pre", _make_preprocessor(feature_cols)),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        eval_metric="logloss",
                        random_state=cfg.random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        # Touch the underlying library to surface OSError (e.g. missing libomp)
        # *before* we add it to the model dict.
        xgb_pipe.fit(X, y.astype(int))
        models["XGBoost"] = xgb_pipe
    except (ImportError, OSError, Exception) as exc:  # pragma: no cover
        import warnings as _w
        _w.warn(f"XGBoost baseline unavailable ({exc}); skipping.")

    for name, pipe in models.items():
        if name == "XGBoost":
            continue  # already fitted above
        pipe.fit(X, y.astype(int))
    return models


def predict_proba(
    models: dict[str, Pipeline],
    df: pd.DataFrame,
    target: str = "target",
) -> dict[str, np.ndarray]:
    """Return per-model P(target=1 | features) on rows of ``df``."""
    feature_cols = [c for c in df.columns if c != target]
    out: dict[str, np.ndarray] = {}
    for name, pipe in models.items():
        proba = pipe.predict_proba(df[feature_cols])[:, 1]
        out[name] = proba
    return out
