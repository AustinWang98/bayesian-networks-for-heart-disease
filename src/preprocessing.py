"""Preprocessing pipeline: imputation, discretization, splitting.

Owner: Yiou Wang (Data & EDA)

Bayesian Networks in pgmpy operate on **discrete** variables, so we use
domain-motivated binning for continuous columns. The bin edges are
chosen from medical guidelines where possible (e.g. ATP-III thresholds
for cholesterol; JNC-7 thresholds for blood pressure) rather than from
purely data-driven quantile cuts. This keeps the network interpretable
to clinicians while still being supported by data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ----- Bin schema (clinically motivated) -------------------------------------
# Each entry: (bin_edges, bin_labels). Edges use +/- inf to guarantee coverage.
DISCRETIZATION: dict[str, tuple[list[float], list[str]]] = {
    "age": (
        [-np.inf, 45, 55, 65, np.inf],
        ["<45", "45-54", "55-64", "65+"],
    ),
    "trestbps": (
        # JNC-7: normal <120, prehypertension 120-139, hypertensive >=140.
        [-np.inf, 120, 140, np.inf],
        ["normal", "prehyper", "hyper"],
    ),
    "chol": (
        # ATP-III: desirable <200, borderline 200-239, high >=240.
        [-np.inf, 200, 240, np.inf],
        ["desirable", "borderline", "high"],
    ),
    "thalach": (
        # Max heart rate tertiles relative to typical 220-age envelope.
        [-np.inf, 140, 170, np.inf],
        ["low", "mid", "high"],
    ),
    "oldpeak": (
        # ST depression: clinically <1 normal, 1-2 mild, >2 marked.
        [-np.inf, 1.0, 2.0, np.inf],
        ["none", "mild", "marked"],
    ),
}


# Variables that are already discrete in the source data.
CATEGORICAL_VARS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


@dataclass
class PreprocessConfig:
    """Configuration for the preprocessing pipeline."""

    binarize_target: bool = True  # Collapse num∈{0,1,2,3,4} into {0,1}
    test_size: float = 0.2
    random_state: int = 42
    drop_na_thal_ca: bool = True  # The 6 rows with '?' in thal/ca are dropped


# ----- Public API ------------------------------------------------------------
def impute(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """Impute the handful of missing values in `ca` and `thal`.

    The Cleveland subset has 4 missing values in `ca` and 2 in `thal`.
    These are conventionally either dropped or imputed with the mode.
    """
    df = df.copy()
    df["ca"] = pd.to_numeric(df["ca"], errors="coerce")
    df["thal"] = pd.to_numeric(df["thal"], errors="coerce")

    if cfg.drop_na_thal_ca:
        df = df.dropna(subset=["ca", "thal"]).reset_index(drop=True)
    else:
        df["ca"] = df["ca"].fillna(df["ca"].mode().iloc[0])
        df["thal"] = df["thal"].fillna(df["thal"].mode().iloc[0])
    return df


def discretize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clinically-motivated binning to continuous variables."""
    df = df.copy()
    for col, (edges, labels) in DISCRETIZATION.items():
        df[col] = pd.cut(df[col], bins=edges, labels=labels, include_lowest=True)
    # Cast remaining numeric categoricals to string labels so pgmpy treats
    # them as discrete states (not orderable floats).
    for col in CATEGORICAL_VARS:
        df[col] = df[col].astype(int).astype(str)
    return df


def prepare_target(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """Map the multi-class severity score to a binary disease indicator."""
    df = df.copy()
    if cfg.binarize_target:
        df["target"] = (df["num"].astype(int) > 0).astype(int).astype(str)
    else:
        df["target"] = df["num"].astype(int).astype(str)
    df = df.drop(columns=["num"])
    return df


def build_dataset(
    df_raw: pd.DataFrame, cfg: PreprocessConfig | None = None
) -> pd.DataFrame:
    """End-to-end preprocessing: imputation + discretization + relabel target."""
    cfg = cfg or PreprocessConfig()
    df = impute(df_raw, cfg)
    df = discretize(df)
    df = prepare_target(df, cfg)
    # pgmpy >= 1.0 requires every variable to be a pandas Categorical
    # (it does not accept the modern `str` dtype). We materialize the
    # categories explicitly and force them unordered so pgmpy treats
    # them as unordered discrete states.
    df = df.apply(lambda s: pd.Categorical(s.astype(str), ordered=False))
    df = pd.DataFrame({c: pd.Categorical(df[c], ordered=False) for c in df.columns})
    return df


def train_test_split_df(
    df: pd.DataFrame, cfg: PreprocessConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split honoring the (possibly binarized) target."""
    cfg = cfg or PreprocessConfig()
    train, test = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=df["target"],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def variable_state_names(df: pd.DataFrame) -> Mapping[str, list[str]]:
    """Collect the observed state set per variable.

    Required by pgmpy to lock down state ordering so that CPDs learned
    on the training fold are evaluable on the test fold even if the
    test fold happens to miss a particular state.
    """
    out: dict[str, list[str]] = {}
    for col in df.columns:
        vals = df[col].astype(str).unique().tolist()
        out[col] = sorted(vals)
    return out
