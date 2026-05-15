"""Data loader for the UCI Heart Disease dataset.

Owner: Member 1 (Data & EDA)

The processed Cleveland subset is the canonical benchmark used in the
Bayesian-network literature for this problem. We support two retrieval
paths, in priority order:

1. ``ucimlrepo`` — official, programmatic access to UCI ML Repository
   (dataset id 45). Cached on first call.
2. URL fallback — direct download from the UCI archive if ``ucimlrepo``
   is unavailable or blocked.

After loading, the raw frame is persisted to ``data/raw/heart.csv`` so
that subsequent runs are deterministic and offline-friendly.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical column names for the Cleveland processed file (no header).
CLEVELAND_COLUMNS: list[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

DEFAULT_RAW_PATH = Path("data/raw/heart.csv")


def _load_via_ucimlrepo() -> pd.DataFrame:
    """Try the official UCI ML Repo Python client (dataset id 45)."""
    from ucimlrepo import fetch_ucirepo  # local import: optional dep

    bundle = fetch_ucirepo(id=45)
    X = bundle.data.features.copy()
    y = bundle.data.targets.copy()
    df = pd.concat([X, y], axis=1)
    df.columns = [c.lower() for c in df.columns]
    if "num" not in df.columns and df.columns[-1] != "num":
        df = df.rename(columns={df.columns[-1]: "num"})
    return df


def _load_via_url(url: str = UCI_URL) -> pd.DataFrame:
    """Fallback: parse the raw .data file from the UCI archive."""
    import urllib.request

    with urllib.request.urlopen(url, timeout=30) as resp:
        raw_bytes = resp.read()
    df = pd.read_csv(
        io.BytesIO(raw_bytes),
        header=None,
        names=CLEVELAND_COLUMNS,
        na_values="?",
    )
    return df


def load_heart_disease(
    cache_path: Optional[Path] = DEFAULT_RAW_PATH,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return the UCI Heart Disease (Cleveland) DataFrame.

    Parameters
    ----------
    cache_path
        Local CSV destination. If it exists and ``force_refresh`` is
        False, the cached copy is read directly.
    force_refresh
        Re-download even if the cache exists.

    Returns
    -------
    pd.DataFrame
        Columns: ``age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal, num``.
    """
    cache_path = Path(cache_path) if cache_path is not None else None

    if cache_path is not None and cache_path.exists() and not force_refresh:
        logger.info("Loading cached dataset from %s", cache_path)
        return pd.read_csv(cache_path)

    df: Optional[pd.DataFrame] = None
    try:
        df = _load_via_ucimlrepo()
        logger.info("Loaded dataset via ucimlrepo (id=45).")
    except Exception as exc:  # noqa: BLE001 — graceful fallback
        logger.warning("ucimlrepo path failed (%s); falling back to URL.", exc)
        df = _load_via_url()
        logger.info("Loaded dataset via direct URL.")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info("Cached dataset to %s", cache_path)

    return df


def describe_schema() -> pd.DataFrame:
    """Return a human-readable schema description for the dataset."""
    rows = [
        ("age", "continuous", "Age in years"),
        ("sex", "binary", "1 = male; 0 = female"),
        ("cp", "categorical", "Chest pain type (1: typical angina, 2: atypical, "
                              "3: non-anginal, 4: asymptomatic)"),
        ("trestbps", "continuous", "Resting blood pressure (mm Hg)"),
        ("chol", "continuous", "Serum cholesterol (mg/dl)"),
        ("fbs", "binary", "Fasting blood sugar > 120 mg/dl"),
        ("restecg", "categorical", "Resting ECG (0: normal, 1: ST-T abnormality, "
                                   "2: LV hypertrophy)"),
        ("thalach", "continuous", "Maximum heart rate achieved"),
        ("exang", "binary", "Exercise-induced angina"),
        ("oldpeak", "continuous", "ST depression induced by exercise vs. rest"),
        ("slope", "categorical", "Slope of peak exercise ST segment (1/2/3)"),
        ("ca", "ordinal", "Number of major vessels (0–3) colored by fluoroscopy"),
        ("thal", "categorical", "3: normal, 6: fixed defect, 7: reversible defect"),
        ("num", "ordinal target", "Diagnosis 0 (no disease) … 4 (severe)"),
    ]
    return pd.DataFrame(rows, columns=["variable", "type", "description"])
