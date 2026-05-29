"""Shared helpers for the CardioRisk Copilot Streamlit MVP."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_heart_disease
from src.expert_network import build_expert_dag
from src.inference import make_engine
from src.parameter_learning import ParameterFitConfig, fit_parameters
from src.preprocessing import (
    PreprocessConfig,
    build_dataset,
    train_test_split_df,
    variable_state_names,
)


FEATURE_LABELS: dict[str, str] = {
    "age": "Age group",
    "sex": "Sex (1 = male)",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar > 120",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate (exercise)",
    "exang": "Exercise angina",
    "oldpeak": "ST depression",
    "slope": "ST slope",
    "ca": "Major vessels",
    "thal": "Thallium stress test",
    "target": "Heart disease",
}

VALUE_HINTS: dict[str, dict[str, str]] = {
    "cp": {"1": "Typical angina", "2": "Atypical", "3": "Non-anginal", "4": "Asymptomatic"},
    "sex": {"0": "Female", "1": "Male"},
    "fbs": {"0": "No", "1": "Yes"},
    "exang": {"0": "No", "1": "Yes"},
    "thal": {"3": "Normal", "6": "Fixed defect", "7": "Reversible defect"},
}

PRESET_PROFILES: dict[str, dict[str, str]] = {
    "Low risk (example)": {
        "age": "<45",
        "sex": "0",
        "cp": "3",
        "trestbps": "normal",
        "chol": "desirable",
        "fbs": "0",
        "restecg": "0",
        "thalach": "high",
        "exang": "0",
        "oldpeak": "none",
        "slope": "1",
        "ca": "0",
        "thal": "3",
    },
    "Moderate risk (example)": {
        "age": "55-64",
        "sex": "1",
        "cp": "2",
        "trestbps": "prehyper",
        "chol": "borderline",
        "fbs": "0",
        "restecg": "1",
        "thalach": "mid",
        "exang": "0",
        "oldpeak": "mild",
        "slope": "2",
        "ca": "1",
        "thal": "3",
    },
    "High risk (example)": {
        "age": "65+",
        "sex": "1",
        "cp": "4",
        "trestbps": "hyper",
        "chol": "high",
        "fbs": "1",
        "restecg": "2",
        "thalach": "low",
        "exang": "1",
        "oldpeak": "marked",
        "slope": "3",
        "ca": "3",
        "thal": "7",
    },
}


@dataclass
class DemoBundle:
    bn: Any
    engine: Any
    states: dict[str, list[str]]
    train: pd.DataFrame
    test: pd.DataFrame
    positive_state: str


def inject_custom_css() -> None:
    st.markdown(
        """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main .block-container { padding-top: 1.25rem; max-width: 1180px; }
    .hero {
        background: linear-gradient(135deg, #12332f 0%, #215c63 52%, #6e6a2f 100%);
        border-radius: 8px; padding: 1.35rem 1.5rem; color: #fff;
        margin-bottom: 1rem; box-shadow: 0 8px 26px rgba(18,51,47,0.22);
    }
    .hero h1 { color: #fff !important; font-size: 1.95rem !important; margin-bottom: 0.25rem; }
    .hero p { color: rgba(255,255,255,0.9); font-size: 1rem; margin: 0; }
    .metric-card {
        background: #f8fafc; border: 1px solid #d9e2e8; border-radius: 8px;
        padding: 0.85rem 1rem; text-align: center; color: #334155 !important;
    }
    .metric-card * { color: #334155 !important; }
    .metric-card strong { font-size: 1.45rem; color: #12332f !important; display: block; }
    .answer-panel {
        border: 1px solid #d9e2e8; border-radius: 8px; padding: 0.85rem 1rem;
        background: #fbfcfd; margin-bottom: 0.75rem;
    }
    .answer-panel strong { color: #12332f; }
    .small-note {
        color: #52606d; font-size: 0.9rem; line-height: 1.35;
    }
    .risk-low { color: #16a34a; }
    .risk-mid { color: #ca8a04; }
    .risk-high { color: #dc2626; }
</style>
        """,
        unsafe_allow_html=True,
    )


def risk_band(prob: float) -> tuple[str, str, str]:
    if prob < 0.30:
        return "Low", "risk-low", "#16a34a"
    if prob < 0.60:
        return "Moderate", "risk-mid", "#ca8a04"
    return "High", "risk-high", "#dc2626"


@st.cache_resource(show_spinner="Loading CardioRisk model...")
def load_demo_bundle() -> DemoBundle:
    cfg = PreprocessConfig()
    raw = load_heart_disease()
    df = build_dataset(raw, cfg)
    train, test = train_test_split_df(df, cfg)
    states = variable_state_names(df)
    dag = build_expert_dag()
    bn = fit_parameters(
        dag, train, cfg=ParameterFitConfig(method="bayes"), state_names=states
    )
    return DemoBundle(
        bn=bn,
        engine=make_engine(bn),
        states=states,
        train=train,
        test=test,
        positive_state=sorted(states["target"])[-1],
    )


def plotly_gauge(risk: float, title: str = "P(heart disease)") -> go.Figure:
    _, _, color = risk_band(risk)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk * 100,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "#dcfce7"},
                    {"range": [30, 60], "color": "#fef9c3"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#1e293b", "width": 3},
                    "thickness": 0.8,
                    "value": risk * 100,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def format_option(var: str, val: str) -> str:
    hint = VALUE_HINTS.get(var, {}).get(val, "")
    label = FEATURE_LABELS.get(var, var)
    return f"{label}: {val}" + (f" ({hint})" if hint else "")
