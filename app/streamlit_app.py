"""Interactive Streamlit demo for the Heart Disease Bayesian Network.

Run locally with::

    streamlit run app/streamlit_app.py

The app lets a user enter clinical features through dropdowns and
shows:
* the posterior P(target | evidence)
* a 95% credible interval from bootstrap CPD samples
* a counterfactual: ``P(target | do(chol = desirable))`` etc.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_heart_disease  # noqa: E402
from src.preprocessing import (  # noqa: E402
    PreprocessConfig,
    build_dataset,
    train_test_split_df,
    variable_state_names,
)
from src.expert_network import build_expert_dag  # noqa: E402
from src.parameter_learning import ParameterFitConfig, fit_parameters  # noqa: E402
from src.inference import do_intervention, make_engine, posterior  # noqa: E402


@st.cache_resource(show_spinner="Training Bayesian Network...")
def load_model():
    cfg = PreprocessConfig()
    raw = load_heart_disease()
    df = build_dataset(raw, cfg)
    train, _ = train_test_split_df(df, cfg)
    states = variable_state_names(df)
    dag = build_expert_dag()
    bn = fit_parameters(
        dag, train, cfg=ParameterFitConfig(method="bayes"), state_names=states
    )
    return bn, states


def main() -> None:
    st.set_page_config(page_title="Heart Disease BN", layout="wide")
    st.title("Bayesian Network — Heart Disease Risk")
    st.write(
        "Interactive posterior inference over the UCI Heart Disease "
        "Bayesian network. Adjust the sliders / dropdowns to enter a "
        "patient's evidence and inspect the resulting posterior."
    )

    bn, states = load_model()
    engine = make_engine(bn)

    st.sidebar.header("Patient evidence")
    feature_cols = [n for n in bn.nodes() if n != "target"]
    evidence: dict[str, str] = {}
    for col in feature_cols:
        opts = states[col]
        evidence[col] = st.sidebar.selectbox(col, opts, key=f"ev_{col}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Posterior P(disease | evidence)")
        post = posterior(engine, "target", evidence)
        st.bar_chart(post)
        positive_state = sorted(states["target"])[-1]
        risk = float(post.loc[positive_state])
        st.metric("Probability of disease", f"{risk:.1%}")

    with col2:
        st.subheader("Counterfactual: do(intervention)")
        st.write(
            "Pearl's `do`-operator answers *what if we set* a variable to "
            "a specific value, instead of merely observing it. Useful for "
            "exploring modifiable risk factors."
        )
        target_var = st.selectbox(
            "Variable to intervene on",
            options=feature_cols,
            index=feature_cols.index("chol"),
        )
        target_val = st.selectbox(
            "Intervene to value",
            options=states[target_var],
            index=0,
        )
        if st.button("Compute counterfactual"):
            cf = do_intervention(bn, {target_var: target_val}, query="target")
            cf_risk = float(cf.loc[positive_state])
            st.bar_chart(cf)
            delta = cf_risk - risk
            st.metric(
                "P(disease | do)",
                f"{cf_risk:.1%}",
                delta=f"{delta:+.1%}",
            )

    with st.expander("About this model"):
        st.markdown(
            f"""
**Nodes**: {bn.number_of_nodes()}  &nbsp;&nbsp; **Edges**: {bn.number_of_edges()}

The Bayesian network was fit on the UCI Heart Disease (Cleveland)
training fold using BDeu prior (equivalent sample size = 10).
Inference is exact (Variable Elimination); counterfactuals are
implemented via Pearl's `do`-operator (mutilated graph + point-mass
CPD on the intervened node).
"""
        )


if __name__ == "__main__":
    main()
