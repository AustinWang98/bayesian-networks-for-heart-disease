"""CardioRisk Copilot - simplified Streamlit MVP demo.

Run locally:

    streamlit run app/streamlit_app.py

Or from the repo root:

    ./run_demo.sh
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP_DIR))

from demo_utils import (  # noqa: E402
    FEATURE_LABELS,
    PRESET_PROFILES,
    DemoBundle,
    format_option,
    inject_custom_css,
    load_demo_bundle,
    plotly_gauge,
    risk_band,
)
from src.inference import posterior  # noqa: E402


SIMPLE_INPUT_FIELDS = ["age", "sex", "cp", "trestbps", "chol", "thalach", "exang"]
CARE_ACTION_FIELDS = ["chol", "trestbps", "thalach", "exang"]


def feature_columns(bundle: DemoBundle) -> list[str]:
    return [node for node in bundle.bn.nodes() if node != "target"]


def simple_patient_input(bundle: DemoBundle) -> dict[str, str]:
    """Collect the small set of fields an MVP demo audience can understand quickly."""
    profile_names = list(PRESET_PROFILES)
    current_profile = st.session_state.get("simple_profile", "Moderate risk (example)")
    profile_index = profile_names.index(current_profile) if current_profile in profile_names else 1

    profile = st.selectbox(
        "Start with an example patient",
        profile_names,
        index=profile_index,
        key="simple_profile",
    )

    evidence = {
        col: PRESET_PROFILES[profile].get(col, bundle.states[col][0])
        for col in feature_columns(bundle)
    }

    st.caption("Edit a few familiar fields. The remaining chart fields stay at the selected example profile.")
    cols = st.columns(2)
    for idx, col in enumerate(SIMPLE_INPUT_FIELDS):
        if col not in evidence:
            continue
        with cols[idx % 2]:
            options = bundle.states[col]
            default = evidence[col] if evidence[col] in options else options[0]
            evidence[col] = st.selectbox(
                FEATURE_LABELS.get(col, col),
                options,
                index=options.index(default),
                key=f"simple_{profile}_{col}",
                format_func=lambda v, variable=col: format_option(variable, v),
            )
    return evidence


def risk_probability(bundle: DemoBundle, evidence: dict[str, str]) -> tuple[pd.Series, float]:
    post = posterior(bundle.engine, "target", evidence)
    return post, float(post.loc[bundle.positive_state])


def evidence_impact(bundle: DemoBundle, evidence: dict[str, str]) -> pd.DataFrame:
    """Estimate each field's directional contribution by withholding it once."""
    _, full_risk = risk_probability(bundle, evidence)
    rows = []
    for var in feature_columns(bundle):
        partial_evidence = {key: val for key, val in evidence.items() if key != var}
        _, partial_risk = risk_probability(bundle, partial_evidence)
        rows.append(
            {
                "Variable": FEATURE_LABELS.get(var, var),
                "State": format_option(var, evidence[var]),
                "Contribution": full_risk - partial_risk,
            }
        )
    df = pd.DataFrame(rows)
    return df.reindex(df["Contribution"].abs().sort_values(ascending=False).index)


def care_guidance(risk: float) -> tuple[str, str, list[str]]:
    if risk < 0.30:
        return (
            "Low priority",
            "Routine prevention",
            [
                "Keep standard prevention and annual review cadence.",
                "Use the scenario preview to show which measurements matter most.",
                "No urgent escalation signal from this model output.",
            ],
        )
    if risk < 0.60:
        return (
            "Watch closely",
            "Targeted follow-up",
            [
                "Review modifiable measurements such as blood pressure, cholesterol, and exercise response.",
                "Schedule a clinician follow-up if symptoms or recent changes are present.",
                "Use the scenario preview to prioritize the next conversation.",
            ],
        )
    return (
        "High priority",
        "Clinician review",
        [
            "Escalate for timely clinical review and confirmatory testing.",
            "Focus the visit on the strongest risk drivers in this chart.",
            "Track whether planned interventions move the modeled risk band.",
        ],
    )


def render_care_guidance(risk: float) -> None:
    priority, workflow, steps = care_guidance(risk)
    st.markdown(
        f"""
<div class="answer-panel">
  <strong>Suggested workflow</strong><br>
  {priority}: <strong>{workflow}</strong>
</div>
        """,
        unsafe_allow_html=True,
    )
    for step in steps:
        st.write(f"- {step}")


def render_top_driver_table(bundle: DemoBundle, evidence: dict[str, str]) -> None:
    drivers = evidence_impact(bundle, evidence).head(4).copy()
    drivers["Impact"] = drivers["Contribution"].map(
        lambda v: f"Raises estimate by {v:.1%}" if v >= 0 else f"Lowers estimate by {abs(v):.1%}"
    )
    st.dataframe(
        drivers.rename(columns={"Variable": "Top driver", "State": "Current value"})[
            ["Top driver", "Current value", "Impact"]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_care_action_preview(bundle: DemoBundle, evidence: dict[str, str], current_risk: float) -> None:
    lever = st.selectbox(
        "Care action to preview",
        [field for field in CARE_ACTION_FIELDS if field in evidence],
        format_func=lambda x: FEATURE_LABELS.get(x, x),
        key="simple_care_lever",
    )
    options = bundle.states[lever]
    current = evidence[lever] if evidence[lever] in options else options[0]
    goal = st.selectbox(
        "Goal value",
        options,
        index=options.index(current),
        key=f"simple_goal_{lever}",
        format_func=lambda v, variable=lever: format_option(variable, v),
    )

    scenario_evidence = {**evidence, lever: goal}
    _, projected = risk_probability(bundle, scenario_evidence)

    metric_cols = st.columns(2)
    metric_cols[0].metric("Current estimate", f"{current_risk:.1%}")
    metric_cols[1].metric("Scenario estimate", f"{projected:.1%}", delta=f"{projected - current_risk:+.1%}")

    fig = go.Figure(
        go.Bar(
            x=["Current", "Scenario"],
            y=[current_risk, projected],
            marker_color=["#215c63", "#c2410c"],
            text=[f"{current_risk:.1%}", f"{projected:.1%}"],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=20, t=20, b=30),
        yaxis=dict(tickformat=".0%", range=[0, min(1.05, max(current_risk, projected) + 0.15)]),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Scenario estimates keep the rest of this patient profile fixed.")


def render_mvp_snapshot(bundle: DemoBundle) -> None:
    st.subheader("MVP snapshot")
    s1, s2, s3 = st.columns(3)
    s1.markdown(
        f'<div class="metric-card"><strong>{len(bundle.train)}</strong>'
        '<span style="color:#334155;">training patients</span></div>',
        unsafe_allow_html=True,
    )
    s2.markdown(
        f'<div class="metric-card"><strong>{len(bundle.test)}</strong>'
        '<span style="color:#334155;">held-out patients</span></div>',
        unsafe_allow_html=True,
    )
    s3.markdown(
        '<div class="metric-card"><strong>Bayesian</strong>'
        '<span style="color:#334155;">transparent risk model</span></div>',
        unsafe_allow_html=True,
    )


def page_home(bundle: DemoBundle) -> None:
    st.markdown(
        """
<div class="hero">
  <h1>CardioRisk Copilot</h1>
  <p>A simple decision-support MVP: estimate risk, explain the main drivers, and preview one care action.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    input_col, answer_col = st.columns([0.95, 1.05])
    with input_col:
        st.subheader("Patient snapshot")
        evidence = simple_patient_input(bundle)

    _, risk = risk_probability(bundle, evidence)
    band, klass, _ = risk_band(risk)

    with answer_col:
        st.markdown(
            f"""
<div class="answer-panel">
  <strong>Care-team answer</strong><br>
  Estimated heart-disease risk from this chart: <strong>{risk:.1%}</strong>
  <span class="{klass}">({band})</span>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(plotly_gauge(risk), use_container_width=True)
        render_care_guidance(risk)

    driver_col, scenario_col = st.columns([1, 1])
    with driver_col:
        st.subheader("Why this estimate")
        render_top_driver_table(bundle, evidence)
    with scenario_col:
        st.subheader("Preview one care action")
        render_care_action_preview(bundle, evidence, risk)

    st.markdown(
        """
<div class="answer-panel small-note">
  Methodology remains the same: a transparent probabilistic model trained on the UCI
  Cleveland heart-disease cohort. The MVP keeps the user experience focused on a care-team
  workflow rather than research details.
</div>
        """,
        unsafe_allow_html=True,
    )
    render_mvp_snapshot(bundle)


def main() -> None:
    st.set_page_config(
        page_title="CardioRisk Copilot",
        page_icon="heart",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_custom_css()
    bundle = load_demo_bundle()
    page_home(bundle)


if __name__ == "__main__":
    main()
