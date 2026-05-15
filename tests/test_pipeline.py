"""Light smoke tests so future teammates can verify the pipeline runs.

These do *not* require an internet connection — they synthesize a tiny
discrete dataframe with the same schema as the preprocessed heart
disease data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.expert_network import build_expert_dag
from src.inference import do_intervention, make_engine, posterior, predict_proba
from src.parameter_learning import ParameterFitConfig, fit_parameters
from src.preprocessing import variable_state_names


def _make_synthetic_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    columns = {
        "age": rng.choice(["<45", "45-54", "55-64", "65+"], size=n),
        "sex": rng.choice(["0", "1"], size=n),
        "cp": rng.choice(["1", "2", "3", "4"], size=n),
        "trestbps": rng.choice(["normal", "prehyper", "hyper"], size=n),
        "chol": rng.choice(["desirable", "borderline", "high"], size=n),
        "fbs": rng.choice(["0", "1"], size=n),
        "restecg": rng.choice(["0", "1", "2"], size=n),
        "thalach": rng.choice(["low", "mid", "high"], size=n),
        "exang": rng.choice(["0", "1"], size=n),
        "oldpeak": rng.choice(["none", "mild", "marked"], size=n),
        "slope": rng.choice(["1", "2", "3"], size=n),
        "ca": rng.choice(["0", "1", "2", "3"], size=n),
        "thal": rng.choice(["3", "6", "7"], size=n),
        "target": rng.choice(["0", "1"], size=n),
    }
    df = pd.DataFrame(columns)
    return pd.DataFrame({c: pd.Categorical(df[c].astype(str), ordered=False) for c in df.columns})


def test_fit_and_predict_runs():
    df = _make_synthetic_df()
    dag = build_expert_dag()
    states = variable_state_names(df)
    bn = fit_parameters(
        dag, df, cfg=ParameterFitConfig(method="bayes"), state_names=states
    )
    assert bn.number_of_nodes() == 14
    proba = predict_proba(bn, df.head(10), show_progress=False)
    assert proba.shape == (10, 2)
    np.testing.assert_allclose(proba.sum(axis=1).values, 1.0, atol=1e-6)


def test_query_returns_distribution():
    df = _make_synthetic_df()
    dag = build_expert_dag()
    states = variable_state_names(df)
    bn = fit_parameters(
        dag, df, cfg=ParameterFitConfig(method="bayes"), state_names=states
    )
    engine = make_engine(bn)
    post = posterior(engine, "target", {"age": "65+", "chol": "high"})
    assert pytest.approx(post.sum(), abs=1e-6) == 1.0
    assert set(post.index) == {"0", "1"}


def test_do_intervention_changes_marginal():
    df = _make_synthetic_df()
    dag = build_expert_dag()
    states = variable_state_names(df)
    bn = fit_parameters(
        dag, df, cfg=ParameterFitConfig(method="bayes"), state_names=states
    )
    p_obs = do_intervention(bn, {"chol": "high"})
    p_des = do_intervention(bn, {"chol": "desirable"})
    # The two interventional posteriors should not be identical.
    assert not np.allclose(p_obs.values, p_des.values)
