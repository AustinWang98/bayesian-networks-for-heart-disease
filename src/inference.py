"""Exact inference and batch posterior predictions.

Owner: Chenqi Wang (Parameter Learning & Inference)

We use pgmpy's Variable Elimination for exact marginal inference.
Approximate inference (Gibbs, Metropolis-Hastings) lives in
``src/mcmc.py`` so that the two can be compared cleanly.
"""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def make_engine(bn: DiscreteBayesianNetwork) -> VariableElimination:
    """Build a Variable Elimination engine for ``bn``."""
    return VariableElimination(bn)


def posterior(
    engine: VariableElimination,
    query: str,
    evidence: Mapping[str, str],
    show_progress: bool = False,
) -> pd.Series:
    """Return P(query | evidence) as a pandas Series indexed by state."""
    factor = engine.query(
        variables=[query],
        evidence=dict(evidence),
        show_progress=show_progress,
    )
    states = factor.state_names[query]
    return pd.Series(factor.values, index=states, name=f"P({query}|evidence)")


def predict_proba(
    bn: DiscreteBayesianNetwork,
    data: pd.DataFrame,
    target: str = "target",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute P(target | evidence) for every row in ``data``.

    Rows must contain all variables except ``target`` (and may contain
    extra columns, which are ignored).
    """
    engine = make_engine(bn)
    target_states = bn.get_cpds(target).state_names[target]
    feature_cols = [c for c in bn.nodes() if c != target]

    probs = np.zeros((len(data), len(target_states)))
    iterator = data[feature_cols].itertuples(index=False, name=None)
    if show_progress:
        iterator = tqdm(iterator, total=len(data), desc="VE inference")

    for i, row in enumerate(iterator):
        evidence = dict(zip(feature_cols, row))
        # Skip evidence values not present in any CPD state set: these would
        # be ignored anyway and they slow pgmpy down with warnings.
        cleaned = {
            k: v
            for k, v in evidence.items()
            if v in bn.get_cpds(k).state_names[k]
        }
        try:
            factor = engine.query([target], evidence=cleaned, show_progress=False)
            probs[i] = factor.values
        except Exception as exc:  # noqa: BLE001
            logger.warning("Inference failed on row %d (%s). Using marginal.", i, exc)
            marg = engine.query([target], show_progress=False)
            probs[i] = marg.values

    return pd.DataFrame(probs, columns=target_states)


def map_predict(
    bn: DiscreteBayesianNetwork,
    data: pd.DataFrame,
    target: str = "target",
) -> pd.Series:
    """Hard MAP prediction: argmax_y P(y | evidence)."""
    proba = predict_proba(bn, data, target=target, show_progress=False)
    return proba.idxmax(axis=1).rename(target)


def do_intervention(
    bn: DiscreteBayesianNetwork,
    intervention: Mapping[str, str],
    query: str = "target",
) -> pd.Series:
    """Pearl's ``do``-operator: cut incoming edges to intervened nodes.

    Returns P(query | do(intervention)). This goes beyond conditioning:
    it tells you what would happen *if* the variable were set
    (counterfactually), not merely if it were observed.
    """
    import copy

    bn_do = copy.deepcopy(bn)
    for node in intervention:
        # Remove incoming edges so the node is exogenous.
        parents = list(bn_do.predecessors(node))
        for p in parents:
            bn_do.remove_edge(p, node)
        # Replace its CPD with a point mass on the intervened value.
        from pgmpy.factors.discrete import TabularCPD

        states = bn.get_cpds(node).state_names[node]
        value = intervention[node]
        if value not in states:
            raise ValueError(
                f"Intervention value '{value}' not a valid state of '{node}': {states}"
            )
        values = np.zeros((len(states), 1))
        values[states.index(value), 0] = 1.0
        cpd = TabularCPD(
            variable=node,
            variable_card=len(states),
            values=values,
            state_names={node: states},
        )
        bn_do.remove_cpds(node)
        bn_do.add_cpds(cpd)
    bn_do.check_model()
    engine = VariableElimination(bn_do)
    factor = engine.query([query], show_progress=False)
    return pd.Series(
        factor.values, index=factor.state_names[query], name=f"P({query}|do)"
    )
