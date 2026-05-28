"""Parameter (CPD) estimation for a fixed DAG.

Owner: Chenqi Wang (Parameter Learning & Inference)

We support two estimators:

* **MLE** — maximum likelihood. Fast, but produces zero-probability
  cells whenever a parent configuration is unobserved in training.
* **Bayesian (Dirichlet)** — posterior-mean estimates with a symmetric
  Dirichlet prior. Uses pgmpy's BDeu equivalent sample size to control
  the prior strength. This is the recommended estimator for small
  datasets like UCI Heart Disease.

Both estimators return a ready-to-use ``DiscreteBayesianNetwork``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import networkx as nx
import pandas as pd
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork

logger = logging.getLogger(__name__)


@dataclass
class ParameterFitConfig:
    """Configuration for parameter estimation."""

    method: str = "bayes"  # one of {mle, bayes}
    prior_type: str = "BDeu"  # one of {BDeu, dirichlet, K2}
    equivalent_sample_size: int = 10  # only used for BDeu


def fit_parameters(
    dag: nx.DiGraph,
    data: pd.DataFrame,
    cfg: ParameterFitConfig | None = None,
    state_names: Mapping[str, list[str]] | None = None,
) -> DiscreteBayesianNetwork:
    """Estimate CPDs for ``dag`` from ``data`` and return a fitted BN.

    Parameters
    ----------
    dag
        Acyclic structure over the columns of ``data``.
    data
        Discrete dataframe (string-valued).
    cfg
        See ``ParameterFitConfig``.
    state_names
        Optional explicit state-name dictionary to lock state ordering
        across train/test (essential when discretization yields rare
        categories absent from a fold).
    """
    cfg = cfg or ParameterFitConfig()

    bn = DiscreteBayesianNetwork()
    bn.add_nodes_from(dag.nodes())
    bn.add_edges_from(dag.edges())

    fit_kwargs: dict = {}
    if state_names is not None:
        fit_kwargs["state_names"] = state_names

    method = cfg.method.lower()
    if method == "mle":
        bn.fit(
            data, estimator=MaximumLikelihoodEstimator, **fit_kwargs
        )
    elif method == "bayes":
        bn.fit(
            data,
            estimator=BayesianEstimator,
            prior_type=cfg.prior_type,
            equivalent_sample_size=cfg.equivalent_sample_size,
            **fit_kwargs,
        )
    else:
        raise ValueError(f"Unknown estimator '{cfg.method}'.")

    # Isolated nodes (no parents and no children) are not assigned CPDs by
    # pgmpy 1.0's fit(). We supply their empirical marginal so the BN is
    # complete and ``check_model`` succeeds.
    _attach_marginals_to_isolated_nodes(bn, data, state_names)
    bn.check_model()
    logger.info(
        "Fitted BN with %d nodes, %d edges via %s.",
        bn.number_of_nodes(),
        bn.number_of_edges(),
        cfg.method,
    )
    return bn


def _attach_marginals_to_isolated_nodes(
    bn: DiscreteBayesianNetwork,
    data: pd.DataFrame,
    state_names: Mapping[str, list[str]] | None = None,
) -> None:
    """Give every node missing a CPD its empirical marginal.

    Hill-Climb and PC can return DAGs in which some variables have no
    edges at all. pgmpy doesn't auto-fit those.
    """
    from pgmpy.factors.discrete import TabularCPD
    import numpy as np

    for node in bn.nodes():
        if bn.get_cpds(node) is not None:
            continue
        if state_names is not None and node in state_names:
            states = list(state_names[node])
        else:
            states = sorted(data[node].astype(str).unique().tolist())
        counts = data[node].astype(str).value_counts()
        probs = np.array([counts.get(s, 0.0) for s in states], dtype=float)
        # Laplace smoothing (helps inference on small samples).
        probs = probs + 1.0
        probs = probs / probs.sum()
        cpd = TabularCPD(
            variable=node,
            variable_card=len(states),
            values=probs.reshape(-1, 1),
            state_names={node: states},
        )
        bn.add_cpds(cpd)


def cpd_summary(bn: DiscreteBayesianNetwork) -> pd.DataFrame:
    """Return a one-row-per-node summary of the fitted CPDs."""
    rows = []
    for node in bn.nodes():
        cpd = bn.get_cpds(node)
        parents = cpd.get_evidence()
        n_states = len(cpd.state_names[node])
        card_product_parents = cpd.values.size // n_states
        # Free params: each parent configuration has n_states − 1 degrees of
        # freedom (last entry fixed by sum-to-one).
        n_params = (n_states - 1) * card_product_parents
        rows.append(
            {
                "node": node,
                "n_states": n_states,
                "parents": ", ".join(parents) if parents else "-",
                "n_params": n_params,
                "card_product_parents": card_product_parents,
            }
        )
    return pd.DataFrame(rows)


def log_likelihood(bn: DiscreteBayesianNetwork, data: pd.DataFrame) -> float:
    """Compute the log-likelihood of ``data`` under ``bn`` (mean per sample).

    Robust to differences in axis ordering between pgmpy versions: we
    use ``cpd.get_value`` which accepts kwarg-style state lookups.

    If any CPD lookup fails (which only happens when ``state_names`` is
    not locked across train/test), we fall back to a uniform probability
    *and* emit a single warning with the offending row — so the silent
    masking that used to live here can no longer hide a broken state set.
    """
    import numpy as np

    total = 0.0
    n = len(data)
    n_fallback = 0
    fallback_sample: tuple | None = None
    for _, row in data.iterrows():
        joint = 1.0
        for node in bn.nodes():
            cpd = bn.get_cpds(node)
            parents = cpd.get_evidence()
            kwargs = {node: str(row[node])}
            for p in parents:
                kwargs[p] = str(row[p])
            try:
                p_val = float(cpd.get_value(**kwargs))
            except Exception as exc:
                if n_fallback == 0:
                    fallback_sample = (node, dict(kwargs), str(exc))
                n_fallback += 1
                p_val = 1.0 / len(cpd.state_names[node])
            joint *= max(p_val, 1e-12)
        total += np.log(joint)
    if n_fallback:
        node, kwargs, exc = fallback_sample
        logger.warning(
            "log_likelihood: %d CPD lookups fell back to uniform "
            "(first: node=%r evidence=%r — %s). Pass an explicit "
            "state_names dict to fit_parameters to lock state ordering.",
            n_fallback, node, kwargs, exc,
        )
    return total / n
