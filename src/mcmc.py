"""Approximate inference: Gibbs and Metropolis-Hastings.

Owner: Chenqi Wang (Parameter Learning & Inference)

While Variable Elimination is tractable on this small network, the
class proposal explicitly calls for **sampling-based posterior
inference**, so we implement two textbook MCMC samplers from scratch
on top of pgmpy CPDs and compare them to the exact result.

Both samplers:

* condition on evidence by **locking** those variables to their
  observed values (rather than sampling from the joint and rejecting
  inconsistent samples — which has catastrophically low yield);
* exploit the **Markov blanket** of each variable so each sweep only
  touches local CPDs;
* return a DataFrame whose rows are samples of the **unobserved**
  variables.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Gibbs (single-site, conditional on evidence)
# ----------------------------------------------------------------------------
@dataclass
class GibbsConfig:
    n_samples: int = 5000
    burn_in: int = 1000
    thin: int = 1
    seed: int = 42


def _markov_blanket_dist(
    bn: DiscreteBayesianNetwork,
    node: str,
    assignment: dict[str, str],
) -> np.ndarray:
    """Return the conditional probabilities P(node = s | blanket) over s."""
    cpd = bn.get_cpds(node)
    states = cpd.state_names[node]
    probs = np.zeros(len(states))
    children = list(bn.successors(node))
    for i, s in enumerate(states):
        # P(node = s | parents(node))
        kwargs = {node: str(s)}
        for p in cpd.get_evidence():
            kwargs[p] = str(assignment[p])
        try:
            p_self = float(cpd.get_value(**kwargs))
        except Exception:
            p_self = 1.0 / len(states)
        # Product over children: P(child | parents(child)) with node = s.
        p_children = 1.0
        for c in children:
            ccpd = bn.get_cpds(c)
            ck = {c: str(assignment[c])}
            for p in ccpd.get_evidence():
                ck[p] = str(s) if p == node else str(assignment[p])
            try:
                p_children *= float(ccpd.get_value(**ck))
            except Exception:
                p_children *= 1.0 / len(ccpd.state_names[c])
        probs[i] = p_self * p_children
    total = probs.sum()
    if total <= 0:
        return np.full(len(states), 1.0 / len(states))
    return probs / total


def gibbs_posterior(
    bn: DiscreteBayesianNetwork,
    evidence: Mapping[str, str],
    query: str = "target",
    cfg: GibbsConfig | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Single-site Gibbs sampler that conditions on ``evidence`` exactly.

    Returns
    -------
    (pd.Series, pd.DataFrame)
        Empirical posterior over ``query`` and the trace of free variables.
    """
    cfg = cfg or GibbsConfig()
    rng = np.random.default_rng(cfg.seed)

    nodes = list(bn.nodes())
    free_vars = [n for n in nodes if n not in evidence]
    if query not in free_vars:
        raise ValueError(f"Query variable '{query}' must be unobserved.")

    state: dict[str, str] = {k: str(v) for k, v in evidence.items()}
    for v in free_vars:
        s = bn.get_cpds(v).state_names[v]
        state[v] = str(rng.choice(s))

    trace_rows: list[dict[str, str]] = []
    n_total = cfg.n_samples + cfg.burn_in

    for t in tqdm(range(n_total), desc="Gibbs", leave=False):
        for v in free_vars:
            states = bn.get_cpds(v).state_names[v]
            probs = _markov_blanket_dist(bn, v, state)
            state[v] = str(rng.choice(states, p=probs))
        if t >= cfg.burn_in and (t - cfg.burn_in) % cfg.thin == 0:
            trace_rows.append({k: state[k] for k in free_vars})

    trace = pd.DataFrame(trace_rows)
    counts = trace[query].value_counts(normalize=True).sort_index()
    target_states = bn.get_cpds(query).state_names[query]
    posterior = pd.Series(
        [counts.get(s, 0.0) for s in target_states], index=target_states, name=query
    )
    return posterior, trace


# ----------------------------------------------------------------------------
# Metropolis-Hastings (from-scratch)
# ----------------------------------------------------------------------------
@dataclass
class MHConfig:
    n_samples: int = 5000
    burn_in: int = 1000
    seed: int = 42


def _markov_blanket_logprob(
    bn: DiscreteBayesianNetwork,
    node: str,
    assignment: dict[str, str],
) -> float:
    """Log-probability of (node, assignment) conditioned on its blanket.

    For an MH step that proposes a new value of a single variable, the
    acceptance ratio simplifies to a ratio of CPDs in the variable's
    Markov blanket: itself and its children's CPDs.
    """
    nodes = [node] + list(bn.successors(node))
    lp = 0.0
    for v in nodes:
        cpd = bn.get_cpds(v)
        parents = cpd.get_evidence()
        kwargs = {v: str(assignment[v])}
        for p in parents:
            kwargs[p] = str(assignment[p])
        try:
            p_val = float(cpd.get_value(**kwargs))
        except Exception:
            p_val = 1.0 / len(cpd.state_names[v])
        lp += np.log(max(p_val, 1e-12))
    return lp


def metropolis_hastings(
    bn: DiscreteBayesianNetwork,
    evidence: Mapping[str, str],
    query: str = "target",
    cfg: MHConfig | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Single-site MH sampler. Returns (posterior, trace)."""
    cfg = cfg or MHConfig()
    rng = np.random.default_rng(cfg.seed)

    nodes = list(bn.nodes())
    free_vars = [n for n in nodes if n not in evidence]
    if query not in free_vars:
        raise ValueError(f"Query variable '{query}' must be unobserved.")

    # Initialize: evidence locked; free vars sampled uniformly from each
    # variable's state space. Any starting point converges after burn-in;
    # the choice here is uniform rather than from the marginal prior.
    state = dict(evidence)
    for v in free_vars:
        states = bn.get_cpds(v).state_names[v]
        state[v] = rng.choice(states)

    trace_rows: list[dict[str, str]] = []
    n_total = cfg.n_samples + cfg.burn_in
    accepted = 0

    iterator = tqdm(range(n_total), desc="MH", leave=False)
    for t in iterator:
        v = rng.choice(free_vars)
        states = bn.get_cpds(v).state_names[v]
        current = state[v]
        choices = [s for s in states if s != current]
        if not choices:
            continue
        proposal = rng.choice(choices)

        lp_current = _markov_blanket_logprob(bn, v, state)
        new_state = dict(state)
        new_state[v] = proposal
        lp_proposal = _markov_blanket_logprob(bn, v, new_state)

        log_alpha = lp_proposal - lp_current
        # Symmetric proposal: q(x'|x) == q(x|x') == 1 / (|states|-1).
        if np.log(rng.uniform()) < log_alpha:
            state = new_state
            accepted += 1

        if t >= cfg.burn_in:
            trace_rows.append(dict(state))

    trace = pd.DataFrame(trace_rows)
    logger.info(
        "MH acceptance rate: %.3f (%d / %d)",
        accepted / n_total,
        accepted,
        n_total,
    )
    counts = trace[query].value_counts(normalize=True).sort_index()
    target_states = bn.get_cpds(query).state_names[query]
    posterior = pd.Series(
        [counts.get(s, 0.0) for s in target_states], index=target_states, name=query
    )
    return posterior, trace


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
def _resolve_target_state(trace: pd.Series, target_state: str | None) -> str | None:
    """Pick the indicator state for a 0/1 reduction of a categorical trace.

    Returns ``None`` for boolean traces (caller should treat them directly).
    Raises ``ValueError`` for multi-state traces when ``target_state`` is
    not supplied — the old behavior of silently picking ``max(unique)``
    was a latent bug for non-binary targets.
    """
    if trace.dtype == bool:
        return None
    if target_state is not None:
        return target_state
    unique = sorted(trace.unique())
    if len(unique) != 2:
        raise ValueError(
            f"trace has {len(unique)} distinct values {unique}; "
            "pass target_state explicitly for non-binary traces."
        )
    return unique[-1]


def running_mean(
    trace: pd.Series, target_state: str | None = None
) -> np.ndarray:
    """Running estimate of P(trace == target_state).

    For binary traces (e.g. ``target ∈ {'0','1'}``) we default to the
    lexicographically larger state. For multi-state traces, the caller
    must pass ``target_state`` explicitly — see :func:`_resolve_target_state`.
    """
    resolved = _resolve_target_state(trace, target_state)
    if resolved is None:
        x = trace.astype(float).values
    else:
        x = (trace == resolved).astype(float).values
    return np.cumsum(x) / np.arange(1, len(x) + 1)


def autocorrelation(
    trace: pd.Series, max_lag: int = 50, target_state: str | None = None
) -> np.ndarray:
    """Sample autocorrelation of a 0/1 indicator trace.

    See :func:`running_mean` for how ``target_state`` is resolved.
    """
    resolved = _resolve_target_state(trace, target_state)
    if resolved is None:
        x = trace.astype(float).values
    else:
        x = (trace == resolved).astype(float).values
    x = x - x.mean()
    n = len(x)
    acf = np.zeros(max_lag + 1)
    var = np.dot(x, x)
    if var == 0:
        return acf
    for lag in range(max_lag + 1):
        acf[lag] = np.dot(x[: n - lag], x[lag:]) / var
    return acf
