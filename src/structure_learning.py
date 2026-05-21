"""Data-driven Bayesian Network structure learning.

Owner: Qicheng Jin (Structure Learning)

We compare three strategies:

1. **Score-based Hill-Climb** with BIC, K2, and BDeu scores. We force
   the target to remain non-isolated by black-listing edges that would
   make it a sink only (configurable).
2. **Constraint-based PC** (Peter-Clark) which performs conditional
   independence tests and orients edges via collider rules. The
   resulting CPDAG is converted to a DAG by orienting remaining
   undirected edges via Meek's rules / random tie-break.
3. **Hybrid MMHC** (Max-Min Hill Climb) which first constrains the
   neighborhood via the Max-Min Parents-and-Children skeleton and then
   runs a score-based search inside that skeleton.

Each learner returns a ``networkx.DiGraph`` for downstream use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import networkx as nx
import pandas as pd

from pgmpy.estimators import (
    BDeu as BDeuScore,
    BIC as BicScore,
    ExpertKnowledge,
    HillClimbSearch,
    K2 as K2Score,
    MmhcEstimator,
    PC,
)

logger = logging.getLogger(__name__)


@dataclass
class StructureSearchConfig:
    """Configuration for the structure search."""

    max_indegree: int = 4
    scoring: str = "bic"  # one of {bic, k2, bdeu}
    equivalent_sample_size: int = 10  # only used for BDeu
    pc_alpha: float = 0.05  # significance level for PC tests
    fixed_edges: Optional[list[tuple[str, str]]] = None  # whitelist
    forbidden_edges: Optional[list[tuple[str, str]]] = None  # blacklist


def _score_for(name: str, data: pd.DataFrame, cfg: StructureSearchConfig):
    name = name.lower()
    if name == "bic":
        return BicScore(data)
    if name == "k2":
        return K2Score(data)
    if name == "bdeu":
        return BDeuScore(data, equivalent_sample_size=cfg.equivalent_sample_size)
    raise ValueError(f"Unknown score '{name}'.")


def _expert_knowledge_from_cfg(
    cfg: StructureSearchConfig,
) -> Optional[ExpertKnowledge]:
    """Translate the user-friendly cfg knobs into pgmpy 1.0's
    ``ExpertKnowledge`` object (which replaced the old whitelist /
    blacklist keyword arguments).
    """
    if not cfg.fixed_edges and not cfg.forbidden_edges:
        return None
    return ExpertKnowledge(
        required_edges=cfg.fixed_edges or [],
        forbidden_edges=cfg.forbidden_edges or [],
    )


def learn_hill_climb(
    data: pd.DataFrame, cfg: StructureSearchConfig | None = None
) -> nx.DiGraph:
    """Hill-climbing structure search."""
    cfg = cfg or StructureSearchConfig()
    score = _score_for(cfg.scoring, data, cfg)
    searcher = HillClimbSearch(data)
    best_model = searcher.estimate(
        scoring_method=score,
        max_indegree=cfg.max_indegree,
        expert_knowledge=_expert_knowledge_from_cfg(cfg),
        show_progress=False,
    )
    g = nx.DiGraph()
    g.add_nodes_from(data.columns)
    g.add_edges_from(best_model.edges())
    return g


def learn_pc(
    data: pd.DataFrame, cfg: StructureSearchConfig | None = None
) -> nx.DiGraph:
    """PC algorithm: skeleton via CI tests, then collider orientation."""
    cfg = cfg or StructureSearchConfig()
    est = PC(data)
    cpdag = est.estimate(
        ci_test="chi_square",
        significance_level=cfg.pc_alpha,
        return_type="dag",
        show_progress=False,
        n_jobs=1,
    )
    g = nx.DiGraph()
    g.add_nodes_from(data.columns)
    g.add_edges_from(cpdag.edges())
    return g


def learn_mmhc(
    data: pd.DataFrame, cfg: StructureSearchConfig | None = None
) -> nx.DiGraph:
    """Hybrid MMHC: constraint-screened skeleton, then HC inside it."""
    cfg = cfg or StructureSearchConfig()
    est = MmhcEstimator(data)
    model = est.estimate(significance_level=cfg.pc_alpha)
    g = nx.DiGraph()
    g.add_nodes_from(data.columns)
    g.add_edges_from(model.edges())
    return g


def score_dag(
    dag: nx.DiGraph,
    data: pd.DataFrame,
    scoring: str = "bic",
    cfg: StructureSearchConfig | None = None,
) -> float:
    """Return the chosen score for a candidate DAG on the data."""
    cfg = cfg or StructureSearchConfig()
    s = _score_for(scoring, data, cfg)
    return float(s.score(dag))


def compare_structures(
    learners: dict[str, nx.DiGraph],
    data: pd.DataFrame,
    scoring: tuple[str, ...] = ("bic", "k2", "bdeu"),
) -> pd.DataFrame:
    """Build a side-by-side table of score values per (learner, score)."""
    rows = []
    for name, dag in learners.items():
        row = {"learner": name, "n_edges": dag.number_of_edges()}
        for s in scoring:
            row[s] = score_dag(dag, data, scoring=s)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("bic", ascending=False).reset_index(drop=True)


def edge_set_diff(
    expert: nx.DiGraph, learned: nx.DiGraph
) -> dict[str, list[tuple[str, str]]]:
    """Symmetric difference of two edge sets (ignoring orientation differences).

    Useful for the qualitative comparison section: which expert edges
    were *not* discovered by data, and which new edges did the data
    suggest?
    """
    expert_e = {tuple(sorted(e)) for e in expert.edges()}
    learned_e = {tuple(sorted(e)) for e in learned.edges()}
    only_expert = sorted(expert_e - learned_e)
    only_learned = sorted(learned_e - expert_e)
    shared = sorted(expert_e & learned_e)
    return {
        "only_in_expert": [tuple(e) for e in only_expert],
        "only_in_learned": [tuple(e) for e in only_learned],
        "shared": [tuple(e) for e in shared],
    }
