"""Expert / domain-driven Bayesian Network structure.

Owner: Qicheng Jin (Structure Learning)

We encode standard cardiology knowledge as the prior DAG:

* Demographics (age, sex) sit at the root.
* Cholesterol, resting BP, and fasting blood sugar are *physiological*
  risk factors influenced by demographics.
* Exercise findings (thalach, exang, oldpeak, slope) are *manifestations*
  of underlying coronary disease.
* `thal` (thallium stress test) and `ca` (vessels colored by fluoroscopy)
  are downstream imaging findings.
* `cp` (chest pain type) is a symptom of disease.
* `target` (heart disease) is the latent class node.

This is a hand-crafted prior; the data-driven module learns an
alternative and we compare them in evaluation.
"""

from __future__ import annotations

from typing import Iterable

import networkx as nx


EXPERT_EDGES: list[tuple[str, str]] = [
    # demographic risk -> physiology
    ("age", "trestbps"),
    ("age", "chol"),
    ("age", "fbs"),
    ("sex", "chol"),
    ("sex", "thalach"),
    # demographics + physiology -> latent disease
    ("age", "target"),
    ("sex", "target"),
    ("chol", "target"),
    ("trestbps", "target"),
    ("fbs", "target"),
    # disease -> manifestations
    ("target", "cp"),
    ("target", "exang"),
    ("target", "thalach"),
    ("target", "oldpeak"),
    ("target", "restecg"),
    ("target", "thal"),
    ("target", "ca"),
    # local symptom couplings
    ("exang", "oldpeak"),
    ("oldpeak", "slope"),
]


def build_expert_dag(extra_edges: Iterable[tuple[str, str]] | None = None) -> nx.DiGraph:
    """Return the hand-crafted DAG as a ``networkx.DiGraph``.

    The structure is checked for acyclicity before being returned so that
    accidental edits to ``EXPERT_EDGES`` fail fast.
    """
    g = nx.DiGraph()
    g.add_edges_from(EXPERT_EDGES)
    if extra_edges:
        g.add_edges_from(extra_edges)
    if not nx.is_directed_acyclic_graph(g):
        cycles = list(nx.simple_cycles(g))
        raise ValueError(f"Expert DAG contains cycles: {cycles}")
    return g


def expert_edges() -> list[tuple[str, str]]:
    """Return a defensive copy of the expert edge list."""
    return list(EXPERT_EDGES)


def edge_rationale() -> dict[tuple[str, str], str]:
    """Short clinical justifications for each prior edge, used in the report."""
    return {
        ("age", "trestbps"): "Arterial stiffening with age elevates resting BP.",
        ("age", "chol"): "Serum cholesterol drifts upward with age.",
        ("age", "fbs"): "Insulin resistance increases with age.",
        ("sex", "chol"): "Estrogen lowers LDL until menopause.",
        ("sex", "thalach"): "Max heart rate differs by sex at equivalent fitness.",
        ("age", "target"): "Age is the single strongest risk factor for CAD.",
        ("sex", "target"): "Pre-menopausal women have lower CAD risk.",
        ("chol", "target"): "Hyperlipidemia drives atherogenesis.",
        ("trestbps", "target"): "Hypertension is a major modifiable risk factor.",
        ("fbs", "target"): "Diabetic dysglycemia accelerates atherosclerosis.",
        ("target", "cp"): "Ischemia produces angina (typical / atypical).",
        ("target", "exang"): "Effort ischemia causes exercise-induced angina.",
        ("target", "thalach"): "CAD limits chronotropic response on stress.",
        ("target", "oldpeak"): "Ischemia causes ST-segment depression.",
        ("target", "restecg"): "Prior MI / LVH shows on resting ECG.",
        ("target", "thal"): "Reversible defects on thallium imaging indicate CAD.",
        ("target", "ca"): "Number of diseased vessels reflects disease burden.",
        ("exang", "oldpeak"): "Symptomatic angina co-occurs with ST depression.",
        ("oldpeak", "slope"): "Greater ST depression alters slope morphology.",
    }
