"""Plotting helpers shared across notebooks.

Owner: Shared utility.
"""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


# ----------------------------------------------------------------------------
# DAG plotting
# ----------------------------------------------------------------------------
def _layered_layout(
    dag: nx.DiGraph,
    x_jitter: float = 0.0,
) -> dict[str, tuple[float, float]]:
    """Hierarchical layered layout for a DAG.

    Uses ``networkx.topological_generations`` to place each node on a
    horizontal layer (top = roots, bottom = sinks). Nodes inside a layer
    are spread evenly along the x-axis. Isolated nodes are pushed to a
    dedicated bottom layer so they don't collide with the rest of the
    graph.
    """
    nodes = list(dag.nodes())
    isolated = [n for n in nodes if dag.degree(n) == 0]
    core = dag.subgraph([n for n in nodes if n not in isolated]).copy()

    generations: list[list[str]] = []
    if core.number_of_nodes() > 0:
        try:
            generations = [sorted(layer) for layer in nx.topological_generations(core)]
        except nx.NetworkXUnfeasible:
            generations = [sorted(core.nodes())]
    if isolated:
        generations.append(sorted(isolated))

    pos: dict[str, tuple[float, float]] = {}
    n_layers = max(len(generations), 1)
    for i, layer in enumerate(generations):
        y = 1.0 - (i / max(n_layers - 1, 1))
        m = len(layer)
        for j, node in enumerate(layer):
            # Evenly spaced in [0.05, 0.95], centered.
            if m == 1:
                x = 0.5
            else:
                x = 0.05 + 0.9 * j / (m - 1)
            # Slight jitter per layer to break visual ties of straight columns.
            if x_jitter and i % 2 == 1:
                x += x_jitter
            pos[node] = (x, y)
    return pos


def _resolve_layout(dag: nx.DiGraph, layout: str) -> dict[str, tuple[float, float]]:
    """Pick a layout, preferring graphviz `dot` when available."""
    layout = (layout or "layered").lower()
    if layout in ("layered", "hierarchical", "dot"):
        # Try graphviz dot first (best hierarchical layout for DAGs).
        try:
            from networkx.drawing.nx_agraph import graphviz_layout

            return graphviz_layout(dag, prog="dot")
        except Exception:
            pass
        try:
            from networkx.drawing.nx_pydot import graphviz_layout as _gv

            return _gv(dag, prog="dot")
        except Exception:
            pass
        return _layered_layout(dag, x_jitter=0.03)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(dag)
    if layout == "spring":
        return nx.spring_layout(dag, seed=42, k=1.8, iterations=200)
    if layout == "shell":
        return nx.shell_layout(dag)
    return _layered_layout(dag)


def plot_dag(
    dag: nx.DiGraph,
    title: str = "",
    highlight_node: str | None = "target",
    layout: str = "layered",
    ax=None,
    figsize: tuple[int, int] = (8, 6),
    node_size: int = 1700,
    font_size: int = 9,
):
    """Render a DAG with a highlighted target node and a clean layered layout."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    pos = _resolve_layout(dag, layout)

    node_colors = [
        "#ef553b" if n == highlight_node else "#636efa" for n in dag.nodes()
    ]
    nx.draw_networkx_nodes(
        dag,
        pos,
        node_color=node_colors,
        node_size=node_size,
        edgecolors="white",
        linewidths=2,
        ax=ax,
    )
    nx.draw_networkx_edges(
        dag,
        pos,
        edge_color="#888",
        arrows=True,
        arrowsize=14,
        width=1.2,
        node_size=node_size,
        connectionstyle="arc3,rad=0.10",
        min_source_margin=12,
        min_target_margin=12,
        ax=ax,
    )
    nx.draw_networkx_labels(
        dag, pos, font_size=font_size, font_color="white", font_weight="bold", ax=ax
    )
    # Add a small margin so labels at the edges aren't clipped.
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if xs and ys:
        x_pad = (max(xs) - min(xs)) * 0.08 + 0.05
        y_pad = (max(ys) - min(ys)) * 0.12 + 0.05
        ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
        ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)
    ax.set_title(title)
    ax.set_axis_off()
    return ax


def plot_two_dags(
    dag_a: nx.DiGraph,
    dag_b: nx.DiGraph,
    title_a: str = "Expert",
    title_b: str = "Learned",
    figsize: tuple[int, int] = (18, 8),
    layout: str = "layered",
):
    """Side-by-side DAG comparison with clean hierarchical layouts."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_dag(dag_a, title=title_a, ax=axes[0], layout=layout)
    plot_dag(dag_b, title=title_b, ax=axes[1], layout=layout)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Calibration / ROC
# ----------------------------------------------------------------------------
def plot_reliability(
    reliability_df: pd.DataFrame,
    label: str = "model",
    ax=None,
):
    """Reliability diagram from the table returned by ``evaluation.reliability_curve``."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    df = reliability_df.dropna()
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfectly calibrated")
    ax.plot(df["avg_predicted"], df["frac_positive"], "o-", label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical fraction positive")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Reliability diagram")
    return ax


def plot_reliability_multi(
    curves: Mapping[str, pd.DataFrame], ax=None
):
    """Compare multiple reliability curves on one axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    for label, df in curves.items():
        df = df.dropna()
        ax.plot(df["avg_predicted"], df["frac_positive"], "o-", label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical fraction positive")
    ax.set_title("Reliability diagrams")
    ax.legend()
    return ax


def plot_roc_curves(
    y_true: np.ndarray, probas: dict[str, np.ndarray], ax=None
):
    """ROC curves for multiple models on one axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend()
    return ax


# ----------------------------------------------------------------------------
# MCMC diagnostics
# ----------------------------------------------------------------------------
def plot_trace_and_running(trace_value: np.ndarray, running: np.ndarray, ax=None):
    """Trace plot + running posterior estimate side-by-side."""
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot(trace_value, lw=0.6)
    ax[0].set_title("MCMC trace (indicator of positive state)")
    ax[0].set_xlabel("iteration")
    ax[1].plot(running, color="crimson")
    ax[1].set_title("Running mean estimate of P(target=1)")
    ax[1].set_xlabel("iteration")
    ax[1].set_ylim(0, 1)
    return ax


def plot_autocorrelation(acf: np.ndarray, ax=None):
    """Stem-plot of autocorrelation."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    ax.stem(acf, basefmt=" ")
    ax.set_xlabel("lag")
    ax.set_ylabel("autocorrelation")
    ax.set_title("MCMC autocorrelation")
    return ax


# ----------------------------------------------------------------------------
# Uncertainty
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Pipeline / methodology diagram
# ----------------------------------------------------------------------------
def plot_pipeline_diagram(ax=None, figsize: tuple[float, float] = (13, 6)):
    """Render the end-to-end methodology pipeline as a matplotlib figure."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    boxes = [
        ("UCI Heart Disease\n(303 × 14)", (0.5, 5.0), "#dbe9ff"),
        ("Clinical binning &\ntrain/test split", (3.0, 5.0), "#dbe9ff"),
        ("Expert DAG\n(cardiology)", (6.0, 6.2), "#fff2cc"),
        ("Hill-Climb / PC\n(data-driven DAGs)", (6.0, 3.8), "#fff2cc"),
        ("Param fit\n(MLE / BDeu)", (9.0, 5.0), "#e1d5e7"),
        ("Variable\nElimination", (12.0, 6.5), "#d5e8d4"),
        ("Gibbs / MH\nMCMC", (12.0, 5.0), "#d5e8d4"),
        ("do-operator\n(counterfactual)", (12.0, 3.5), "#d5e8d4"),
        ("Eval, calibration,\nuncertainty", (15.5, 5.0), "#f8cecc"),
    ]
    for label, (x, y), color in boxes:
        box = mpatches.FancyBboxPatch(
            (x - 0.85, y - 0.55),
            1.7, 1.1,
            boxstyle="round,pad=0.05",
            linewidth=1.2,
            facecolor=color,
            edgecolor="#333333",
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=9)

    arrows = [
        ((1.35, 5.0), (2.15, 5.0)),
        ((3.85, 5.0), (5.15, 6.2)),
        ((3.85, 5.0), (5.15, 3.8)),
        ((6.85, 6.2), (8.15, 5.1)),
        ((6.85, 3.8), (8.15, 4.9)),
        ((9.85, 5.1), (11.15, 6.3)),
        ((9.85, 5.0), (11.15, 5.0)),
        ((9.85, 4.9), (11.15, 3.7)),
        ((12.85, 6.5), (14.65, 5.1)),
        ((12.85, 5.0), (14.65, 5.0)),
        ((12.85, 3.5), (14.65, 4.9)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.4),
        )

    ax.set_xlim(-0.5, 17.0)
    ax.set_ylim(2.5, 7.5)
    ax.set_axis_off()
    ax.set_title("End-to-end methodology pipeline", fontsize=13)
    legend_handles = [
        mpatches.Patch(facecolor="#dbe9ff", edgecolor="#333", label="Data"),
        mpatches.Patch(facecolor="#fff2cc", edgecolor="#333", label="Structure"),
        mpatches.Patch(facecolor="#e1d5e7", edgecolor="#333", label="Parameters"),
        mpatches.Patch(facecolor="#d5e8d4", edgecolor="#333", label="Inference"),
        mpatches.Patch(facecolor="#f8cecc", edgecolor="#333", label="Evaluation"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))
    return ax


# ----------------------------------------------------------------------------
# Confusion matrix grid + per-model metric bar chart
# ----------------------------------------------------------------------------
def plot_confusion_grid(
    y_true: np.ndarray,
    probas: Mapping[str, np.ndarray],
    threshold: float = 0.5,
    figsize: tuple[float, float] | None = None,
):
    """Render a row of confusion matrices, one per model."""
    n = len(probas)
    figsize = figsize or (4 * n, 3.8)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, (name, p) in zip(axes, probas.items()):
        y_pred = (np.asarray(p) >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No disease", "Disease"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(name, fontsize=11)
        ax.grid(False)
    fig.suptitle(f"Confusion matrices at threshold = {threshold}", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def plot_pr_curves(
    y_true: np.ndarray, probas: Mapping[str, np.ndarray], ax=None
):
    """Precision-recall curves for multiple models."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))
    base_rate = float(np.mean(y_true))
    ax.axhline(base_rate, color="gray", linestyle="--", label=f"prevalence = {base_rate:.2f}")
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax.plot(rec, prec, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curves")
    ax.legend()
    return ax


def plot_metric_barchart(
    benchmark_df: pd.DataFrame,
    metrics: tuple[str, ...] = ("roc_auc", "accuracy", "f1", "brier", "ece"),
    figsize: tuple[float, float] | None = None,
):
    """Grouped bar chart of selected metrics across all models.

    ``benchmark_df`` is the output of ``evaluation.benchmark``. Lower
    values are better for Brier and ECE — we invert them visually by
    plotting ``1 - x`` for those columns so that 'taller is always better'.
    """
    df = benchmark_df[list(metrics)].copy()
    invert = {"brier", "ece", "log_loss"}
    for col in df.columns:
        if col in invert:
            df[col] = 1 - df[col]
            df.rename(columns={col: f"1 - {col}"}, inplace=True)

    figsize = figsize or (max(7, 1.5 * len(df.columns) * len(df)), 4.2)
    ax = df.T.plot.bar(figsize=figsize, edgecolor="white", width=0.85)
    ax.set_ylabel("score (higher = better)")
    ax.set_ylim(0, 1)
    ax.set_title("Head-to-head: BN vs. discriminative baselines")
    ax.legend(title="model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticklabels(df.columns, rotation=0)
    plt.tight_layout()
    return ax


# ----------------------------------------------------------------------------
# CPD heatmap (for the disease node)
# ----------------------------------------------------------------------------
def plot_target_cpd_heatmap(bn, target: str = "target", figsize=(11, 4.2), ax=None):
    """Heatmap of P(target = 1 | parents).

    Only works if the target has at most ~3 parents (otherwise the
    heatmap becomes unreadable). For our expert DAG, ``target`` has the
    parents (age, sex, chol, trestbps, fbs) — too many. We marginalize
    out everything except (age, chol) using the network's joint factor.
    """
    from itertools import product
    import seaborn as sns

    cpd = bn.get_cpds(target)
    states = cpd.state_names[target]
    positive = sorted(states)[-1]
    parents = cpd.get_evidence()
    if not parents:
        return None
    # Pick the two most-discriminative parents if there are >2.
    if len(parents) > 2:
        parents_used = ["age", "chol"] if {"age", "chol"} <= set(parents) else parents[:2]
    else:
        parents_used = list(parents)

    other_parents = [p for p in parents if p not in parents_used]
    other_state_combos = (
        list(product(*[cpd.state_names[p] for p in other_parents]))
        if other_parents
        else [()]
    )

    parent_states = [cpd.state_names[p] for p in parents_used]
    matrix = np.zeros((len(parent_states[0]), len(parent_states[1])))
    for i, s1 in enumerate(parent_states[0]):
        for j, s2 in enumerate(parent_states[1]):
            probs = []
            for combo in other_state_combos:
                kwargs = {target: positive, parents_used[0]: s1, parents_used[1]: s2}
                for p, v in zip(other_parents, combo):
                    kwargs[p] = v
                try:
                    probs.append(float(cpd.get_value(**kwargs)))
                except Exception:
                    pass
            matrix[i, j] = float(np.mean(probs)) if probs else np.nan

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        xticklabels=parent_states[1],
        yticklabels=parent_states[0],
        cmap="rocket_r",
        cbar_kws={"label": f"P({target}={positive})"},
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel(parents_used[1])
    ax.set_ylabel(parents_used[0])
    extra = ""
    if other_parents:
        extra = "\n(averaged over " + ", ".join(other_parents) + ")"
    ax.set_title(f"CPD slice: P({target}={positive} | {', '.join(parents_used)}){extra}")
    return ax


def plot_uncertainty_intervals(
    mean: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    y_true: np.ndarray | None = None,
    max_show: int = 60,
    ax=None,
):
    """Caterpillar plot: mean ± 95% CI for the first ``max_show`` patients."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    n = min(len(mean), max_show)
    order = np.argsort(mean[:n])
    x = np.arange(n)
    ax.errorbar(
        x,
        mean[:n][order],
        yerr=[
            mean[:n][order] - ci_low[:n][order],
            ci_high[:n][order] - mean[:n][order],
        ],
        fmt="o",
        ms=3,
        capsize=2,
        elinewidth=0.8,
        color="#3366CC",
        ecolor="#9bb8e2",
        label="P(target=1) ± 95% CI",
    )
    if y_true is not None:
        truth = y_true[:n][order]
        ax.scatter(
            x,
            truth,
            marker="x",
            color="crimson",
            s=20,
            label="ground truth",
        )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(f"patient index (sorted by mean prediction; first {n})")
    ax.set_ylabel("posterior P(target=1)")
    ax.set_title("Epistemic uncertainty around the Bayesian Network prediction")
    ax.legend()
    return ax
