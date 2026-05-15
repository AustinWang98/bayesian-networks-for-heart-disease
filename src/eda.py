"""Exploratory data analysis helpers.

Owner: Member 1 (Data & EDA)
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def class_balance(df: pd.DataFrame, target: str = "target") -> pd.Series:
    """Return the empirical marginal P(target)."""
    return df[target].value_counts(normalize=True).sort_index()


def plot_class_balance(df: pd.DataFrame, target: str = "target", ax=None):
    """Bar plot of P(target)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    counts = df[target].value_counts(normalize=True).sort_index()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="viridis")
    ax.set_ylabel("P(target)")
    ax.set_xlabel("target")
    ax.set_title("Class balance")
    return ax


def plot_feature_distributions(
    df: pd.DataFrame,
    features: Iterable[str],
    target: str = "target",
    n_cols: int = 3,
):
    """Grid of per-feature conditional bar plots: P(feature | target)."""
    features = list(features)
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows))
    axes = np.atleast_2d(axes).ravel()
    for i, feat in enumerate(features):
        ct = pd.crosstab(df[feat], df[target], normalize="columns")
        ct.plot.bar(ax=axes[i], width=0.8, edgecolor="white")
        axes[i].set_title(f"P({feat} | target)")
        axes[i].set_ylabel("probability")
        axes[i].set_xlabel(feat)
        axes[i].legend(title="target", fontsize=8)
    for j in range(len(features), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    return fig


def mutual_information_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise (symmetric) mutual information between discrete columns.

    Mutual information is a non-linear dependence measure that's a
    natural pre-screen for which edges might appear in the learned DAG.
    """
    from sklearn.metrics import mutual_info_score

    cols = list(df.columns)
    mi = np.zeros((len(cols), len(cols)))
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i <= j:
                score = mutual_info_score(df[a], df[b])
                mi[i, j] = score
                mi[j, i] = score
    return pd.DataFrame(mi, index=cols, columns=cols)


def plot_mi_heatmap(mi: pd.DataFrame, ax=None):
    """Heatmap of the mutual information matrix."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        mi,
        ax=ax,
        cmap="rocket_r",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "MI (nats)"},
        square=True,
    )
    ax.set_title("Pairwise mutual information between variables")
    return ax


def plot_violin_by_target(
    raw_df: pd.DataFrame,
    features: list[str] | None = None,
    target_col: str = "num",
    figsize: tuple[float, float] | None = None,
):
    """Violin plots of continuous variables, split by disease status.

    Operates on the *raw* (pre-binning) dataframe to show what the
    BN sees *before* discretization.
    """
    features = features or ["age", "trestbps", "chol", "thalach", "oldpeak"]
    figsize = figsize or (3.4 * len(features), 4.0)
    fig, axes = plt.subplots(1, len(features), figsize=figsize)
    df = raw_df.copy()
    df["disease"] = (pd.to_numeric(df[target_col], errors="coerce") > 0).astype(int)
    for ax, feat in zip(axes, features):
        sns.violinplot(
            data=df, x="disease", y=feat, ax=ax,
            hue="disease",
            palette=["#7CB9E8", "#FF6B6B"],
            inner="quartile",
            legend=False,
        )
        ax.set_title(feat)
        ax.set_xticklabels(["no disease", "disease"])
        ax.set_xlabel("")
    fig.suptitle("Continuous risk factors by disease status", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def plot_categorical_target_heatmap(
    df: pd.DataFrame,
    features: list[str] | None = None,
    target: str = "target",
    figsize: tuple[float, float] | None = None,
):
    """Heatmap grid of P(target=1 | feature) for categorical features."""
    features = features or ["cp", "sex", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    figsize = figsize or (1.4 * len(features), 4.0)
    fig, axes = plt.subplots(1, len(features), figsize=figsize)
    for ax, feat in zip(axes, features):
        ct = pd.crosstab(df[feat], df[target], normalize="index")
        positive = sorted(ct.columns)[-1]
        s = ct[positive].sort_index()
        s.plot.barh(ax=ax, color="#FF8C42", edgecolor="white")
        ax.set_xlim(0, 1)
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("")
        ax.axvline(df[target].astype(str).eq(positive).mean(),
                   color="gray", linestyle="--", linewidth=0.8)
    fig.suptitle(f"P({target}=1 | feature)   — dashed line: base rate", y=1.02)
    fig.tight_layout()
    return fig


def plot_bin_boundaries(
    raw_df: pd.DataFrame,
    discretization: dict | None = None,
    figsize: tuple[float, float] = (14, 4.0),
):
    """Show clinical bin edges over the empirical KDE of each continuous variable."""
    from .preprocessing import DISCRETIZATION
    discretization = discretization or DISCRETIZATION
    cont_vars = [v for v in discretization if v in raw_df.columns]
    fig, axes = plt.subplots(1, len(cont_vars), figsize=figsize)
    for ax, var in zip(axes, cont_vars):
        x = pd.to_numeric(raw_df[var], errors="coerce").dropna()
        sns.kdeplot(x, fill=True, ax=ax, color="#1f77b4")
        edges, _ = discretization[var]
        for e in edges:
            if not np.isfinite(e):
                continue
            ax.axvline(e, color="crimson", linestyle="--", linewidth=1.2)
        ax.set_title(var, fontsize=10)
        ax.set_xlabel("")
    fig.suptitle("Clinical bin boundaries (red dashed) over empirical KDEs", y=1.02)
    fig.tight_layout()
    return fig


def chi_square_table(df: pd.DataFrame, target: str = "target") -> pd.DataFrame:
    """χ² test of independence between each feature and the target."""
    from scipy.stats import chi2_contingency

    rows = []
    for col in df.columns.drop(target):
        ct = pd.crosstab(df[col], df[target])
        chi2, p, dof, _ = chi2_contingency(ct)
        rows.append({"feature": col, "chi2": chi2, "dof": dof, "p_value": p})
    out = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    out["significant_at_0.05"] = out["p_value"] < 0.05
    return out
