"""Build the four supplementary notebooks under ``notebooks/``.

Each one is a deep-dive that complements ``main.ipynb`` and is owned by
exactly one team member, so the four notebooks can grow independently
without merge conflicts.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def _cid() -> str:
    return uuid.uuid4().hex[:8]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "id": _cid(),
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "id": _cid(),
        "outputs": [],
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


def save(cells: list[dict], filename: str) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
                "mimetype": "text/x-python",
                "pygments_lexer": "ipython3",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NB_DIR / filename).write_text(json.dumps(nb, indent=1))
    print(f"  wrote {NB_DIR / filename}")


SETUP_CELL = r"""
import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
np.random.seed(42)
"""


# ============================================================================
# 01 — Data exploration (Yiou Wang)
# ============================================================================
def build_01():
    cells = [
        md(r"""
        # 01 · Data Exploration & Preprocessing

        **Owner: Yiou Wang.**

        Deep-dive companion to §2 of `main.ipynb`. We dig into the
        statistical structure of the UCI Heart Disease (Cleveland)
        dataset and validate the clinical bin choices used in
        `src/preprocessing.py`.
        """),
        code(SETUP_CELL),
        code(r"""
        from src.data_loader import load_heart_disease, describe_schema
        df_raw = load_heart_disease()
        print(df_raw.shape)
        df_raw.describe(include='all').T.head(15)
        """),
        md(r"""
        ## Continuous variable distributions

        Are the clinical bin edges (JNC-7 / ATP-III etc.) reasonable
        given *this* sample? We overlay the chosen cut points on the
        empirical KDE.
        """),
        code(r"""
        from src.preprocessing import DISCRETIZATION
        cont_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        for ax, var in zip(axes.flat, cont_vars):
            sns.kdeplot(df_raw[var].dropna(), fill=True, ax=ax)
            edges, _ = DISCRETIZATION[var]
            for e in edges[1:-1]:
                ax.axvline(e, color='crimson', linestyle='--', linewidth=1)
            ax.set_title(var)
        for ax in axes.flat[len(cont_vars):]:
            ax.axis('off')
        plt.tight_layout(); plt.show()
        """),
        md(r"""
        ## Stratified by outcome

        How separable are the continuous variables once we condition
        on disease status?
        """),
        code(r"""
        df_raw_bin = df_raw.copy()
        df_raw_bin['target'] = (df_raw_bin['num'] > 0).astype(int)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, v in zip(axes, ['age', 'chol', 'thalach']):
            sns.kdeplot(data=df_raw_bin, x=v, hue='target', fill=True, common_norm=False, ax=ax)
            ax.set_title(v)
        plt.tight_layout(); plt.show()
        """),
        md(r"""
        ## Correlation vs. mutual information

        A correlation matrix can miss non-monotonic dependence. We
        contrast both views.
        """),
        code(r"""
        from src.preprocessing import PreprocessConfig, build_dataset
        from src.eda import mutual_information_matrix, plot_mi_heatmap

        df = build_dataset(df_raw, PreprocessConfig())
        mi = mutual_information_matrix(df)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        df_num = df_raw.replace('?', np.nan).apply(pd.to_numeric, errors='coerce')
        sns.heatmap(df_num.corr().abs(), ax=axes[0], cmap='rocket_r', annot=True, fmt='.2f',
                    cbar_kws={'label': '|Pearson r|'})
        axes[0].set_title('|Pearson correlation| (raw)')
        sns.heatmap(mi, ax=axes[1], cmap='rocket_r', annot=True, fmt='.2f',
                    cbar_kws={'label': 'MI (nats)'})
        axes[1].set_title('Mutual information (discretized)')
        plt.tight_layout(); plt.show()
        """),
        md(r"""
        ## χ² test of independence with the target
        """),
        code(r"""
        from src.eda import chi_square_table
        chi_square_table(df)
        """),
    ]
    save(cells, "01_data_exploration.ipynb")


# ============================================================================
# 02 — Structure learning (Qicheng Jin)
# ============================================================================
def build_02():
    cells = [
        md(r"""
        # 02 · Structure Learning Deep Dive

        **Owner: Qicheng Jin.**

        Companion to §4 of `main.ipynb`. We test the *sensitivity* of
        the learned DAG to hyperparameter choices — scoring function,
        maximum in-degree, PC significance level — so we can defend
        our final choice in the presentation.
        """),
        code(SETUP_CELL),
        code(r"""
        from src.data_loader import load_heart_disease
        from src.preprocessing import (
            PreprocessConfig, build_dataset, train_test_split_df, variable_state_names,
        )
        from src.expert_network import build_expert_dag
        from src.structure_learning import (
            StructureSearchConfig, learn_hill_climb, learn_pc,
            compare_structures, edge_set_diff,
        )

        df = build_dataset(load_heart_disease(), PreprocessConfig())
        train, test = train_test_split_df(df, PreprocessConfig())
        expert = build_expert_dag()
        """),
        md(r"""
        ## Sensitivity to scoring function
        """),
        code(r"""
        from src.visualization import plot_dag
        dags = {}
        for scoring in ['bic', 'k2', 'bdeu']:
            cfg = StructureSearchConfig(scoring=scoring)
            dags[f'HC-{scoring}'] = learn_hill_climb(train, cfg)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for ax, (name, g) in zip(axes, dags.items()):
            plot_dag(g, title=f'{name}  ({g.number_of_edges()} edges)', ax=ax)
        plt.show()
        """),
        md(r"""
        ## Sensitivity to maximum in-degree

        Restricting in-degree is a common regularization technique —
        too high and the model overfits, too low and it underfits.
        """),
        code(r"""
        rows = []
        for k in [1, 2, 3, 4, 5]:
            g = learn_hill_climb(train, StructureSearchConfig(max_indegree=k))
            rows.append({'max_indegree': k, 'edges': g.number_of_edges()})
        pd.DataFrame(rows)
        """),
        md(r"""
        ## Sensitivity to PC α

        Higher α produces a denser skeleton.
        """),
        code(r"""
        rows = []
        for alpha in [0.001, 0.01, 0.05, 0.1]:
            g = learn_pc(train, StructureSearchConfig(pc_alpha=alpha))
            rows.append({'alpha': alpha, 'edges': g.number_of_edges()})
        pd.DataFrame(rows)
        """),
        md(r"""
        ## Whitelist / blacklist experiment

        What if we *insist* that all `target → manifestation` expert
        edges remain in the learned DAG?
        """),
        code(r"""
        forced = [(u, v) for (u, v) in expert.edges() if u == 'target']
        cfg = StructureSearchConfig(fixed_edges=forced)
        constrained = learn_hill_climb(train, cfg)
        print(f'Constrained DAG: {constrained.number_of_edges()} edges')

        diff = edge_set_diff(expert, constrained)
        print('Edges added by data on top of forced expert spine:')
        for e in diff['only_in_learned']:
            print(' ', e[0], '↔', e[1])
        """),
    ]
    save(cells, "02_structure_learning.ipynb")


# ============================================================================
# 03 — Inference & MCMC (Chenqi Wang)
# ============================================================================
def build_03():
    cells = [
        md(r"""
        # 03 · Inference & MCMC Deep Dive

        **Owner: Chenqi Wang.**

        Companion to §§6–8 of `main.ipynb`. We measure MCMC
        convergence quantitatively and replicate the exact answer with
        two different samplers.
        """),
        code(SETUP_CELL),
        code(r"""
        from src.data_loader import load_heart_disease
        from src.preprocessing import (
            PreprocessConfig, build_dataset, train_test_split_df, variable_state_names,
        )
        from src.expert_network import build_expert_dag
        from src.parameter_learning import ParameterFitConfig, fit_parameters
        from src.inference import make_engine, posterior

        df = build_dataset(load_heart_disease(), PreprocessConfig())
        train, test = train_test_split_df(df, PreprocessConfig())
        states = variable_state_names(df)
        bn = fit_parameters(build_expert_dag(), train, ParameterFitConfig(method='bayes'), state_names=states)
        engine = make_engine(bn)
        """),
        md(r"""
        ## Effect of burn-in & sample size on MH

        We sweep the post-burn-in sample size and plot the TV distance
        to the exact Variable-Elimination posterior.
        """),
        code(r"""
        from src.mcmc import MHConfig, metropolis_hastings

        evidence = {'age': '65+', 'sex': '1', 'chol': 'high', 'trestbps': 'hyper'}
        exact = posterior(engine, 'target', evidence)

        rows = []
        for n in [200, 500, 1000, 2000, 4000, 8000]:
            cfg = MHConfig(n_samples=n, burn_in=500, seed=0)
            mh_post, _ = metropolis_hastings(bn, evidence, query='target', cfg=cfg)
            tv = 0.5 * np.abs(mh_post.values - exact.values).sum()
            rows.append({'n_samples': n, 'TV(MH, exact)': tv})
        df_conv = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df_conv['n_samples'], df_conv['TV(MH, exact)'], 'o-')
        ax.set_xscale('log'); ax.set_xlabel('post-burn-in samples'); ax.set_ylabel('TV distance to exact')
        ax.set_title('MH convergence to exact posterior')
        plt.show()
        df_conv
        """),
        md(r"""
        ## Gibbs vs. MH on the same query
        """),
        code(r"""
        from src.mcmc import GibbsConfig, gibbs_posterior
        gibbs_post, _ = gibbs_posterior(bn, evidence, query='target',
                                        cfg=GibbsConfig(n_samples=6000, burn_in=1000))
        mh_post, _ = metropolis_hastings(bn, evidence, query='target',
                                         cfg=MHConfig(n_samples=6000, burn_in=1000))
        pd.DataFrame({'exact': exact, 'Gibbs': gibbs_post, 'MH': mh_post})
        """),
        md(r"""
        ## Counterfactual sensitivity

        How much does P(disease) change as we sweep each modifiable
        risk factor?
        """),
        code(r"""
        from src.inference import do_intervention

        baseline = posterior(engine, 'target', evidence)
        positive = sorted(bn.get_cpds('target').state_names['target'])[-1]
        baseline_risk = float(baseline.loc[positive])

        rows = []
        for var in ['chol', 'trestbps', 'fbs']:
            for val in sorted(set(states[var])):
                cf = do_intervention(bn, {var: val}, query='target')
                rows.append({'variable': var, 'do(value)': val,
                             'P(disease | do)': float(cf.loc[positive]),
                             'Δ from baseline': float(cf.loc[positive]) - baseline_risk})
        pd.DataFrame(rows).sort_values(['variable', 'do(value)'])
        """),
    ]
    save(cells, "03_inference_mcmc.ipynb")


# ============================================================================
# 04 — Evaluation & Uncertainty (Jingyuan Wang)
# ============================================================================
def build_04():
    cells = [
        md(r"""
        # 04 · Evaluation, Uncertainty & Decision Theory

        **Owner: Jingyuan Wang.**

        Companion to §§9–10 of `main.ipynb`. We add learning-curve
        analysis, threshold sensitivity, and per-subgroup metrics.
        """),
        code(SETUP_CELL),
        code(r"""
        from src.data_loader import load_heart_disease
        from src.preprocessing import (
            PreprocessConfig, build_dataset, train_test_split_df, variable_state_names,
        )
        from src.expert_network import build_expert_dag
        from src.parameter_learning import ParameterFitConfig, fit_parameters
        from src.inference import predict_proba

        df = build_dataset(load_heart_disease(), PreprocessConfig())
        train, test = train_test_split_df(df, PreprocessConfig())
        states = variable_state_names(df)
        bn = fit_parameters(build_expert_dag(), train, ParameterFitConfig(method='bayes'), state_names=states)
        positive = sorted(bn.get_cpds('target').state_names['target'])[-1]
        bn_probs = predict_proba(bn, test, target='target')[positive].values
        y_test = test['target'].astype(int).values
        """),
        md(r"""
        ## Learning curve

        How well does the Bayesian Network perform with only k% of
        the training fold? This tells us how data-hungry the model is.
        """),
        code(r"""
        from sklearn.metrics import roc_auc_score, log_loss

        fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        rows = []
        for frac in fractions:
            sub = train.sample(frac=frac, random_state=0)
            bn_k = fit_parameters(build_expert_dag(), sub, ParameterFitConfig(method='bayes'), state_names=states)
            probs = predict_proba(bn_k, test, target='target')[positive].values
            rows.append({
                'fraction': frac,
                'n_train': len(sub),
                'AUC': roc_auc_score(y_test, probs),
                'log_loss': log_loss(y_test, np.clip(probs, 1e-7, 1-1e-7)),
            })
        lc = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(lc['fraction'], lc['AUC'], 'o-'); axes[0].set_title('Test AUC'); axes[0].set_xlabel('train fraction')
        axes[1].plot(lc['fraction'], lc['log_loss'], 'o-', color='crimson'); axes[1].set_title('Test log-loss')
        plt.show()
        lc
        """),
        md(r"""
        ## Per-subgroup metrics

        Group AUC by sex and age band — does the model perform
        equitably?
        """),
        code(r"""
        df_test = test.copy()
        df_test['prob'] = bn_probs
        df_test['y'] = y_test

        rows = []
        for sex_val, grp in df_test.groupby('sex', observed=True):
            if grp['y'].nunique() < 2: continue
            rows.append({'subgroup': f'sex={sex_val}', 'n': len(grp),
                         'AUC': roc_auc_score(grp['y'], grp['prob'])})
        for age_val, grp in df_test.groupby('age', observed=True):
            if grp['y'].nunique() < 2: continue
            rows.append({'subgroup': f'age={age_val}', 'n': len(grp),
                         'AUC': roc_auc_score(grp['y'], grp['prob'])})
        pd.DataFrame(rows)
        """),
        md(r"""
        ## Threshold sensitivity under different FN:FP cost ratios
        """),
        code(r"""
        from src.uncertainty import UtilityMatrix, optimal_threshold

        rows = []
        for ratio in [1, 2, 5, 10, 20, 50]:
            util = UtilityMatrix(cost_fp=1.0, cost_fn=float(ratio))
            t, _ = optimal_threshold(y_test, bn_probs, utility=util)
            rows.append({'FN:FP ratio': ratio, 'optimal threshold': t})
        pd.DataFrame(rows)
        """),
        md(r"""
        ## Width of credible interval by confidence band

        Where in the probability range is the BN most *uncertain*?
        """),
        code(r"""
        from src.uncertainty import UncertaintyConfig, posterior_predictive

        uq = posterior_predictive(build_expert_dag(), train, test, state_names=states,
                                  cfg=UncertaintyConfig(n_posterior_samples=15))
        widths = uq['ci_high'] - uq['ci_low']
        bands = pd.cut(uq['mean'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
        pd.DataFrame({'mean_band': bands, 'CI_width': widths}).groupby('mean_band', observed=True)['CI_width'].agg(['mean', 'count'])
        """),
    ]
    save(cells, "04_evaluation_uncertainty.ipynb")


def main():
    print("Building supplementary notebooks ...")
    build_01()
    build_02()
    build_03()
    build_04()


if __name__ == "__main__":
    main()
