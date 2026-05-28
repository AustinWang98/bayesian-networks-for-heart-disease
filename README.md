# Bayesian Networks for Heart Disease Risk Assessment

> A probabilistic framework that combines **clinical domain knowledge** with **data-driven structure learning** to produce *interpretable*, *uncertainty-aware* heart disease risk predictions on the UCI Heart Disease (Cleveland) dataset.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="pgmpy" src="https://img.shields.io/badge/pgmpy-0.1.25%2B-purple">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Status" src="https://img.shields.io/badge/status-final--project-orange">
</p>

---

## Table of contents

1. [Why Bayesian Networks?](#1--why-bayesian-networks)
2. [Dataset](#2--dataset)
3. [System architecture](#3--system-architecture)
4. [Methodology](#4--methodology)
5. [Results](#5--results)
6. [How to run](#6--how-to-run)
7. [Repository structure](#7--repository-structure)
8. [Team & contributions](#8--team--contributions)
9. [References](#9--references)

---

## 1 · Why Bayesian Networks?

Most heart-disease classifiers in the literature (logistic regression, random forests, gradient boosting) deliver excellent ROC-AUC numbers but answer only one question: *"Is this patient sick — yes or no?"*

In a clinical setting, three other questions matter just as much:

| Question | What a black-box classifier gives you | What a Bayesian Network gives you |
| --- | --- | --- |
| **How confident are we?** | A single point estimate of `P(disease)`. | A full posterior `P(disease \| evidence)` plus a credible interval over the parameters. |
| **Why?** | Feature importances at the population level. | A directed acyclic graph encoding the *mechanism* by which evidence updates belief. |
| **What if we intervened?** | Out of scope. | Pearl's `do`-operator answers counterfactual queries like `P(disease \| do(chol = desirable))`. |

A Bayesian Network (BN) is therefore a strict generalization of the classification task that also delivers **decision-theoretic** outputs.

---

## 2 · Dataset

We use the **UCI Heart Disease (Cleveland)** dataset — 303 patients × 14 attributes, the de facto benchmark for this problem since 1988.

- **Source:** <https://archive.ics.uci.edu/dataset/45/heart+disease>
- **Backup:** <https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data>
- **Target:** binarized from the 5-level severity score `num` ∈ {0,1,2,3,4} → `target` ∈ {0, 1}.

Continuous variables (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) are discretized using **clinically-motivated thresholds** (JNC-7 for blood pressure, ATP-III for cholesterol, etc.) rather than equal-frequency bins. This keeps the resulting CPDs interpretable to cardiologists.

```text
age      : <45  | 45-54 | 55-64 | 65+
trestbps : normal (<120) | prehyper (120-139) | hyper (≥140)
chol     : desirable (<200) | borderline (200-239) | high (≥240)
thalach  : low (<140) | mid (140-169) | high (≥170)
oldpeak  : none (<1) | mild (1-2) | marked (>2)
```

---

## 3 · System architecture

```
                  ┌──────────────────┐
                  │  UCI fetch + cache│  (data_loader.py)
                  └────────┬─────────┘
                           │
                  ┌────────▼─────────┐
                  │ Preprocess + bin │  (preprocessing.py)
                  └────────┬─────────┘
                           │
       ┌───────────────────┼────────────────────┐
       │                   │                    │
┌──────▼──────┐    ┌───────▼───────┐    ┌───────▼────────┐
│ Expert DAG  │    │ Data-driven   │    │ Baseline ML    │
│ (cardiology)│    │ DAG (HC/PC/   │    │ (LR / RF / XGB)│
│             │    │  MMHC)        │    │                │
└──────┬──────┘    └───────┬───────┘    └───────┬────────┘
       │                   │                    │
       └──────────┬────────┘                    │
                  │                             │
         ┌────────▼─────────┐                   │
         │ Parameter fit    │                   │
         │ (MLE / BDeu)     │                   │
         └────────┬─────────┘                   │
                  │                             │
       ┌──────────┼───────────┐                 │
       │          │           │                 │
┌──────▼─┐ ┌──────▼───┐ ┌─────▼──────┐          │
│Exact VE│ │Gibbs MCMC│ │Metropolis- │          │
│        │ │          │ │Hastings    │          │
└──────┬─┘ └──────┬───┘ └─────┬──────┘          │
       │          │           │                 │
       └──────────┴─────┬─────┘                 │
                        │                       │
                ┌───────▼────────┐     ┌────────▼─────────┐
                │ Evaluation +   │◄────┤ Discrimination,  │
                │ uncertainty    │     │ calibration,     │
                │ + decision     │     │ probabilistic    │
                └────────────────┘     │ scoring          │
                                       └──────────────────┘
```

Every box is a tested, reusable module in `src/` (see [Repository structure](#7--repository-structure)).

---

## 4 · Methodology

### 4.1 — Structure learning

We compare **four** candidate DAGs and pick the best by held-out log-likelihood and BIC.

| # | Strategy | Idea | Module |
|---|---|---|---|
| 1 | **Expert (cardiology)** | Hand-crafted DAG encoding established cardiology knowledge. | `src/expert_network.py` |
| 2 | **Hill-Climb (BIC)** | Greedy edge-add/remove/reverse over the BIC scoring function. | `src/structure_learning.py` |
| 3 | **PC algorithm** | Constraint-based: build the skeleton via χ² conditional independence tests, orient v-structures. | `src/structure_learning.py` |
| 4 | **MMHC (hybrid)** | Max-Min skeleton + Hill-Climb inside it — usually the strongest small-data DAG learner. | `src/structure_learning.py` |

### 4.2 — Parameter learning

For each DAG we estimate the conditional probability distributions (CPDs) two ways and compare:

- **Maximum likelihood (MLE)** — closed-form, but fragile with sparse parent configurations.
- **Bayesian (Dirichlet / BDeu prior, ESS = 10)** — recommended for `n ≈ 300` patients.

### 4.3 — Inference

| Method | Use case |
|---|---|
| **Variable Elimination** (`src/inference.py`) | Fast exact marginal `P(target \| evidence)`. |
| **Gibbs sampling** (`src/mcmc.py`) | Per-class proposal requirement; benchmarks exact result. |
| **Metropolis-Hastings** (`src/mcmc.py`) | Implemented from scratch on top of the Markov blanket. Records acceptance rate, autocorrelation, ESS. |
| **`do`-operator** (`src/inference.py`) | Pearl's interventional query — implemented via graph mutilation. |

### 4.4 — Uncertainty & decision theory

Two layers of uncertainty are reported:

1. **Aleatoric** — captured by `P(target = 1 | evidence)` itself.
2. **Epistemic** — captured by **posterior-bootstrap CPD sampling**: we resample the training data, refit the BN parameters, run inference, and report a 95% credible interval around the prediction.

A **utility-aware threshold** is then chosen to minimize expected cost under a clinical asymmetry (false negatives ≫ false positives).

---

## 5 · Results

> Numbers in this section are produced by `main.ipynb`. Re-run it end-to-end to refresh.

### 5.1 — Held-out benchmark (typical run)

| Model | Accuracy | ROC-AUC | Avg. precision | Brier ↓ | Log-loss ↓ | ECE ↓ |
|---|---|---|---|---|---|---|
| **BN — expert DAG (BDeu)**       | 0.83 | 0.91 | 0.91 | 0.13 | 0.42 | 0.06 |
| **BN — Hill-Climb DAG (BDeu)**   | 0.85 | 0.92 | 0.92 | 0.12 | 0.39 | 0.05 |
| Logistic Regression              | 0.85 | 0.91 | 0.90 | 0.13 | 0.41 | 0.07 |
| Random Forest                    | 0.83 | 0.90 | 0.89 | 0.14 | 0.45 | 0.10 |
| XGBoost                          | 0.84 | 0.91 | 0.90 | 0.13 | 0.42 | 0.09 |

**Take-away:** the Bayesian Network is *competitive on discrimination* (within sampling noise of the best baseline) and **better calibrated** (lower ECE). On top of that it provides interpretable structure, credible intervals, and counterfactual reasoning — which the baselines do not.

### 5.2 — Reliability diagrams

We plot each model's reliability curve in `main.ipynb`. The BN sits closer to the diagonal — i.e. when it says *"60% risk"*, ~60% of those patients really do have disease.

### 5.3 — Counterfactuals

Sample question we can answer with the BN but not the baselines:

> **For a 60-year-old male with high cholesterol and prehypertension, how much would his predicted risk drop if we (hypothetically) restored his cholesterol to the desirable range?**

```python
posterior(engine, "target", evidence)
# →  P(disease | observed) = 0.71

do_intervention(bn, {"chol": "desirable"}, query="target")
# →  P(disease | do(chol = desirable)) = 0.53   (Δ = −0.18)
```

### 5.4 — MCMC vs. exact

The Gibbs and Metropolis-Hastings posteriors converge to within 0.01 of the Variable-Elimination answer after ~2 000 post-burn-in samples on a representative query (see `notebooks/03_inference_mcmc.ipynb` for trace plots, autocorrelation, and effective sample size).

---

## 6 · How to run

### 6.1 — Set up the environment

```bash
git clone <this-repo>
cd <this-repo>

python -m venv .venv
source .venv/bin/activate     # macOS / Linux
# .venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### 6.2 — Get the data

The UCI dataset is fetched and cached automatically on first use, but you can pre-warm the cache:

```bash
python scripts/download_data.py
```

### 6.3 — Reproduce the report

```bash
jupyter notebook main.ipynb
```

Run all cells. End-to-end runtime is ~2 minutes on a laptop.

### 6.4 — Try the interactive app

```bash
streamlit run app/streamlit_app.py
```

A browser tab opens with sliders for every clinical feature and a live posterior bar chart.

### 6.5 — Sanity tests

```bash
pytest -q
```

---

## 7 · Repository structure

```
.
├── README.md                  ← (this file)
├── main.ipynb                 ← presentation notebook (run me!)
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                   ← cached UCI CSV lives here
│   └── processed/             ← (created on demand)
│
├── src/
│   ├── data_loader.py         ← UCI download + caching       [Yiou Wang]
│   ├── preprocessing.py       ← clinical binning + splits    [Yiou Wang]
│   ├── eda.py                 ← MI matrix, χ², plots         [Yiou Wang]
│   ├── expert_network.py      ← hand-crafted DAG             [Qicheng Jin]
│   ├── structure_learning.py  ← HC / PC / MMHC               [Qicheng Jin]
│   ├── parameter_learning.py  ← MLE & Bayesian CPDs          [Chenqi Wang]
│   ├── inference.py           ← VE + Pearl's do-operator     [Chenqi Wang]
│   ├── mcmc.py                ← Gibbs & Metropolis-Hastings  [Chenqi Wang]
│   ├── baselines.py           ← LR / RF / XGB                [Jingyuan Wang]
│   ├── evaluation.py          ← metrics + reliability        [Jingyuan Wang]
│   ├── uncertainty.py         ← bootstrap CIs + decision     [Jingyuan Wang]
│   └── visualization.py       ← shared plotting helpers      [all]
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_structure_learning.ipynb
│   ├── 03_inference_mcmc.ipynb
│   └── 04_evaluation_uncertainty.ipynb
│
├── app/
│   └── streamlit_app.py       ← interactive demo
│
├── scripts/
│   └── download_data.py
│
└── tests/
    └── test_pipeline.py
```

---

## 8 · Team & contributions

We split the work to play to each teammate's interests while keeping a clean separation of concerns.

| Member | Role | Modules owned |
|---|---|---|
| **Yiou Wang** | Data engineering, EDA, clinical binning | `data_loader.py`, `preprocessing.py`, `eda.py`, `notebooks/01_data_exploration.ipynb` |
| **Qicheng Jin** | Structure learning, DAG design & comparison | `expert_network.py`, `structure_learning.py`, `notebooks/02_structure_learning.ipynb` |
| **Chenqi Wang** | Parameter learning, exact inference, MCMC samplers, counterfactuals | `parameter_learning.py`, `inference.py`, `mcmc.py`, `notebooks/03_inference_mcmc.ipynb` |
| **Jingyuan Wang** | Baselines, evaluation, uncertainty, decision theory, Streamlit app | `baselines.py`, `evaluation.py`, `uncertainty.py`, `app/streamlit_app.py`, `notebooks/04_evaluation_uncertainty.ipynb` |

`main.ipynb` is co-authored — each section is owned by the corresponding member.

---

## 9 · References

1. Detrano, R. et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* American Journal of Cardiology, 64(5), 304–310. — original UCI Heart Disease paper.
2. Koller, D. & Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques.* MIT Press.
3. Pearl, J. (2009). *Causality.* Cambridge University Press. — `do`-operator.
4. Heckerman, D., Geiger, D. & Chickering, D. (1995). *Learning Bayesian Networks: The Combination of Knowledge and Statistical Data.* — BDeu prior.
5. Tsamardinos, I., Brown, L. E. & Aliferis, C. F. (2006). *The max-min hill-climbing Bayesian network structure learning algorithm.* — MMHC.
6. Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. (2017). *On Calibration of Modern Neural Networks.* — ECE definition.
7. pgmpy documentation — <https://pgmpy.org>.

---

<sub>Final project · Generative AI & Bayesian Methods · Week 10 presentation.</sub>
