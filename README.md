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
│ (cardiology)│    │ DAG (HC / PC) │    │ (LR / RF / XGB)│
│             │    │               │    │                │
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

We compare **three** candidate DAGs in `main.ipynb` and rank them by BIC / K2 / BDeu network scores. A fourth method (MMHC) is implemented in `src/structure_learning.py` for completeness but is not run end-to-end in `main.ipynb` (it is comparatively slow on this small sample and adds little signal beyond Hill-Climb here).

| # | Strategy | Idea | Module | In `main.ipynb`? |
|---|---|---|---|---|
| 1 | **Expert (cardiology)** | Hand-crafted DAG encoding established cardiology knowledge. | `src/expert_network.py` | ✓ |
| 2 | **Hill-Climb (BIC)** | Greedy edge-add/remove/reverse over the BIC scoring function. | `src/structure_learning.py` | ✓ |
| 3 | **PC algorithm** | Constraint-based: build the skeleton via χ² conditional independence tests, orient v-structures. | `src/structure_learning.py` | ✓ |
| 4 | **MMHC (hybrid)** | Max-Min skeleton + Hill-Climb inside it. | `src/structure_learning.py` | implemented, not benchmarked live |

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

### 5.1 — Held-out benchmark (single 80/20 split, seed = 42)

Numbers below are the actual values produced by `main.ipynb`, sorted by ROC-AUC. Because this is a **single train/test split** on ~300 patients, small differences should be read as ties rather than rankings.

| Model | Accuracy | ROC-AUC | Avg. precision | Brier ↓ | Log-loss ↓ | ECE ↓ |
|---|---|---|---|---|---|---|
| Random Forest                  | 0.833 | **0.938** | **0.931** | **0.110** | 0.352 | 0.172 |
| **BN — Hill-Climb DAG (BDeu)** | 0.783 | 0.935 | 0.921 | 0.122 | **0.350** | 0.120 |
| Logistic Regression            | 0.817 | 0.917 | 0.901 | 0.120 | 0.372 | **0.094** |
| XGBoost                        | 0.833 | 0.916 | 0.903 | 0.127 | 0.406 | 0.124 |
| **BN — expert DAG (BDeu)**     | 0.733 | 0.872 | 0.842 | 0.178 | 0.654 | 0.184 |

**Take-away:** the head-to-head is a *trade-off*, not a single winner.

- **Random Forest** is strongest on discrimination (ROC-AUC, average precision) and Brier.
- **BN — Hill-Climb DAG** is right behind on discrimination and takes the best `log_loss`.
- **Logistic Regression** is the best-calibrated model by ECE on this split.
- **BN — expert DAG** is the weakest predictor but the most defensible to a cardiologist (every edge has a written rationale).

The Bayesian Network's value is therefore *not* "wins every metric"; it is **competitive predictive performance combined with structural interpretability, Pearl-style counterfactual queries, and per-patient credible intervals** — capabilities the discriminative baselines do not have.

### 5.2 — Reliability diagrams

`main.ipynb` plots each model's reliability curve side-by-side. The diagonal target — *"when the model says 60%, ~60% of those patients truly have disease"* — is approached by Logistic Regression most closely on this split; the BN variants are reasonable but not best.

### 5.3 — Counterfactuals

Sample question we can answer with the BN but not the baselines:

> **For a high-risk patient (`age = 65+`, `sex = 1`, `chol = high`, `trestbps = hyper`, `exang = 1`), how much would the predicted risk drop if we (hypothetically) normalized cholesterol or blood pressure?**

```python
posterior(engine, "target", evidence)
# →  P(disease | observed)              = 0.83

do_intervention(bn, {"chol": "desirable"})
# →  P(disease | do(chol = desirable))  = 0.42   (Δ = −0.41)

do_intervention(bn, {"trestbps": "normal"})
# →  P(disease | do(BP = normal))       = 0.41   (Δ = −0.42)

do_intervention(bn, {"chol": "desirable", "trestbps": "normal"})
# →  P(disease | do(chol+BP normalized))= 0.32   (Δ = −0.52)
```

### 5.4 — MCMC vs. exact

On the same representative query, with 3 000 post-burn-in samples:

| | P(target = 0) | P(target = 1) |
|---|---|---|
| Exact (Variable Elimination) | 0.169 | 0.831 |
| Gibbs                        | 0.186 | 0.814 |
| Metropolis-Hastings          | 0.143 | 0.857 |

Both samplers land within ~0.03 of the exact posterior — close enough to be a useful sanity check on both the network and the inference implementations. Trace plots, autocorrelation, and convergence-vs-sample-size curves are in `notebooks/03_inference_mcmc.ipynb`.

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
