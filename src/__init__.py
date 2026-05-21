"""Bayesian Network for Heart Disease Risk Assessment.

A probabilistic framework for interpretable, uncertainty-aware heart disease
risk assessment built on the UCI Heart Disease dataset.

Package layout
--------------
- ``data_loader``       Fetch and cache the UCI Heart Disease dataset.
- ``preprocessing``     Clean, impute, discretize, and split the data.
- ``eda``               Exploratory data analysis utilities.
- ``expert_network``    Hand-crafted (domain-driven) Bayesian Network DAG.
- ``structure_learning`` Data-driven structure learning (Hill Climbing, PC).
- ``parameter_learning`` MLE and Bayesian (Dirichlet) parameter estimation.
- ``inference``         Exact inference via Variable Elimination.
- ``mcmc``              Gibbs and Metropolis-Hastings posterior sampling.
- ``baselines``         Logistic Regression / Random Forest / XGBoost.
- ``evaluation``        Discrimination, calibration, probabilistic scoring.
- ``uncertainty``       Epistemic uncertainty via posterior CPD sampling.
- ``decision``          Decision-theoretic utility analysis.
- ``visualization``     DAG plotting, calibration curves, etc.
"""

__version__ = "1.0.0"
__authors__ = (
    "Yiou Wang — Data & EDA",
    "Qicheng Jin — Structure Learning",
    "Qichen Wang — Parameter Learning & Inference",
    "Jingyuan Wang — Evaluation & Uncertainty",
)
