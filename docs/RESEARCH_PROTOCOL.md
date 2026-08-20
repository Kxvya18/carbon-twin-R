# Research Protocol

## 1. Data engineering

1. Raw UCI file is immutable (`data/raw`).
2. SHA-256 lineage manifest is created.
3. Quality gates check schema, duplicates, timestamp validity, negative targets and missingness.
4. Random negative-control features `rv1` and `rv2` are removed.
5. Clean data is written to Parquet (`data/silver`).
6. Feature construction is deterministic and leakage-aware.

## 2. Temporal split discipline

Data are never randomly shuffled.

- Train: 60%
- Model-selection holdout: 15%
- Conformal calibration: 10%
- Final test: 15%

Hyperparameter tuning uses `TimeSeriesSplit` **inside the training period only**. The selection period chooses the model family. The conformal block calibrates uncertainty. The test block is evaluated once after final refit.

## 3. Model selection

Core candidates:

- Elastic Net — linear/sparse baseline;
- Random Forest — nonlinear bagged trees;
- Extra Trees — high-variance-reduction ensemble;
- Histogram Gradient Boosting — boosted nonlinear tabular model.

Each family receives sampled hyperparameter search over expanding temporal folds. Selection is based on later-period MAE, with CV MAE/std retained for stability inspection.

Advanced XGBoost/temporal/GNN experiments are optional. They should be promoted only if untouched-period evaluation justifies them.

## 4. Fine-tuning policy

"Fine-tuning" is not used as a synonym for hyperparameter tuning.

- Tree/linear models: **hyperparameter tuning**.
- Neural temporal model: **pretraining + low-learning-rate fine-tuning** on a later clean adaptation window.
- A drift report (KS statistic + PSI) is generated to determine whether adaptation is plausible.
- Pretrained vs fine-tuned performance is compared on an untouched evaluation window.
- If fine-tuning does not improve the held-out period, the project explicitly reports that result.

## 5. Counterfactual benchmark

The original measured test target is treated as the known healthy reference for controlled benchmark experiments. A fault is injected into a copy of the test period, creating `observed_target`. Thus the true additional energy is known exactly.

This makes Carbon Debt estimation quantitatively testable instead of anecdotal.

## 6. Statistical graph

The room graph is not hand-drawn. Graphical Lasso is fit on training-only room temperature/humidity data to learn sparse conditional dependencies. Fault/test data do not participate in topology estimation.

## 7. Uncertainty and sequential statistics

- split conformal interval calibrated on a dedicated block;
- Kalman latent degradation state;
- CUSUM;
- Bayesian-style online change score;
- SPRT.

These components are evaluated as detectors, not all assumed to be equally useful.

## 8. Causal boundary

The causal module uses a known-effect semi-synthetic intervention generated from **real measured covariates**. Double ML is judged by recovery error against the known treatment effect.

The project does not claim that observational feature importance is causal.

## 9. Reporting

Every headline result should include:

- data split;
- model/hyperparameters;
- MAE/RMSE/R²;
- conformal empirical coverage;
- fault F1/detection delay across scenarios;
- Carbon Debt error;
- graph localization accuracy;
- uncertainty over model comparisons via block bootstrap;
- limitations.
