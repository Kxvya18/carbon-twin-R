# Research Upgrade Changelog — v0.2

## Replaced / upgraded

- Synthetic-only default → real UCI measured building-energy default.
- One fixed gradient-boosting model → temporal-CV model family + hyperparameter selection.
- Three-way split → train / selection / conformal / final-test protocol.
- Hard-coded facility graph → Graphical-Lasso learned room graph.
- Static Streamlit results page → interactive research simulator.
- Informal "fine-tuning" language → explicit distinction between hyperparameter tuning and neural fine-tuning.
- No data lineage → immutable raw layer, source manifest, SHA-256 and quality report.
- No Gold layer → deterministic model matrix with split labels.
- Generic causal demo → known-effect semi-synthetic DML benchmark on real covariates.

## Preserved

- Conformal uncertainty.
- Kalman latent state.
- CUSUM / change score / SPRT.
- Carbon Debt concept.
- Pareto optimization.
- Legacy synthetic pipeline for unit/debug work only.

## Validation performed in this build environment

- Python compile check: passed.
- Unit/protocol tests: **13 passed**.
- Offline end-to-end smoke test using an UCI-shaped time-series fixture: passed for prepare → tune/select → simulate → graph localization → Carbon Debt.
- Network download of the 11.4 MB UCI file could not be executed inside the build sandbox, so the packaged downloader is based on the current official UCI static dataset endpoint and must be run on the user's machine.
