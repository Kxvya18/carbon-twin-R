# CarbonTwin-R v0.2 — Research Simulator

**Causal temporal/graph machine learning for counterfactual detection of avoidable carbon emissions.**

CarbonTwin-R now uses a **real measured building-energy dataset by default**, applies explicit data-engineering and leakage controls, performs time-aware model/hyperparameter selection, calibrates uncertainty on a separate holdout, learns a statistical room graph, and exposes the system as an **interactive simulator rather than a static dashboard**.

## What changed from v0.1

- Default research data: **UCI Appliances Energy Prediction**, not generated facility data.
- Immutable raw → validated Silver Parquet → model artifacts and lineage manifests.
- Explicit removal of UCI's random negative-control variables `rv1`/`rv2`.
- Four chronological partitions: train / model-selection / conformal-calibration / final test.
- Hyperparameter search uses `TimeSeriesSplit`; no random shuffled CV.
- Model comparison: Elastic Net, Random Forest, Extra Trees, Histogram Gradient Boosting.
- Optional neural **pretraining + fine-tuning** is separate from hyperparameter tuning and is only claimed if it improves an untouched period.
- Graphical Lasso learns the nine-zone conditional-dependence graph from training data only.
- Fault simulation happens on the real held-out time series, preserving exact healthy counterfactual ground truth.
- Streamlit app is now an interactive **Fault / Digital-Twin / Intervention Simulator**.
- Causal DML is validated against a known-effect semi-synthetic intervention on real measured covariates.
- Existing synthetic pipeline remains only as a unit/debug fallback.

## Default real dataset

The UCI dataset contains 19,735 ten-minute observations over about 4.5 months from a low-energy building: appliance energy, lighting, nine indoor temperature/humidity zones and external weather. License: CC BY 4.0; DOI `10.24432/C5VC8G`.

The project never calls the injected failures "real field failures." The **measurements are real**; controlled failures are added to the untouched test period so true onset, severity, healthy target and avoidable energy are known for evaluation.

## 1. Install

```bash
cd CarbonTwin-R-Research
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If Conda base is also active, use `conda deactivate` first so `python`, `pip` and `pytest` resolve to the same `.venv`.

## 2. Download + validate the real data

```bash
python -m carbontwin_r.cli data-download --config configs/real_uci.yaml
python -m carbontwin_r.cli prepare --config configs/real_uci.yaml
```

This creates:

```text
data/
├── raw/uci_appliances/
│   ├── energydata_complete.csv
│   └── source_manifest.json          # source, license, SHA-256, timestamp
└── silver/uci_appliances/
    ├── clean.parquet
    ├── quality_report.json
    └── preparation_manifest.json
```

## 3. Train, tune and select models

Fast development search:

```bash
python -m carbontwin_r.cli train --config configs/real_uci.yaml --budget quick
```

Larger research search:

```bash
python -m carbontwin_r.cli train --config configs/real_uci.yaml --budget research
```

Protocol:

```text
first 60%     train + expanding TimeSeriesSplit tuning
next 15%      model-family selection
next 10%      conformal calibration only
final 15%     one final clean test evaluation
```

Artifacts:

```text
outputs/research/
├── models/best_counterfactual.joblib
├── metrics/model_selection.csv
├── metrics/best_model.json
├── metrics/training_registry.json
├── metrics/drift_report.csv
├── metrics/causal_benchmark.json
├── metrics/pareto_front.csv
└── graphs/
    ├── room_adjacency.npy
    ├── room_partial_correlation.npy
    └── graph_metadata.json
```

## 4. Run a scenario from the CLI

```bash
python -m carbontwin_r.cli simulate \
  --config configs/real_uci.yaml \
  --fault hvac_efficiency_drift \
  --severity 0.18 \
  --start 0.65 \
  --zone 2
```

Available benchmark faults:

- `hvac_efficiency_drift`
- `standby_load`
- `lighting_schedule`
- `sensor_bias` — negative-control data-quality fault with **zero true extra energy**

Changing fault, severity, start time or zone reruns the inference and changes the simulator outputs.

## 5. Launch the interactive simulator

```bash
streamlit run simulator/app.py
```

The app has six labs:

1. **Data Engineering** — download, quality gates and lineage.
2. **Model Lab** — run temporal CV hyperparameter search and inspect selected parameters.
3. **Fault Simulator** — change scenario/severity/start/zone and rerun the digital twin.
4. **Graph Lab** — learned Graphical-Lasso topology and current fault localization.
5. **Statistics** — conformal exceedance, Kalman state, CUSUM, change score and SPRT.
6. **Intervention Lab** — DML benchmark and Pareto decision candidates.

This is no longer a fixed chart viewer. It recomputes a scenario from the real held-out data when you press **RUN DIGITAL TWIN**.

## 6. Optional neural fine-tuning

First install advanced dependencies:

```bash
python -m pip install -e '.[advanced]'
```

Then:

```bash
python -m carbontwin_r.cli fine-tune --config configs/real_uci.yaml
```

The experiment:

```text
earlier clean period → GRU pretraining
later clean adaptation period → lower-LR fine-tuning
untouched test period → before-vs-after MAE
```

CarbonTwin-R does **not** call tree hyperparameter search "fine-tuning." If the neural fine-tuned model fails to improve the untouched evaluation period, `claim_finetuning_helped` is `false` and that result should be reported honestly.

## 7. API

```bash
uvicorn api.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/docs`.

Endpoints include:

- `GET /health`
- `GET /model/registry`
- `POST /simulate`

## 8. Docker

```bash
docker compose up --build
```

- API: `http://localhost:8000/docs`
- Simulator: `http://localhost:8501`

The `data/` and `outputs/` folders are mounted so downloads/models survive container restarts.

## 9. Tests

```bash
python -m pytest -q
```

Tests cover leakage-safe lags, chronological partitions, known fault ground truth, sensor-bias negative control, learned graph properties, mathematical routines, conformal coverage and temporal model-selection protocol.

## Research stance

The goal is not to win by naming many algorithms. Each component has a falsifiable role:

- Does nonlinear ML beat a linear baseline on the untouched period?
- Do tuned hyperparameters remain stable across expanding time folds?
- Is 95% conformal coverage actually close to 95%?
- Does the learned graph help localize injected zone degradation?
- How quickly do sequential detectors identify known change points?
- How close is estimated Carbon Debt to the known benchmark value?
- Does fine-tuning improve a later distribution, or should it be rejected?
- Can DML recover a known causal effect under deliberately confounded assignment?

See `docs/RESEARCH_PROTOCOL.md` and `docs/DATA_CARD.md` before writing CV/project claims.
