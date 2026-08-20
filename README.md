````markdown 
# CarbonTwin-R — Research Simulator

**Causal temporal/graph machine learning for counterfactual detection, attribution and quantification of avoidable carbon emissions.**

CarbonTwin-R is an **active research project** investigating whether a machine-learning digital twin can distinguish normal energy variability from persistent operational degradation, localize the affected subsystem, and estimate avoidable carbon impact under controlled scenarios.

The system uses a **real measured building-energy dataset by default** and combines data engineering, temporal machine learning, probabilistic uncertainty, sequential change detection, graph-based localization, carbon accounting logic and causal inference into one reproducible research pipeline.

> **Research Status:** Active development and empirical evaluation.
>
> The end-to-end data, modelling, simulation, causal-analysis, API and interactive experimentation pipeline has been implemented. Current work focuses on improving counterfactual prediction under temporal shift, reducing false alarms in degradation detection and calibrating Carbon Debt estimates.

---

## Research Objective

CarbonTwin-R studies the following question:

> **Can we learn the counterfactual energy consumption of a healthy system and use deviations from that counterfactual to detect, localize and quantify avoidable operational carbon emissions?**

The intended research pipeline is:

```text
Real measured energy + environmental data
                ↓
Data validation and temporal feature engineering
                ↓
Healthy-energy counterfactual model
                ↓
Prediction uncertainty / conformal calibration
                ↓
Residual and latent degradation modelling
                ↓
Sequential change detection
                ↓
Graph-based subsystem localization
                ↓
Avoidable energy / Carbon Debt estimation
                ↓
Causal intervention analysis
                ↓
Multi-objective mitigation recommendations
```

## Current Research Status

### Implemented

The current repository includes:

- real-data ingestion and provenance tracking;
- data-quality validation and reproducible preparation;
- chronological train / selection / calibration / test partitioning;
- leakage-aware temporal feature engineering;
- classical ML baselines and nonlinear models;
- time-aware hyperparameter search using TimeSeriesSplit;
- baseline-aware model comparison;
- conformal uncertainty estimation;
- temporal drift analysis;
- neural pretraining and lower-learning-rate fine-tuning;
- controlled degradation injection on real held-out measurements;
- latent degradation modelling;
- sequential change-detection methods;
- learned Graphical-Lasso room topology;
- graph-based fault localization;
- avoidable-energy and Carbon Debt estimation;
- semi-synthetic causal inference benchmark using Double ML;
- multi-objective intervention analysis;
- FastAPI inference endpoints;
- interactive Streamlit research simulator;
- automated tests for core statistical, temporal and simulation logic.

### Preliminary empirical findings

The current results should be interpreted as preliminary rather than final benchmark claims.

So far:

- The project operates on the real UCI Appliances Energy Prediction dataset containing 19,735 measured observations.
- Chronological evaluation revealed substantial temporal distribution shift between earlier and later periods.
- A GRU adaptation experiment improved its own held-out MAE from approximately 41.76 to 39.99, corresponding to a relative improvement of about 4.25%.
- The semi-synthetic causal benchmark recovered the known intervention effect closely:  
  - true ATE ≈ -8.55  
  - estimated ATE ≈ -8.26  
  - absolute estimation error ≈ 0.29
- Graph-based localization correctly identified the injected zone in an initial controlled degradation experiment.
- More complex models have not consistently outperformed the seasonal baseline on all metrics, reinforcing the need for baseline-gated model selection rather than complexity-based model choice.
- Temporal distribution shift has exposed weaknesses in static uncertainty calibration.
- Initial degradation-detection experiments revealed premature false alarms in some scenarios.
- Carbon Debt estimation currently requires further calibration because early false alarms can cause excess-energy accumulation to be overestimated.

These observations are being used to guide the next research iteration rather than being hidden or treated as successful final results.

## Current Research Improvements

The current development cycle focuses on five main areas.

### 1. Stronger counterfactual modelling

The healthy-energy predictor is being refined to better model temporal demand while avoiding leakage from post-fault observations.

Candidate improvements include:

- stronger temporal baselines;
- lagged / autoregressive information with leakage-safe inference;
- recurrent temporal models;
- regime-aware prediction;
- baseline-gated model selection;
- rolling and blocked temporal evaluation.

A more complex model will only be selected if it provides measurable improvement over simpler baselines.

### 2. Shift-aware uncertainty calibration

Preliminary experiments show strong temporal distribution shift, which can degrade static conformal coverage.

Current work explores:

- adaptive calibration;
- weighted conformal prediction;
- rolling calibration windows;
- explicit drift-aware uncertainty evaluation.

The target is not merely to generate intervals, but to verify that empirical coverage remains close to the requested coverage level on later unseen data.

### 3. More reliable degradation detection

Sequential detection is being refined so isolated prediction residuals do not immediately trigger a degradation event.

Current improvements include:

- uncertainty-band exceedance;
- persistent evidence across multiple windows;
- latent degradation probability;
- calibrated sequential thresholds;
- explicit no-fault false-positive measurement;
- correct distinction between early false alarms, missed detections and true detection delay.

### 4. Better Carbon Debt estimation

Carbon Debt should represent statistically supported avoidable energy rather than accumulated model residual noise.

The estimator is therefore being refined around evidence-backed excess energy such as:

```text
avoidable energy
    = max(0, observed energy - upper healthy-energy bound)
```

with optional degradation-probability gating.

This is intended to reduce overestimation caused by uncertain or premature degradation decisions.

### 5. Full scenario-matrix evaluation

The final evaluation will cover multiple:

- fault families;
- fault severities;
- fault start times;
- zones;
- random seeds.

Results will be aggregated across:

- counterfactual MAE / RMSE;
- uncertainty coverage;
- false-positive rate;
- missed-detection rate;
- detection delay;
- localization accuracy;
- Carbon Debt estimation error;
- causal-effect estimation error;
- robustness under distribution shift.

Final research claims will only be written after this evaluation is complete.

## Default Real Dataset

CarbonTwin-R currently uses the UCI Appliances Energy Prediction dataset.

It contains 19,735 ten-minute observations collected over approximately 4.5 months from a low-energy residential building.

Measurements include:

- appliance energy consumption;
- lighting energy;
- temperature and relative humidity from nine indoor zones;
- outside temperature and humidity;
- pressure;
- wind speed;
- visibility;
- dew-point temperature;
- temporal information.

- License: CC BY 4.0
- DOI: 10.24432/C5VC8G

UCI also contains the random variables `rv1` and `rv2`. They are explicitly removed from model development because they are negative-control random features and should not contribute predictive information.

### Real measurements vs controlled degradation

CarbonTwin-R does not claim that injected faults are measured field failures.

The underlying energy and environmental measurements are real.

Controlled degradation scenarios are introduced only into the held-out evaluation period so that the experiment has known:

- fault onset;
- fault severity;
- affected zone;
- healthy counterfactual;
- avoidable energy.

This provides controlled ground truth for evaluating the detection and attribution methods.

## Installation

```bash
cd CarbonTwin-R-Research
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If Conda base is active:

```bash
conda deactivate
```

Then reactivate the project environment:

```bash
source .venv/bin/activate
```

Using `python -m ...` throughout the project helps ensure commands use the same Python environment.

## Download and Validate the Real Data

```bash
python -m carbontwin_r.cli data-download \
  --config configs/real_uci.yaml
```

```bash
python -m carbontwin_r.cli prepare \
  --config configs/real_uci.yaml
```

The data pipeline creates:

```text
data/
├── raw/uci_appliances/
│   ├── energydata_complete.csv
│   └── source_manifest.json
│
└── silver/uci_appliances/
    ├── clean.parquet
    ├── quality_report.json
    └── preparation_manifest.json
```

The manifests record source and transformation provenance, including checksums and preparation metadata.

Raw and generated datasets are excluded from Git.

## Temporal Experimental Protocol

The default research split is chronological:

```text
first 60%   → training + expanding TimeSeriesSplit tuning
next 15%    → model-family selection
next 10%    → conformal calibration only
final 15%   → untouched final evaluation
```

Random shuffling is intentionally avoided because the deployment problem is temporal.

The project treats:

- training,
- model selection,
- uncertainty calibration,
- final evaluation

as separate experimental stages.

## Train, Tune and Select Models

### Quick development search

```bash
python -m carbontwin_r.cli train \
  --config configs/real_uci.yaml \
  --budget quick
```

### Larger research search

```bash
python -m carbontwin_r.cli train \
  --config configs/real_uci.yaml \
  --budget research
```

Candidate model families currently include:

- seasonal statistical baseline;
- Elastic Net;
- Random Forest;
- Extra Trees;
- Histogram Gradient Boosting;
- optional advanced temporal/neural models.

Hyperparameters are evaluated using time-aware validation.

Complexity alone is never treated as evidence of improvement.

## Research Artifacts

Experiments generate artifacts under:

```text
outputs/research/
├── models/
├── metrics/
├── graphs/
└── experiment outputs/
```

Typical generated artifacts include:

```text
models/best_counterfactual.joblib

metrics/
├── model_selection.csv
├── best_model.json
├── training_registry.json
├── drift_report.csv
├── causal_benchmark.json
└── pareto_front.csv

graphs/
├── room_adjacency.npy
├── room_partial_correlation.npy
└── graph_metadata.json
```

Generated model/data/output artifacts are excluded from Git by default.

## Run a Controlled Scenario

Example:

```bash
python -m carbontwin_r.cli simulate \
  --config configs/real_uci.yaml \
  --fault hvac_efficiency_drift \
  --severity 0.18 \
  --start 0.65 \
  --zone 2
```

Available benchmark scenarios include:

- `hvac_efficiency_drift`
- `standby_load`
- `lighting_schedule`
- `sensor_bias`

`sensor_bias` is deliberately treated as a data-quality negative control with zero true additional energy.

Changing the fault family, severity, start time or zone creates a different controlled experiment on the real held-out time series.

## Interactive Research Simulator

Launch:

```bash
python -m streamlit run simulator/app.py
```

The interface is designed as an interactive experimentation environment rather than a static results dashboard.

It contains six research labs.

### Data Engineering

Inspect:

- data acquisition;
- quality gates;
- preprocessing;
- lineage;
- provenance.

### Model Lab

Run and inspect:

- temporal cross-validation;
- hyperparameter search;
- model-family comparison;
- selected parameters.

### Fault Simulator

Modify:

- fault family;
- severity;
- onset;
- affected zone.

Then rerun the digital twin.

### Graph Lab

Inspect:

- learned room topology;
- partial correlations;
- subsystem localization.

### Statistics Lab

Inspect:

- prediction intervals;
- conformal exceedance;
- latent degradation state;
- CUSUM;
- change score;
- SPRT-style evidence.

### Intervention Lab

Inspect:

- causal benchmark estimates;
- possible interventions;
- multi-objective decision candidates.

## Neural Pretraining and Fine-Tuning

Install the optional research dependencies:

```bash
python -m pip install -e '.[advanced]'
```

Then run:

```bash
python -m carbontwin_r.cli fine-tune \
  --config configs/real_uci.yaml
```

The temporal adaptation experiment follows:

```text
earlier clean period
        ↓
neural pretraining
        ↓
later clean adaptation period
        ↓
lower-learning-rate fine-tuning
        ↓
untouched evaluation period
        ↓
before-vs-after comparison
```

Fine-tuning is only considered useful when it improves the untouched evaluation period.

CarbonTwin-R deliberately distinguishes:

- hyperparameter tuning;
- model selection;
- neural fine-tuning.

They are not treated as interchangeable concepts.

## Causal ML Benchmark

CarbonTwin-R includes a semi-synthetic causal experiment constructed on real measured covariates.

Because the intervention effect is known during construction, the experiment can test whether causal ML recovers the known effect under deliberately confounded treatment assignment.

The current implementation uses Double Machine Learning and reports:

- estimated ATE;
- standard error;
- confidence interval;
- known true ATE;
- absolute estimation error.

This benchmark evaluates causal-estimation methodology; it is not presented as observational proof of a real-world building intervention.

## API

Launch:

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive API documentation: <http://127.0.0.1:8000/docs>

Endpoints include:

- `GET /health`
- `GET /model/registry`
- `POST /simulate`

## Docker

```bash
docker compose up --build
```

Services:

- API: <http://localhost:8000/docs>
- Simulator: <http://localhost:8501>

Local data and experiment directories can be mounted so downloaded datasets and generated artifacts survive container restarts.

## Tests

Run:

```bash
python -m pytest -q
```

The test suite covers core areas including:

- chronological splitting;
- leakage-aware temporal processing;
- fault ground truth;
- sensor-bias negative controls;
- graph properties;
- mathematical routines;
- causal routines;
- conformal logic;
- model-selection protocol.

## Research Principles

CarbonTwin-R is intentionally built around questions that can fail.

The research asks:

- Can an ML counterfactual actually outperform a simple temporal baseline?
- Are improvements stable across later time periods?
- Does neural adaptation help under measurable distribution shift?
- Does nominal 95% uncertainty actually produce approximately 95% empirical coverage?
- Can degradation be detected without generating unacceptable false alarms?
- Does the learned graph provide useful subsystem localization?
- How accurately can avoidable energy and Carbon Debt be recovered?
- Can causal ML recover a known intervention effect under controlled confounding?
- Which components remain useful when individual modelling assumptions fail?

A negative result is treated as experimental evidence rather than hidden by selecting a more favourable metric.

## Current Limitations

CarbonTwin-R is currently a research system, not a production carbon-accounting platform.

Important limitations include:

- The current default dataset represents a residential/low-energy building rather than an industrial facility.
- Controlled degradation scenarios are semi-synthetic and should not be interpreted as observed equipment failures.
- Temporal distribution shift affects both prediction and uncertainty calibration.
- Current degradation detection is undergoing false-alarm reduction.
- Carbon Debt estimation is undergoing calibration against known injected ground truth.
- The full multi-seed, multi-fault experiment matrix is still being completed.
- Final performance comparisons and research conclusions have therefore not yet been frozen.

These limitations are explicitly documented so implementation progress is not confused with empirical validation.
````
