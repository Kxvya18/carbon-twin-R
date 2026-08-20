# CarbonTwin-R v0.2 — Research Implementation Map

The old 39-phase list is retained conceptually, but v0.2 reorganizes it around a research-valid workflow.

| Research layer | What is implemented | Main files |
|---|---|---|
| Real data acquisition | Official UCI dataset downloader | `data/uci_appliances.py` |
| Raw lineage | SHA-256, source/license/download manifest | `data/lineage.py` |
| Data quality | schema, timestamp, duplicate, missing/negative checks | `data/quality.py` |
| Silver layer | validated clean Parquet/CSV-gzip fallback | `data/prepare.py`, `data/storage.py` |
| Leakage-safe features | cyclic time, thermal aggregates, optional shifted target history | `features/real_energy.py` |
| Temporal splitting | train/selection/conformal/test | `data/split.py` |
| Gold layer | model matrix + explicit split label | `research_pipeline.py` |
| Statistical baseline | time-of-week seasonal median | `models/real_baselines.py` |
| Candidate ML families | ElasticNet, RF, ExtraTrees, HistGBR | `models/model_selection.py` |
| Hyperparameter tuning | sampled search + expanding `TimeSeriesSplit` | `models/model_selection.py` |
| Final model registry | parameters, features, test metrics, calibration | `outputs/research/metrics/training_registry.json` |
| Fine-tuning | GRU pretrain → low-LR adaptation → untouched evaluation | `fine_tuning/temporal_adaptation.py` |
| Drift gate | KS + PSI report | `evaluation/drift.py` |
| Real-data fault benchmark | controlled faults on held-out measured series | `simulation/real_faults.py` |
| Counterfactual prediction | selected model predicts healthy target | `research_pipeline.py` |
| Conformal uncertainty | dedicated calibration block | `probabilistic/conformal.py` |
| Latent state | Kalman filtering | `probabilistic/kalman.py` |
| Sequential detection | CUSUM, Bayesian-style change score, SPRT | `detection/` |
| Learned graph | Graphical Lasso on training-only room T/RH | `graph/learned_graph.py` |
| Localization | healthy-reference residual + graph weighting | `graph/learned_graph.py` |
| Carbon Debt | true vs estimated avoidable kWh/CO2 | `carbon/debt.py` |
| Causal ML | known-effect semi-synthetic treatment on real covariates + DML | `causal/real_benchmark.py`, `causal/dml.py` |
| Decision optimization | Pareto frontier | `optimization/pareto.py` |
| Stress/ablation/bootstrap | research utilities retained | `evaluation/` |
| Scenario matrix | multi-seed/fault/severity/start/zone evaluation | `scripts/run_matrix.py` |
| Simulator | six interactive research labs | `simulator/app.py` |
| REST surface | registry + simulation API | `api/main.py` |
| Deployment | Docker Compose API + simulator | `Dockerfile`, `docker-compose.yml` |
| Tests | mathematical, leakage, split, fault, graph, tuning protocol | `tests/` |

## What is comparison vs what is connected?

**Connected inference path:** real data → selected counterfactual model → residual → conformal uncertainty → latent/degradation statistics → graph localization → Carbon Debt → intervention analysis.

**Comparisons:** candidate model families; change detectors; advanced neural fine-tuning; ablations; graph-vs-no-graph experiments.

**Evaluation only:** bootstrap confidence intervals, scenario matrix, stress tests, negative controls.
