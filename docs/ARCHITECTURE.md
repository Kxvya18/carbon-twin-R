# CarbonTwin-R v0.2 Architecture

```mermaid
flowchart TD
    U[UCI real measured building data] --> R[Immutable raw + SHA-256 lineage]
    R --> Q[Quality gates]
    Q --> S[Silver clean table]
    S --> F[Leakage-safe feature engineering]
    F --> T[Train 60%]
    F --> V[Selection 15%]
    F --> C[Conformal 10%]
    F --> E[Final test 15%]

    T --> CV[Expanding TimeSeriesSplit hyperparameter search]
    CV --> M{Model families}
    M --> M1[Elastic Net]
    M --> M2[Random Forest]
    M --> M3[Extra Trees]
    M --> M4[Hist Gradient Boosting]
    V --> SEL[Select family on later period]
    M --> SEL
    SEL --> REFIT[Refit on Train + Selection]
    C --> CONF[Conformal residual calibration]
    REFIT --> CONF
    REFIT --> BASE[Clean final-test estimate]

    E --> INJ[Controlled fault simulator]
    INJ --> OBS[Observed faulted series]
    REFIT --> CF[Healthy counterfactual estimate]
    OBS --> RES[Residual]
    CF --> RES
    CONF --> RES
    RES --> K[Kalman latent degradation]
    RES --> D[CUSUM / change score / SPRT]
    RES --> CD[Carbon Debt]

    T --> GL[Graphical Lasso room graph]
    GL --> LOC[Graph-aware zone localization]
    OBS --> LOC

    E --> CAUSAL[Semi-synthetic known-effect intervention]
    CAUSAL --> DML[Double ML recovery test]
    DML --> P[Pareto intervention layer]

    RES --> SIM[Interactive Streamlit Simulator]
    LOC --> SIM
    CD --> SIM
    DML --> SIM
```

## Separation of concerns

### Data truth

The UCI measurements are real. Benchmark faults/interventions are controlled semi-synthetic overlays. The repository keeps these labels separate.

### Model tuning vs fine-tuning

- classical models: hyperparameter tuning with temporal CV;
- neural sequence model: optional pretraining/fine-tuning experiment;
- no method is promoted merely because it is more complex.

### Graph truth

The room graph is learned from training-only conditional dependencies with Graphical Lasso. It is not described as physical BIM topology.

### Carbon truth

The default carbon factor is a configurable scenario factor. It is not described as historical Belgian grid intensity. Users may replace it with a time-indexed emissions series for production accounting.

### Causal truth

Double ML is tested on a known-effect benchmark. Observational feature importance is never presented as causal evidence.
