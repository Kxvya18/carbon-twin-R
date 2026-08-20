# Evaluation Report Template

## 1. Dataset
- Facility/building type:
- Sampling interval:
- Number of facilities:
- Train/calibration/test periods:
- Observed vs derived vs simulated columns:

## 2. Counterfactual models
Report MAE, RMSE, R² on healthy held-out data.

## 3. Uncertainty
Report:
- target coverage,
- empirical coverage,
- mean interval width.

## 4. Fault detection
For each fault type and severity:
- precision,
- recall,
- F1,
- false alarms,
- detection delay.

## 5. Carbon Debt
Report:
- true benchmark Carbon Debt,
- estimated Carbon Debt,
- absolute error,
- relative error.

## 6. Localization
Report:
- Top-1 accuracy,
- Top-3 accuracy,
- mean reciprocal rank.

## 7. Causal benchmark
Report:
- true ATE,
- estimated ATE,
- absolute error,
- interval coverage where available.

## 8. Ablations
Compare:
- no temporal history,
- no weather,
- no graph,
- random graph,
- full graph,
- no Bayesian smoothing,
- no conformal calibration.

## 9. Statistical uncertainty
Use block bootstrap over time-correlated errors and report 95% intervals.

## 10. Limitations
Never claim simulated faults are measured field failures.
