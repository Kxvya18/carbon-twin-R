from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
import joblib
import matplotlib.pyplot as plt

from .data.generate import make_synthetic_facility
from .data.split import chronological_split
from .features.temporal import add_time_features, model_feature_columns
from .simulation.faults import inject_fault
from .models.baselines import fit_linear_baseline, regression_metrics
from .models.counterfactual import build_counterfactual_model, add_counterfactual_columns
from .probabilistic.conformal import absolute_residual_quantile, intervals, empirical_coverage
from .probabilistic.kalman import kalman_filter_1d, probability_above_threshold
from .detection.changepoint import cusum_positive, bocpd_lite
from .detection.sprt import sprt
from .carbon.intensity import synthetic_carbon_intensity
from .carbon.debt import add_carbon_debt
from .graph.facility import build_engineering_prior_graph, NODES
from .graph.spectral import graph_laplacian, graph_smoothness_series
from .graph.localization import subsystem_localization
from .causal.synthetic import make_intervention_benchmark
from .causal.dml import double_ml_ate
from .optimization.pareto import demo_interventions, pareto_front

def _ensure_dirs(out_dir: Path):
    for p in ["data","models","metrics","figures"]:
        (out_dir/p).mkdir(parents=True, exist_ok=True)

def _plot_series(df, out_dir):
    t = pd.to_datetime(df["timestamp"])
    plt.figure(figsize=(11,4))
    plt.plot(t, df["total_power"], label="Actual", linewidth=1)
    plt.plot(t, df["counterfactual_power"], label="Counterfactual", linewidth=1)
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"figures"/"actual_vs_counterfactual.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11,4))
    plt.plot(t, df["residual"], label="Residual", linewidth=1)
    if "pred_upper" in df:
        excess = df["total_power"] - df["pred_upper"]
        plt.plot(t, excess, label="Excess above interval", linewidth=1)
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"figures"/"residual_and_interval.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11,4))
    plt.plot(t, df["latent_degradation"], label="Latent degradation", linewidth=1)
    plt.plot(t, df["degradation_probability"]*max(1,df["latent_degradation"].max()), label="Scaled probability", alpha=0.7)
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"figures"/"latent_degradation.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11,4))
    plt.plot(t, df["predicted_carbon_debt_kg"], label="Predicted Carbon Debt")
    plt.plot(t, df["true_carbon_debt_kg"], label="True Carbon Debt")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"figures"/"carbon_debt.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11,4))
    plt.plot(t, df["graph_smoothness"], linewidth=1)
    plt.title("Graph signal smoothness")
    plt.tight_layout()
    plt.savefig(out_dir/"figures"/"graph_smoothness.png", dpi=150)
    plt.close()

def run_all(config_path="configs/demo.yaml", output_dir="outputs"):
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(output_dir)
    _ensure_dirs(out_dir)

    healthy = make_synthetic_facility(
        n_steps=cfg["n_steps"],
        freq_minutes=cfg["freq_minutes"],
        seed=cfg["seed"],
    )
    faulty = inject_fault(
        healthy,
        fault_type=cfg["fault"]["type"],
        start_fraction=cfg["fault"]["start_fraction"],
        severity=cfg["fault"]["severity"],
        ramp_steps=cfg["fault"]["ramp_steps"],
        seed=cfg["seed"],
    )

    healthy.to_csv(out_dir/"data"/"healthy.csv", index=False)
    faulty.to_csv(out_dir/"data"/"faulty.csv", index=False)

    healthy_f = add_time_features(healthy).dropna().reset_index(drop=True)
    faulty_f = add_time_features(faulty).dropna().reset_index(drop=True)

    # Align exact row positions after lag feature dropping.
    n = min(len(healthy_f), len(faulty_f))
    healthy_f = healthy_f.iloc[:n].copy()
    faulty_f = faulty_f.iloc[:n].copy()

    train, cal, test_healthy = chronological_split(
        healthy_f,
        train_fraction=cfg["split"]["train_fraction"],
        calibration_fraction=cfg["split"]["calibration_fraction"],
    )

    # Evaluate faulty period matching the healthy test period.
    test_faulty = faulty_f.iloc[test_healthy.index.min():test_healthy.index.max()+1].copy()
    test_faulty = test_faulty.reset_index(drop=True)
    test_healthy = test_healthy.reset_index(drop=True)

    feat = model_feature_columns()
    Xtr, ytr = train[feat], train["total_power"]
    Xcal, ycal = cal[feat], cal["total_power"]
    Xtest_h, ytest_h = test_healthy[feat], test_healthy["total_power"]
    Xtest_f, ytest_f = test_faulty[feat], test_faulty["total_power"]

    linear = fit_linear_baseline(Xtr.fillna(Xtr.median()), ytr)
    linear_pred = linear.predict(Xtest_h.fillna(Xtr.median()))
    linear_metrics = regression_metrics(ytest_h, linear_pred)

    model = build_counterfactual_model(
        random_state=cfg["model"]["random_state"],
        max_iter=cfg["model"]["max_iter"],
    )
    model.fit(Xtr, ytr)
    joblib.dump(model, out_dir/"models"/"counterfactual.joblib")

    pred_cal = model.predict(Xcal)
    pred_h = model.predict(Xtest_h)
    pred_f = model.predict(Xtest_f)

    ml_metrics = regression_metrics(ytest_h, pred_h)

    q = absolute_residual_quantile(ycal, pred_cal, alpha=cfg["uncertainty"]["alpha"])
    low_f, up_f = intervals(pred_f, q)
    coverage_healthy = empirical_coverage(ytest_h, *intervals(pred_h, q))

    result = add_counterfactual_columns(test_faulty, pred_f, low_f, up_f)

    means, vars_ = kalman_filter_1d(
        result["residual"],
        q=cfg["kalman"]["q"],
        r=cfg["kalman"]["r"],
    )
    result["latent_degradation"] = means
    result["latent_variance"] = vars_
    result["degradation_probability"] = probability_above_threshold(
        means, vars_, cfg["kalman"]["critical_degradation"]
    )

    cusum_score, cusum_alarm = cusum_positive(
        result["residual"],
        drift=cfg["cusum"]["drift"],
        threshold=cfg["cusum"]["threshold"],
    )
    result["cusum_score"] = cusum_score
    result["cusum_alarm"] = cusum_alarm
    result["bocpd_score"] = bocpd_lite(result["residual"])

    sprt_llr, sprt_alarm = sprt(
        result["residual"],
        healthy_mean=cfg["sprt"]["healthy_mean"],
        degraded_mean=cfg["sprt"]["degraded_mean"],
        sigma=cfg["sprt"]["sigma"],
        alpha=cfg["sprt"]["alpha"],
        beta=cfg["sprt"]["beta"],
    )
    result["sprt_llr"] = sprt_llr
    result["sprt_alarm"] = sprt_alarm

    result["carbon_intensity_kg_per_kwh"] = synthetic_carbon_intensity(
        pd.to_datetime(result["timestamp"], utc=True),
        base_kg_per_kwh=cfg["carbon"]["base_kg_per_kwh"],
    )
    result = add_carbon_debt(result, dt_hours=cfg["freq_minutes"]/60)

    graph = build_engineering_prior_graph()
    L = graph_laplacian(graph.adjacency)
    node_matrix = result[NODES].to_numpy()
    result["graph_smoothness"] = graph_smoothness_series(node_matrix, L)

    loc = subsystem_localization(test_healthy, result)
    loc.to_csv(out_dir/"metrics"/"subsystem_localization.csv", index=False)

    # Semi-synthetic causal benchmark.
    causal_df = make_intervention_benchmark(test_healthy, seed=cfg["seed"])
    Xc = causal_df[["outdoor_temperature","humidity","occupancy_proxy"]].to_numpy()
    causal = double_ml_ate(
        Xc,
        causal_df["treatment"].to_numpy(),
        causal_df["causal_outcome"].to_numpy(),
        random_state=cfg["seed"],
    )
    causal["true_ate"] = float(causal_df["true_tau"].mean())
    causal["absolute_ate_error"] = abs(causal["ate"] - causal["true_ate"])

    interventions = demo_interventions()
    frontier = pareto_front(interventions)
    frontier.to_csv(out_dir/"metrics"/"pareto_front.csv", index=False)

    result.to_csv(out_dir/"data"/"analysis.csv", index=False)

    true_debt = float(result["true_carbon_debt_kg"].iloc[-1])
    pred_debt = float(result["predicted_carbon_debt_kg"].iloc[-1])

    summary = {
        "n_train": len(train),
        "n_calibration": len(cal),
        "n_test": len(result),
        "linear_healthy_metrics": linear_metrics,
        "counterfactual_healthy_metrics": ml_metrics,
        "healthy_interval_coverage": coverage_healthy,
        "conformal_q": q,
        "predicted_carbon_debt_kg": pred_debt,
        "true_carbon_debt_kg": true_debt,
        "carbon_debt_absolute_error_kg": abs(pred_debt-true_debt),
        "top_localized_subsystem": str(loc.iloc[0]["node"]),
        "causal_dml": causal,
    }
    (out_dir/"metrics"/"summary.json").write_text(json.dumps(summary, indent=2))

    pd.DataFrame([
        {"model":"linear", **linear_metrics},
        {"model":"counterfactual_ml", **ml_metrics},
    ]).to_csv(out_dir/"metrics"/"model_comparison.csv", index=False)

    _plot_series(result, out_dir)
    return summary
