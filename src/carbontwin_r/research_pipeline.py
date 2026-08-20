from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import yaml

from .data.prepare import load_silver, prepare_real_uci
from .data.storage import write_table, read_table
from .data.split import four_way_time_split
from .features.real_energy import add_real_energy_features, real_model_feature_columns
from .models.model_selection import tune_and_select, final_refit, metrics
from .models.real_baselines import SeasonalMedianBaseline
from .probabilistic.conformal import absolute_residual_quantile, intervals, empirical_coverage
from .probabilistic.kalman import kalman_filter_1d, probability_above_threshold
from .detection.changepoint import cusum_positive, bocpd_lite, first_detection_index
from .detection.sprt import sprt
from .simulation.real_faults import inject_real_fault
from .graph.learned_graph import learn_room_graph, zone_state_matrix, graph_energy_series, localize_zones
from .carbon.debt import add_wh_carbon_debt
from .evaluation.drift import drift_report
from .evaluation.bootstrap import block_bootstrap_difference
from .causal.real_benchmark import evaluate_dml_on_real_context
from .optimization.pareto import demo_interventions, pareto_front


def _load_cfg(path):
    return yaml.safe_load(Path(path).read_text())


def _dirs(cfg):
    root = Path(cfg["project"]["artifact_dir"])
    for name in ["data", "models", "metrics", "graphs", "simulations"]:
        (root/name).mkdir(parents=True, exist_ok=True)
    return root


def train_real_models(config_path="configs/real_uci.yaml", budget=None, force_prepare=False):
    cfg = _load_cfg(config_path)
    root = _dirs(cfg)
    if force_prepare:
        prepare_real_uci(config_path, download=True)
    df = load_silver(config_path)
    fcfg = cfg["features"]
    features = add_real_energy_features(
        df, target=fcfg["target"], target_lags=fcfg["target_lags"],
        rolling_windows=fcfg["rolling_windows"], use_target_history=fcfg["use_target_history"]
    ).dropna().reset_index(drop=True)

    s = cfg["split"]
    train, selection, conformal, test = four_way_time_split(features, **s)
    # Gold layer: deterministic feature matrix plus explicit temporal split labels.
    gold = features.copy()
    gold["split"] = "test"
    gold.loc[train.index, "split"] = "train"
    gold.loc[selection.index, "split"] = "selection"
    gold.loc[conformal.index, "split"] = "conformal"
    gold_dir = Path(cfg["project"]["gold_dir"]); gold_dir.mkdir(parents=True, exist_ok=True)
    gold_path = write_table(gold, gold_dir/"model_matrix.parquet")
    cols = real_model_feature_columns(features, target=fcfg["target"], use_target_history=fcfg["use_target_history"])
    target = fcfg["target"]

    mcfg = cfg["model_selection"]
    budget = budget or mcfg["budget"]
    best, results, selected_model = tune_and_select(
        train[cols], train[target], selection[cols], selection[target],
        candidates=mcfg["candidates"], seed=cfg["seed"], cv_splits=mcfg["cv_splits"], budget=budget,
        n_iter_quick=mcfg["n_iter_quick"], n_iter_research=mcfg["n_iter_research"],
        artifact_dir=root/"metrics"
    )

    train_selection = pd.concat([train, selection], ignore_index=True)
    model = final_refit(selected_model, train_selection[cols], train_selection[target], root/"models"/"best_counterfactual.joblib")
    pred_conf = model.predict(conformal[cols])
    cal_residual = conformal[target].to_numpy() - pred_conf
    q = absolute_residual_quantile(conformal[target], pred_conf, alpha=cfg["uncertainty"]["alpha"])
    cal_residual_std = float(np.std(cal_residual, ddof=1))
    pred_test = model.predict(test[cols])
    test_metrics = metrics(test[target], pred_test)
    low, high = intervals(pred_test, q)
    coverage = empirical_coverage(test[target], low, high)

    # Drift diagnostic between early training and late clean test. Fine-tuning is a conditional response, not a ritual.
    drift_cols = [c for c in ["To", "RH_out", "T1", "T2", "T4", "T8", "lights"] if c in train]
    drift = drift_report(train, test, drift_cols)
    drift.to_csv(root/"metrics"/"drift_report.csv", index=False)

    graph = learn_room_graph(train)
    np.save(root/"graphs"/"room_adjacency.npy", graph.adjacency)
    np.save(root/"graphs"/"room_partial_correlation.npy", graph.partial_correlation)
    (root/"graphs"/"graph_metadata.json").write_text(json.dumps({"graphical_lasso_alpha": graph.alpha, "rooms": graph.room_names, "converged": graph.converged, "convergence_warnings": graph.convergence_warnings}, indent=2))

    baseline = SeasonalMedianBaseline().fit(train_selection, target=target)
    baseline_test_metrics = metrics(test[target], baseline.predict(test))

    causal_eval = evaluate_dml_on_real_context(test, seed=cfg["seed"])
    (root/"metrics"/"causal_benchmark.json").write_text(json.dumps(causal_eval, indent=2))
    frontier = pareto_front(demo_interventions())
    frontier.to_csv(root/"metrics"/"pareto_front.csv", index=False)

    registry = {
        "dataset": "UCI Appliances Energy Prediction (real measured data)",
        "target": target,
        "feature_count": len(cols),
        "features": cols,
        "best_model": best.model,
        "best_params": best.best_params,
        "search_budget": budget,
        "test_metrics": test_metrics,
        "seasonal_median_baseline_test_metrics": baseline_test_metrics,
        "conformal_alpha": cfg["uncertainty"]["alpha"],
        "conformal_q_wh": q,
        "conformal_residual_std_wh": cal_residual_std,
        "test_empirical_coverage": coverage,
        "split_rows": {"train":len(train), "selection":len(selection), "conformal":len(conformal), "test":len(test)},
        "gold_model_matrix": str(gold_path),
        "test_start": str(test["date"].min()), "test_end": str(test["date"].max()),
        "use_target_history": fcfg["use_target_history"],
        "fine_tuning_policy": "Only neural temporal models are fine-tuned, and only if drift/adaptation experiments justify it.",
        "causal_benchmark": causal_eval,
    }
    (root/"metrics"/"training_registry.json").write_text(json.dumps(registry, indent=2, default=str))
    # Store clean test and split positions for reproducible simulator use.
    write_table(test, root/"data"/"clean_test.parquet")
    write_table(train_selection, root/"data"/"healthy_reference.parquet")
    return registry


def simulate_real_scenario(config_path="configs/real_uci.yaml", *, fault_type=None, severity=None,
                           start_fraction=None, zone=None, seed=None, ensure_trained=True, output_name="latest"):
    cfg = _load_cfg(config_path)
    root = _dirs(cfg)
    registry_path = root/"metrics"/"training_registry.json"
    model_path = root/"models"/"best_counterfactual.joblib"
    if ensure_trained and (not registry_path.exists() or not model_path.exists()):
        train_real_models(config_path)
    registry = json.loads(registry_path.read_text())
    model = joblib.load(model_path)

    clean_test = read_table(root/"data"/"clean_test.parquet", parse_dates=["date"])
    healthy_ref = read_table(root/"data"/"healthy_reference.parquet", parse_dates=["date"])
    simcfg = cfg["simulation"]
    fault_type = fault_type or simcfg["fault_type"]
    severity = float(simcfg["severity"] if severity is None else severity)
    start_fraction = float(simcfg["start_fraction"] if start_fraction is None else start_fraction)
    zone = int(simcfg["zone"] if zone is None else zone)
    seed = int(cfg["seed"] if seed is None else seed)

    # Fault is injected into real measured test data; original measured target remains ground-truth healthy counterfactual.
    raw_cols = [c for c in clean_test.columns if not c.startswith("Appliances_") and c not in {"tod_sin","tod_cos","dow_sin","dow_cos","doy_sin","doy_cos","is_weekend","indoor_temp_mean","indoor_temp_std","indoor_rh_mean","thermal_delta_outdoor"}]
    base_raw = clean_test[raw_cols].copy()
    sim_raw = inject_real_fault(base_raw, fault_type=fault_type, severity=severity,
                                start_fraction=start_fraction, ramp_steps=simcfg["ramp_steps"], zone=zone, seed=seed)
    fcfg = cfg["features"]
    sim = add_real_energy_features(sim_raw, target=fcfg["target"], target_lags=fcfg["target_lags"],
                                   rolling_windows=fcfg["rolling_windows"], use_target_history=fcfg["use_target_history"])
    sim = sim.dropna().reset_index(drop=True)
    feature_cols = registry["features"]
    pred = model.predict(sim[feature_cols])
    q = float(registry["conformal_q_wh"])
    lo, hi = intervals(pred, q)
    sim["counterfactual_power"] = pred
    sim["residual"] = sim["observed_target"] - sim["counterfactual_power"]
    sim["pred_lower"] = lo; sim["pred_upper"] = hi
    sim["interval_exceedance"] = np.maximum(0, sim["observed_target"] - hi)

    kcfg = cfg["kalman"]
    cal_std = max(float(registry.get("conformal_residual_std_wh", 10.0)), 1.0)
    observation_var = cal_std**2
    process_var = max(1e-4, (0.05*cal_std)**2)
    critical = max(5.0, 0.5*q)
    means, vars_ = kalman_filter_1d(sim["residual"], q=process_var, r=observation_var)
    sim["latent_degradation"] = means; sim["latent_variance"] = vars_
    sim["degradation_probability"] = probability_above_threshold(means, vars_, critical)
    sim["degradation_threshold_wh"] = critical

    ccfg = cfg["cusum"]
    cscore, calarm = cusum_positive(sim["residual"], drift=ccfg["drift"], threshold=ccfg["threshold"])
    sim["cusum_score"] = cscore; sim["cusum_alarm"] = calarm
    sim["bocpd_score"] = bocpd_lite(sim["residual"].to_numpy(), sigma=max(float(np.std(sim["residual"][:max(30, len(sim)//4)])), 1.0))
    scfg = cfg["sprt"]
    llr, salarm = sprt(sim["residual"], healthy_mean=scfg["healthy_mean"], degraded_mean=scfg["degraded_mean"],
                       sigma=scfg["sigma"], alpha=scfg["alpha"], beta=scfg["beta"])
    sim["sprt_llr"] = llr; sim["sprt_alarm"] = salarm

    # Learned statistical graph from training-only data.
    A = np.load(root/"graphs"/"room_adjacency.npy")
    states = zone_state_matrix(sim, healthy_ref)
    sim["graph_energy"] = graph_energy_series(states, A)
    loc = localize_zones(sim, healthy_ref, A)

    sim = add_wh_carbon_debt(sim, kg_per_kwh=cfg["carbon"]["kg_per_kwh"])
    out_dir = root/"simulations"/output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(sim, out_dir/"scenario.parquet")
    loc.to_csv(out_dir/"localization.csv", index=False)

    fault_idx = np.flatnonzero(sim["is_fault"].to_numpy() > 0)
    true_start = int(fault_idx[0]) if len(fault_idx) else None
    detected = np.flatnonzero((sim["cusum_alarm"].to_numpy() > 0) | (sim["degradation_probability"].to_numpy() > 0.95))
    detected_start = int(detected[0]) if len(detected) else None
    delay_steps = None if true_start is None or detected_start is None else max(0, detected_start-true_start)

    true_debt = float(sim["true_carbon_debt_kg"].iloc[-1])
    pred_debt = float(sim["predicted_carbon_debt_kg"].iloc[-1])
    expected_node = str(sim.loc[sim["is_fault"].eq(1), "fault_node"].iloc[0]) if sim["is_fault"].any() else "none"
    summary = {
        "data_basis": "real UCI measured building data + controlled semi-synthetic fault",
        "fault_type": fault_type, "severity": severity, "zone": zone, "start_fraction": start_fraction,
        "model": registry["best_model"], "model_test_mae_wh": registry["test_metrics"]["mae"],
        "true_fault_node": expected_node,
        "top_localized_zone": str(loc.iloc[0]["zone"]),
        "true_carbon_debt_kg": true_debt,
        "predicted_carbon_debt_kg": pred_debt,
        "carbon_debt_absolute_error_kg": abs(pred_debt-true_debt),
        "true_fault_start_row": true_start,
        "detected_start_row": detected_start,
        "detection_delay_minutes": None if delay_steps is None else int(delay_steps*10),
        "max_degradation_probability": float(sim["degradation_probability"].max()),
        "carbon_factor_note": cfg["carbon"]["note"],
    }
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary, sim, loc


def run_research(config_path="configs/real_uci.yaml", budget=None):
    registry = train_real_models(config_path, budget=budget)
    summary, _, _ = simulate_real_scenario(config_path)
    return {"training": registry, "simulation": summary}


def run_temporal_finetune_experiment(config_path="configs/real_uci.yaml"):
    """Optional neural pretrain/fine-tune experiment with an untouched evaluation period.

    This is deliberately separate from tree-model tuning. It reports whether adaptation helped; if not, the
    repository should not claim fine-tuning as a benefit.
    """
    from sklearn.metrics import mean_absolute_error
    from .fine_tuning.temporal_adaptation import pretrain_then_finetune
    cfg = _load_cfg(config_path); root = _dirs(cfg)
    registry_path = root/"metrics"/"training_registry.json"
    if not registry_path.exists():
        train_real_models(config_path)
    registry = json.loads(registry_path.read_text())
    ref = read_table(root/"data"/"healthy_reference.parquet", parse_dates=["date"])
    test = read_table(root/"data"/"clean_test.parquet", parse_dates=["date"])
    cols = registry["features"]
    cut = int(len(ref)*0.82)
    pre, adapt = ref.iloc[:cut], ref.iloc[cut:]
    med = pre[cols].median(numeric_only=True)
    Xpre = pre[cols].fillna(med).to_numpy(dtype=float); ypre = pre["Appliances"].to_numpy(dtype=float)
    Xadapt = adapt[cols].fillna(med).to_numpy(dtype=float); yadapt = adapt["Appliances"].to_numpy(dtype=float)
    Xtest = test[cols].fillna(med).to_numpy(dtype=float); ytest = test["Appliances"].to_numpy(dtype=float)
    fcfg = cfg["fine_tuning"]
    drift_path = root/"metrics"/"drift_report.csv"
    drift_max_ks = None
    fine_tuning_recommended = True
    if drift_path.exists():
        dr = pd.read_csv(drift_path)
        if len(dr):
            drift_max_ks = float(dr["ks_stat"].max())
            fine_tuning_recommended = drift_max_ks >= float(fcfg.get("drift_ks_threshold", 0.20))

    out = pretrain_then_finetune(
        Xpre, ypre, Xadapt, yadapt, Xtest,
        sequence_length=fcfg["sequence_length"], pretrain_epochs=fcfg["pretrain_epochs"],
        finetune_epochs=fcfg["finetune_epochs"], learning_rate=fcfg["learning_rate"],
        finetune_learning_rate=fcfg["finetune_learning_rate"], seed=cfg["seed"]
    )
    offset = out["sequence_offset"]
    truth = ytest[offset:]
    report = {
        "drift_max_ks": drift_max_ks,
        "fine_tuning_recommended_by_drift": bool(fine_tuning_recommended),
        "pretrained_mae": float(mean_absolute_error(truth, out["pred_before"])),
        "finetuned_mae": float(mean_absolute_error(truth, out["pred_after"])),
    }
    report["relative_improvement"] = (report["pretrained_mae"]-report["finetuned_mae"])/max(report["pretrained_mae"],1e-9)
    report["claim_finetuning_helped"] = bool(report["finetuned_mae"] < report["pretrained_mae"])
    (root/"metrics"/"fine_tuning_report.json").write_text(json.dumps(report, indent=2))
    return report
