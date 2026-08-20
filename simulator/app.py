from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from carbontwin_r.data.prepare import prepare_real_uci
from carbontwin_r.research_pipeline import train_real_models, simulate_real_scenario, run_temporal_finetune_experiment

CONFIG = "configs/real_uci.yaml"
cfg = yaml.safe_load(Path(CONFIG).read_text())
root = Path(cfg["project"]["artifact_dir"])

st.set_page_config(page_title="CarbonTwin-R Simulator", layout="wide")
st.title("CarbonTwin-R Research Simulator")
st.caption("Real measured building data → tuned counterfactual ML → controlled faults → uncertainty → statistical detection → learned room graph → Carbon Debt")

with st.sidebar:
    st.header("Scenario controls")
    fault = st.selectbox("Fault scenario", ["hvac_efficiency_drift", "standby_load", "lighting_schedule", "sensor_bias"])
    severity = st.slider("Fault severity", 0.01, 0.40, float(cfg["simulation"]["severity"]), 0.01)
    start = st.slider("Fault start (% through clean test period)", 0.35, 0.85, float(cfg["simulation"]["start_fraction"]), 0.01)
    zone = st.selectbox("Thermal zone", list(range(1, 10)), index=1)
    seed = st.number_input("Simulation seed", value=int(cfg["seed"]), step=1)
    st.divider()
    search_budget = st.radio("Hyperparameter-search budget", ["quick", "research"], index=0,
                             help="Quick is for demos. Research evaluates more sampled hyperparameter settings using TimeSeriesSplit.")

raw_path = Path(cfg["project"]["raw_dir"]) / "energydata_complete.csv"
registry_path = root / "metrics" / "training_registry.json"

setup_tab, model_tab, sim_tab, graph_tab, stats_tab, intervention_tab = st.tabs([
    "1 · Data Engineering", "2 · Model Lab", "3 · Fault Simulator", "4 · Graph Lab", "5 · Statistics", "6 · Intervention Lab"
])

with setup_tab:
    st.subheader("Real data source")
    st.markdown("**UCI Appliances Energy Prediction** — real 10-minute measurements from a low-energy house, including appliance energy, lighting, room temperature/humidity and weather.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw data", "Ready" if raw_path.exists() else "Not downloaded")
    silver = Path(cfg["project"]["silver_dir"]) / "clean.parquet"
    silver_ready = silver.exists() or silver.with_suffix(".csv.gz").exists()
    c2.metric("Silver layer", "Ready" if silver_ready else "Not prepared")
    c3.metric("Training registry", "Ready" if registry_path.exists() else "Not trained")
    if st.button("Download + validate + prepare real data", type="primary"):
        with st.spinner("Downloading from UCI and building immutable raw + validated silver layers..."):
            result = prepare_real_uci(CONFIG, download=True)
        st.success("Real data prepared.")
        st.json(result)
    qpath = Path(cfg["project"]["silver_dir"]) / "quality_report.json"
    if qpath.exists():
        st.subheader("Quality gates")
        st.json(json.loads(qpath.read_text()))
    mpath = Path(cfg["project"]["raw_dir"]) / "source_manifest.json"
    if mpath.exists():
        st.subheader("Lineage / reproducibility")
        st.json(json.loads(mpath.read_text()))

with model_tab:
    st.subheader("Model selection and hyperparameter tuning")
    st.write("Hyperparameters are tuned only on expanding time-series folds. A later selection period chooses the model family; conformal calibration and final test remain untouched during tuning.")
    if st.button("Run model search / retrain", type="primary"):
        with st.spinner("Running temporal CV hyperparameter search. This can take several minutes..."):
            reg = train_real_models(CONFIG, budget=search_budget)
        st.success(f"Selected {reg['best_model']}")
        st.json(reg)
    selection_file = root / "metrics" / "model_selection.csv"
    if selection_file.exists():
        tab = pd.read_csv(selection_file)
        st.dataframe(tab, use_container_width=True)
    if registry_path.exists():
        reg = json.loads(registry_path.read_text())
        a,b,c,d = st.columns(4)
        a.metric("Selected model", reg["best_model"])
        b.metric("Final test MAE", f"{reg['test_metrics']['mae']:.2f} Wh")
        c.metric("Final test R²", f"{reg['test_metrics']['r2']:.3f}")
        d.metric("Conformal coverage", f"{100*reg['test_empirical_coverage']:.1f}%")
        st.caption(f"Feature count: {reg['feature_count']} · target history enabled: {reg['use_target_history']}")
    st.subheader("Optional neural fine-tuning experiment")
    st.write("Fine-tuning is not forced onto tree models. If PyTorch is installed, a GRU is pretrained on an earlier clean period and fine-tuned at lower learning rate on a later adaptation period, then evaluated on an untouched test period.")
    if st.button("Run temporal pretrain → fine-tune experiment"):
        try:
            with st.spinner("Training temporal model..."):
                ft = run_temporal_finetune_experiment(CONFIG)
            st.json(ft)
            if not ft["claim_finetuning_helped"]:
                st.warning("Fine-tuning did not improve the untouched evaluation period. Do not claim it as a benefit.")
        except Exception as e:
            st.error(str(e))

with sim_tab:
    st.subheader("Interactive counterfactual fault laboratory")
    st.write("The base series is real measured UCI data. Controls below create a controlled semi-synthetic failure so the true healthy counterfactual and true avoidable energy remain known.")
    if st.button("RUN DIGITAL TWIN", type="primary", use_container_width=True):
        if not registry_path.exists():
            st.error("Train the model first in Model Lab.")
        else:
            with st.spinner("Injecting scenario and running counterfactual/statistical inference..."):
                summary, scenario, loc = simulate_real_scenario(CONFIG, fault_type=fault, severity=severity,
                                                                 start_fraction=start, zone=zone, seed=int(seed), output_name="simulator_latest")
            st.session_state["scenario_summary"] = summary
            st.session_state["scenario_df"] = scenario
            st.session_state["localization"] = loc
    if "scenario_df" in st.session_state:
        summary = st.session_state["scenario_summary"]; scenario = st.session_state["scenario_df"]
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("True Carbon Debt", f"{summary['true_carbon_debt_kg']:.3f} kg")
        c2.metric("Estimated Carbon Debt", f"{summary['predicted_carbon_debt_kg']:.3f} kg")
        c3.metric("Max P(degradation)", f"{100*summary['max_degradation_probability']:.1f}%")
        c4.metric("Detection delay", "—" if summary["detection_delay_minutes"] is None else f"{summary['detection_delay_minutes']} min")
        c5.metric("Top graph zone", summary["top_localized_zone"])
        plot = scenario.set_index("date")[["observed_target", "counterfactual_power", "healthy_target"]]
        st.line_chart(plot)
        st.caption("Observed = fault-injected real series; healthy_target = original real measurement; counterfactual = model estimate.")
        st.subheader("Cumulative Carbon Debt")
        st.line_chart(scenario.set_index("date")[["predicted_carbon_debt_kg", "true_carbon_debt_kg"]])

with graph_tab:
    st.subheader("Statistically learned room graph")
    st.write("Topology is learned from clean training data using Graphical Lasso conditional dependencies across the nine room temperature/humidity pairs. Test/fault data are not used to learn topology.")
    adj_path = root/"graphs"/"room_adjacency.npy"
    if adj_path.exists():
        A = np.load(adj_path)
        st.dataframe(pd.DataFrame(A, index=[f"zone_{i}" for i in range(1,10)], columns=[f"zone_{i}" for i in range(1,10)]), use_container_width=True)
        gmeta = root/"graphs"/"graph_metadata.json"
        if gmeta.exists(): st.json(json.loads(gmeta.read_text()))
    if "localization" in st.session_state:
        st.subheader("Current scenario localization")
        st.dataframe(st.session_state["localization"], use_container_width=True)
        st.bar_chart(st.session_state["localization"].set_index("zone")["attribution_fraction"])
        st.caption(f"Ground-truth injected node: {st.session_state['scenario_summary']['true_fault_node']}")

with stats_tab:
    st.subheader("Uncertainty, drift and sequential statistics")
    drift_path = root/"metrics"/"drift_report.csv"
    if drift_path.exists():
        st.write("Distribution shift between early training and later clean test data:")
        st.dataframe(pd.read_csv(drift_path), use_container_width=True)
    if "scenario_df" in st.session_state:
        scenario = st.session_state["scenario_df"]
        st.write("Counterfactual residual and calibrated exceedance")
        st.line_chart(scenario.set_index("date")[["residual", "interval_exceedance"]])
        st.write("Latent degradation probability")
        st.line_chart(scenario.set_index("date")[["degradation_probability", "bocpd_score"]])
        st.write("Sequential tests")
        st.line_chart(scenario.set_index("date")[["cusum_score", "sprt_llr"]])

with intervention_tab:
    st.subheader("Causal and decision layer")
    causal_path = root/"metrics"/"causal_benchmark.json"
    if causal_path.exists():
        st.write("Double ML is validated on a **semi-synthetic known-effect intervention built on real measured covariates**. This tests estimator recovery without pretending an observational association is causal proof.")
        st.json(json.loads(causal_path.read_text()))
    frontier = root/"metrics"/"pareto_front.csv"
    if frontier.exists():
        st.subheader("Pareto intervention candidates")
        st.dataframe(pd.read_csv(frontier), use_container_width=True)
    st.info("For a real facility pilot, causal recommendations require actual intervention or quasi-experimental data. The simulator keeps this boundary explicit.")
