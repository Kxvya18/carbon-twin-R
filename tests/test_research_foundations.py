import numpy as np
import pandas as pd

from carbontwin_r.data.split import four_way_time_split
from carbontwin_r.features.real_energy import add_real_energy_features, real_model_feature_columns
from carbontwin_r.simulation.real_faults import inject_real_fault
from carbontwin_r.graph.learned_graph import learn_room_graph


def make_uci_like(n=600, seed=1):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2016-01-01", periods=n, freq="10min")
    d = {"date": t, "Appliances": 80 + 15*np.sin(np.arange(n)/20) + rng.normal(0,4,n), "lights": rng.uniform(0,20,n)}
    for i in range(1,10):
        d[f"T{i}"] = 20 + 2*np.sin(np.arange(n)/50 + i/3) + rng.normal(0,.15,n)
        d[f"RH_{i}"] = 40 + 4*np.sin(np.arange(n)/70 + i/4) + rng.normal(0,.5,n)
    d.update({"To": 10 + 5*np.sin(np.arange(n)/100), "Pressure": 760+rng.normal(0,2,n),
              "RH_out": 70+rng.normal(0,3,n), "Windspeed": rng.uniform(0,5,n),
              "Visibility": rng.uniform(20,60,n), "Tdewpoint": 5+rng.normal(0,2,n)})
    return pd.DataFrame(d)


def test_four_way_split_is_chronological_and_disjoint():
    df = make_uci_like(100)
    tr, sel, cal, te = four_way_time_split(df)
    assert tr.date.max() < sel.date.min() < cal.date.min() < te.date.min()
    assert len(tr)+len(sel)+len(cal)+len(te) == len(df)


def test_target_history_disabled_has_no_target_derived_features():
    df = add_real_energy_features(make_uci_like(), use_target_history=False)
    cols = real_model_feature_columns(df, use_target_history=False)
    assert not any(c.startswith("Appliances_") for c in cols)
    assert "Appliances" not in cols


def test_lag_features_are_shifted_not_future():
    base = make_uci_like(200)
    out = add_real_energy_features(base, use_target_history=True, target_lags=[1], rolling_windows=[6])
    assert np.isclose(out.loc[10, "Appliances_lag_1"], base.loc[9, "Appliances"])
    assert not np.isclose(out.loc[10, "Appliances_lag_1"], base.loc[11, "Appliances"])


def test_fault_severity_changes_known_waste():
    base = make_uci_like(800)
    low = inject_real_fault(base, severity=.05, start_fraction=.5, zone=2)
    high = inject_real_fault(base, severity=.25, start_fraction=.5, zone=2)
    assert high.true_wasted_energy_wh.sum() > low.true_wasted_energy_wh.sum() > 0
    assert np.allclose(low.healthy_target, base.Appliances)


def test_sensor_bias_is_negative_control_for_true_energy_waste():
    base = make_uci_like(800)
    f = inject_real_fault(base, fault_type="sensor_bias", severity=.2, start_fraction=.5, zone=4)
    assert f.is_fault.sum() > 0
    assert np.isclose(f.true_wasted_energy_wh.sum(), 0.0)


def test_graph_is_learned_from_training_data():
    df = make_uci_like(700)
    g = learn_room_graph(df)
    assert g.adjacency.shape == (9,9)
    assert np.allclose(g.adjacency, g.adjacency.T)
    assert np.allclose(np.diag(g.adjacency), 0)
    assert g.alpha > 0
