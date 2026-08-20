from carbontwin_r.data.generate import make_synthetic_facility
from carbontwin_r.causal.synthetic import make_intervention_benchmark
from carbontwin_r.causal.dml import double_ml_ate

def test_dml_recovers_direction_of_effect():
    df = make_synthetic_facility(n_steps=1200)
    d = make_intervention_benchmark(df)
    X = d[["outdoor_temperature","humidity","occupancy_proxy"]].to_numpy()
    out = double_ml_ate(X, d["treatment"], d["causal_outcome"], random_state=42)
    assert out["ate"] < 0
