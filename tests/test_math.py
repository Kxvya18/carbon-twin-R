import numpy as np
import pandas as pd

from carbontwin_r.graph.spectral import graph_laplacian
from carbontwin_r.probabilistic.kalman import kalman_filter_1d
from carbontwin_r.carbon.debt import add_carbon_debt

def test_graph_laplacian_is_symmetric():
    A = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=float)
    L = graph_laplacian(A)
    assert np.allclose(L, L.T)
    assert np.allclose(L.sum(axis=1), 0)

def test_kalman_tracks_constant_signal():
    x = np.ones(100)*3
    means, var = kalman_filter_1d(x, q=0.01, r=0.1)
    assert abs(means[-1] - 3) < 0.1
    assert var[-1] > 0

def test_carbon_debt_exact_case():
    df = pd.DataFrame({
        "residual":[4.0, 0.0],
        "true_wasted_energy":[4.0,0.0],
        "carbon_intensity_kg_per_kwh":[0.5,0.5],
    })
    out = add_carbon_debt(df, dt_hours=0.25)
    assert np.isclose(out["predicted_carbon_debt_kg"].iloc[-1], 0.5)
    assert np.isclose(out["true_carbon_debt_kg"].iloc[-1], 0.5)
