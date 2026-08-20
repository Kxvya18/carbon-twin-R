import numpy as np
import pandas as pd
from carbontwin_r.models.model_selection import tune_and_select


def test_tuning_runs_with_temporal_cv(tmp_path):
    n = 240
    x = np.arange(n)
    X = pd.DataFrame({"x": x, "sin": np.sin(x/10)})
    y = pd.Series(2*np.sin(x/10)+0.01*x)
    best, results, model = tune_and_select(
        X.iloc[:180], y.iloc[:180], X.iloc[180:], y.iloc[180:],
        candidates=["elastic_net"], budget="quick", n_iter_quick=2,
        cv_splits=3, artifact_dir=tmp_path,
    )
    assert best.model == "elastic_net"
    assert len(results) == 1
    assert (tmp_path/"model_selection.csv").exists()
