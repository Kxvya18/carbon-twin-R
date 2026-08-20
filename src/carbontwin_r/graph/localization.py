import pandas as pd
import numpy as np
from .facility import NODES

def subsystem_localization(healthy_df, faulty_df):
    # In a real pipeline, use node-wise counterfactual model residuals.
    # For the benchmark, healthy data is available as exact counterfactual truth.
    common = min(len(healthy_df), len(faulty_df))
    rows = []
    for node in NODES:
        diff = np.abs(faulty_df[node].iloc[:common].to_numpy() - healthy_df[node].iloc[:common].to_numpy())
        rows.append({"node": node, "residual_mass": float(diff.sum())})
    out = pd.DataFrame(rows).sort_values("residual_mass", ascending=False).reset_index(drop=True)
    total = out["residual_mass"].sum()
    out["attribution_fraction"] = out["residual_mass"] / total if total > 0 else 0
    return out
