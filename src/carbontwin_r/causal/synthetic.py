import numpy as np
import pandas as pd

def make_intervention_benchmark(df: pd.DataFrame, seed=42):
    """
    Semi-synthetic treatment benchmark.
    Treatment probability depends on observed context, so naive mean difference is confounded.
    True treatment effect is known.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    temp = out["outdoor_temperature"].to_numpy()
    occ = out["occupancy_proxy"].to_numpy()
    logit = -1.0 + 0.05*(temp-24) + 0.8*occ
    p = 1/(1+np.exp(-logit))
    treatment = rng.binomial(1, p)
    true_tau = -(0.5 + 0.08*np.maximum(temp-24,0) + 0.3*occ)  # kW reduction
    noise = rng.normal(0, 0.4, len(out))
    baseline = out["total_power"].to_numpy()
    outcome = baseline + treatment*true_tau + noise
    out["treatment"] = treatment
    out["causal_outcome"] = outcome
    out["true_tau"] = true_tau
    return out
