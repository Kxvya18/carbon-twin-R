from __future__ import annotations

import numpy as np
import pandas as pd
from .dml import double_ml_ate


def build_real_semisynthetic_intervention(df: pd.DataFrame, seed=42):
    """Create a known-effect intervention benchmark on top of real measured covariates.

    This does NOT claim a field-measured causal intervention. Treatment assignment is deliberately confounded
    by real weather/usage context, while the treatment effect is known, allowing estimator error to be measured.
    """
    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)
    temp = out["To"].to_numpy(dtype=float)
    rh = out["RH_out"].to_numpy(dtype=float)
    lights = out["lights"].to_numpy(dtype=float)
    logits = -0.8 + 0.07*(temp-15) + 0.004*(rh-60) + 0.002*lights
    propensity = 1/(1+np.exp(-logits))
    T = rng.binomial(1, np.clip(propensity, .05, .95))
    # Energy-saving effect varies with outdoor temperature and usage context.
    tau = -(8.0 + 0.7*np.maximum(temp-18, 0) + 0.03*lights)
    noise = rng.normal(0, 4.0, len(out))
    Y = out["Appliances"].to_numpy(dtype=float) + T*tau + noise
    out["treatment"] = T; out["causal_outcome"] = Y; out["true_tau"] = tau
    return out


def evaluate_dml_on_real_context(df: pd.DataFrame, seed=42):
    b = build_real_semisynthetic_intervention(df, seed=seed)
    cols = ["To", "RH_out", "lights", "T1", "RH_1", "T2", "RH_2"]
    estimate = double_ml_ate(b[cols].to_numpy(), b["treatment"].to_numpy(), b["causal_outcome"].to_numpy(), random_state=seed)
    true_ate = float(b["true_tau"].mean())
    estimate["true_ate"] = true_ate
    estimate["absolute_error"] = abs(estimate["ate"]-true_ate)
    estimate["benchmark_type"] = "semi-synthetic known-effect intervention on real measured covariates"
    return estimate
