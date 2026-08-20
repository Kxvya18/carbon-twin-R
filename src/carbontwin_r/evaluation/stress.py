import numpy as np
import pandas as pd

def add_missingness(df: pd.DataFrame, columns, rate=0.10, seed=42):
    out = df.copy()
    rng = np.random.default_rng(seed)
    for c in columns:
        mask = rng.random(len(out)) < rate
        out.loc[mask, c] = np.nan
    return out

def add_gaussian_sensor_noise(df: pd.DataFrame, columns, relative_sigma=0.05, seed=42):
    out = df.copy()
    rng = np.random.default_rng(seed)
    for c in columns:
        scale = max(float(out[c].std()), 1e-8) * relative_sigma
        out[c] = out[c] + rng.normal(0, scale, len(out))
    return out
