from __future__ import annotations

import numpy as np
import pandas as pd

ROOMS = list(range(1, 10))
ROOM_FEATURES = [f"T{i}" for i in ROOMS] + [f"RH_{i}" for i in ROOMS]
WEATHER_FEATURES = ["To", "Pressure", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]


def add_real_energy_features(df: pd.DataFrame, *, target: str = "Appliances",
                             target_lags=(1, 6, 144), rolling_windows=(6, 36, 144),
                             use_target_history: bool = False) -> pd.DataFrame:
    """Leakage-safe feature construction.

    All target-derived features are shifted by at least one observation. They are optional because a
    counterfactual detector may otherwise absorb a persistent fault through autoregressive feedback.
    """
    out = df.copy()
    ts = pd.to_datetime(out["date"])
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    out["tod_sin"] = np.sin(2*np.pi*minute_of_day/(24*60))
    out["tod_cos"] = np.cos(2*np.pi*minute_of_day/(24*60))
    out["dow_sin"] = np.sin(2*np.pi*ts.dt.dayofweek/7)
    out["dow_cos"] = np.cos(2*np.pi*ts.dt.dayofweek/7)
    out["doy_sin"] = np.sin(2*np.pi*ts.dt.dayofyear/365.25)
    out["doy_cos"] = np.cos(2*np.pi*ts.dt.dayofyear/365.25)
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    temp_cols = [c for c in [f"T{i}" for i in ROOMS] if c in out]
    rh_cols = [c for c in [f"RH_{i}" for i in ROOMS] if c in out]
    out["indoor_temp_mean"] = out[temp_cols].mean(axis=1)
    out["indoor_temp_std"] = out[temp_cols].std(axis=1)
    out["indoor_rh_mean"] = out[rh_cols].mean(axis=1)
    out["thermal_delta_outdoor"] = out["indoor_temp_mean"] - out["To"]

    if use_target_history:
        for lag in target_lags:
            out[f"{target}_lag_{lag}"] = out[target].shift(lag)
        for w in rolling_windows:
            shifted = out[target].shift(1)
            out[f"{target}_roll_mean_{w}"] = shifted.rolling(w).mean()
            out[f"{target}_roll_std_{w}"] = shifted.rolling(w).std()

    return out


def real_model_feature_columns(df: pd.DataFrame, *, target: str = "Appliances",
                               use_target_history: bool = False) -> list[str]:
    forbidden = {"date", target, "healthy_target", "observed_target", "is_fault",
                 "fault_type", "fault_node", "fault_severity", "true_wasted_energy_wh"}
    cols = []
    for c in df.columns:
        if c in forbidden or c in {"rv1", "rv2"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if not use_target_history and c.startswith(f"{target}_"):
                continue
            cols.append(c)
    return cols
