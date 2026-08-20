import numpy as np
import pandas as pd

BASE_FEATURES = [
    "outdoor_temperature",
    "humidity",
    "occupancy_proxy",
]

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek
    doy = ts.dt.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    for lag in [1, 4, 96]:
        out[f"total_lag_{lag}"] = out["total_power"].shift(lag)

    out["total_roll_mean_4"] = out["total_power"].shift(1).rolling(4).mean()
    out["total_roll_mean_96"] = out["total_power"].shift(1).rolling(96).mean()
    out["total_roll_std_96"] = out["total_power"].shift(1).rolling(96).std()
    return out

def model_feature_columns():
    return BASE_FEATURES + [
        "hour_sin","hour_cos","dow_sin","dow_cos","doy_sin","doy_cos",
        "total_lag_1","total_lag_4","total_lag_96",
        "total_roll_mean_4","total_roll_mean_96","total_roll_std_96",
    ]
