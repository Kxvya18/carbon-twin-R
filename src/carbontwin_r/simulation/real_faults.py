from __future__ import annotations

import numpy as np
import pandas as pd

FAULTS = ["hvac_efficiency_drift", "standby_load", "lighting_schedule", "sensor_bias"]


def inject_real_fault(df: pd.DataFrame, *, fault_type="hvac_efficiency_drift", severity=0.18,
                      start_fraction=0.65, ramp_steps=600, zone=2, seed=42) -> pd.DataFrame:
    if fault_type not in FAULTS:
        raise ValueError(f"fault_type must be one of {FAULTS}")
    if not (0 <= severity <= 1):
        raise ValueError("severity must be in [0,1]")
    out = df.copy(deep=True).reset_index(drop=True)
    n = len(out)
    start = int(n*start_fraction)
    idx = np.arange(n)
    ramp = np.clip((idx-start)/max(1, ramp_steps), 0, 1)
    active = idx >= start

    out["healthy_target"] = out["Appliances"].astype(float)
    out["observed_target"] = out["healthy_target"].copy()
    out["true_wasted_energy_wh"] = 0.0
    out["is_fault"] = 0
    out["fault_type"] = "none"
    out["fault_node"] = "none"
    out["fault_severity"] = 0.0

    baseline_scale = max(20.0, float(out["Appliances"].quantile(0.75)))
    extra = np.zeros(n)
    fault_node = f"zone_{zone}"

    if fault_type == "hvac_efficiency_drift":
        # Extra appliance energy grows gradually; one thermal zone drifts, giving graph evidence.
        weather_pressure = np.maximum(0, out["To"].to_numpy() - 18.0) / 10.0 + 0.5
        extra = severity * ramp * baseline_scale * weather_pressure
        out[f"T{zone}"] = out[f"T{zone}"] + (8.0*severity*ramp)
        out[f"RH_{zone}"] = out[f"RH_{zone}"] + (20.0*severity*ramp)
    elif fault_type == "standby_load":
        hour = pd.to_datetime(out["date"]).dt.hour.to_numpy()
        mask = active & ((hour <= 6) | (hour >= 23))
        extra[mask] = severity * baseline_scale
        fault_node = "standby_load"
    elif fault_type == "lighting_schedule":
        hour = pd.to_datetime(out["date"]).dt.hour.to_numpy()
        mask = active & ((hour >= 20) | (hour <= 5))
        lighting_extra = severity * max(10.0, float(out["lights"].quantile(0.90)))
        out.loc[mask, "lights"] = out.loc[mask, "lights"] + lighting_extra
        extra[mask] = lighting_extra
        fault_node = "lighting"
    elif fault_type == "sensor_bias":
        # Negative control: data-quality failure, not true energy waste.
        out.loc[active, f"T{zone}"] = out.loc[active, f"T{zone}"] + 3.0*severity
        out.loc[active, f"RH_{zone}"] = out.loc[active, f"RH_{zone}"] + 10.0*severity
        fault_node = f"zone_{zone}_sensor"

    out["observed_target"] = out["healthy_target"] + extra
    out["Appliances"] = out["observed_target"]
    out["true_wasted_energy_wh"] = extra
    fault_mask = active & ((extra > 0) | (fault_type == "sensor_bias"))
    out.loc[fault_mask, "is_fault"] = 1
    out.loc[fault_mask, "fault_type"] = fault_type
    out.loc[fault_mask, "fault_node"] = fault_node
    out.loc[fault_mask, "fault_severity"] = severity if fault_type != "hvac_efficiency_drift" else severity*ramp[fault_mask]
    return out
