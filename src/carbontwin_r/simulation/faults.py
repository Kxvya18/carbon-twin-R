import numpy as np
import pandas as pd

def inject_fault(df: pd.DataFrame, fault_type="gradual_cooling", start_fraction=0.70,
                 severity=0.18, ramp_steps=500, seed=42) -> pd.DataFrame:
    out = df.copy(deep=True)
    n = len(out)
    start = int(n * start_fraction)
    idx = np.arange(n)
    effect = np.zeros(n, dtype=float)
    node = "cooling_power"

    if fault_type == "gradual_cooling":
        ramp = np.clip((idx - start) / max(1, ramp_steps), 0, 1)
        effect = severity * ramp
        node = "cooling_power"
        delta = out[node].to_numpy() * effect
        out[node] += delta
    elif fault_type == "sudden_cooling":
        effect[idx >= start] = severity
        node = "cooling_power"
        delta = out[node].to_numpy() * effect
        out[node] += delta
    elif fault_type == "stuck_fan":
        node = "fan_power"
        off_hours = out["occupancy_proxy"].to_numpy() < 0.1
        mask = (idx >= start) & off_hours
        delta = np.zeros(n)
        delta[mask] = severity * max(1.0, float(out[node].quantile(0.9)))
        effect[mask] = severity
        out[node] += delta
    elif fault_type == "lighting_schedule":
        node = "lighting_power"
        hour = pd.to_datetime(out["timestamp"], utc=True).dt.hour.to_numpy()
        mask = (idx >= start) & ((hour >= 20) | (hour <= 2))
        delta = np.zeros(n)
        delta[mask] = severity * max(1.0, float(out[node].quantile(0.9)))
        effect[mask] = severity
        out[node] += delta
    else:
        raise ValueError(f"Unknown fault_type={fault_type}")

    out["total_power"] = out[
        ["cooling_power","heating_power","fan_power","pump_power","lighting_power","plug_power"]
    ].sum(axis=1)
    out["is_fault"] = (effect > 0).astype(int)
    out["fault_type"] = np.where(out["is_fault"].eq(1), fault_type, "none")
    out["fault_node"] = np.where(out["is_fault"].eq(1), node, "none")
    out["fault_severity"] = effect
    out["true_wasted_energy"] = delta
    return out
