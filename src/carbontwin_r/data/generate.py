import numpy as np
import pandas as pd

def make_synthetic_facility(n_steps: int = 3000, freq_minutes: int = 15, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n_steps, freq=f"{freq_minutes}min", tz="UTC")
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    dow = ts.dayofweek.to_numpy()

    daily = np.sin(2 * np.pi * (hour - 6) / 24)
    seasonal = np.sin(np.linspace(0, 2 * np.pi, n_steps))
    outdoor_temp = 24 + 7 * daily + 2 * seasonal + rng.normal(0, 1.0, n_steps)
    humidity = np.clip(60 - 0.7 * (outdoor_temp - 24) + rng.normal(0, 4, n_steps), 25, 95)

    work_hour = ((hour >= 8) & (hour <= 19) & (dow < 5)).astype(float)
    occupancy = np.clip(work_hour * (0.75 + 0.20 * np.sin(np.pi * (hour - 8) / 11)) + rng.normal(0, 0.04, n_steps), 0, 1)

    cooling = np.maximum(0, (outdoor_temp - 22)) * (1.8 + 0.8 * occupancy) + rng.normal(0, 0.7, n_steps)
    heating = np.maximum(0, (19 - outdoor_temp)) * 1.5 + rng.normal(0, 0.3, n_steps)
    fan = 2.0 + 5.0 * occupancy + 0.12 * cooling + rng.normal(0, 0.35, n_steps)
    pump = 1.0 + 0.10 * cooling + 0.08 * heating + rng.normal(0, 0.20, n_steps)
    lighting = 1.2 + 7.0 * occupancy + rng.normal(0, 0.45, n_steps)
    plug = 2.2 + 5.5 * occupancy + rng.normal(0, 0.40, n_steps)

    for arr in (cooling, heating, fan, pump, lighting, plug):
        arr[arr < 0] = 0

    total = cooling + heating + fan + pump + lighting + plug

    return pd.DataFrame({
        "timestamp": ts,
        "total_power": total,
        "cooling_power": cooling,
        "heating_power": heating,
        "fan_power": fan,
        "pump_power": pump,
        "lighting_power": lighting,
        "plug_power": plug,
        "outdoor_temperature": outdoor_temp,
        "humidity": humidity,
        "occupancy_proxy": occupancy,
        "is_fault": 0,
        "fault_type": "none",
        "fault_node": "none",
        "fault_severity": 0.0,
        "true_wasted_energy": 0.0,
    })
