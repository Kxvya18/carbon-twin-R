import numpy as np

def synthetic_carbon_intensity(timestamps, base_kg_per_kwh=0.55):
    hour = timestamps.dt.hour.to_numpy() + timestamps.dt.minute.to_numpy()/60
    # Demo-only variable grid intensity.
    ci = base_kg_per_kwh + 0.08*np.sin(2*np.pi*(hour-15)/24)
    return np.clip(ci, 0.20, 0.90)
