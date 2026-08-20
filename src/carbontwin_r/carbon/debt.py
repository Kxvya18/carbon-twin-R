import numpy as np
import pandas as pd

def add_carbon_debt(df: pd.DataFrame, dt_hours=0.25) -> pd.DataFrame:
    out = df.copy()
    # Power is treated as kW in the demo, so multiply by interval hours.
    predicted_waste_kwh = np.maximum(0, out["residual"].to_numpy()) * dt_hours
    true_waste_kwh = np.maximum(0, out.get("true_wasted_energy", 0)) * dt_hours
    out["predicted_avoidable_kwh"] = predicted_waste_kwh
    out["true_avoidable_kwh"] = true_waste_kwh
    out["predicted_avoidable_co2_kg"] = predicted_waste_kwh * out["carbon_intensity_kg_per_kwh"]
    out["true_avoidable_co2_kg"] = true_waste_kwh * out["carbon_intensity_kg_per_kwh"]
    out["predicted_carbon_debt_kg"] = out["predicted_avoidable_co2_kg"].cumsum()
    out["true_carbon_debt_kg"] = out["true_avoidable_co2_kg"].cumsum()
    return out


def add_wh_carbon_debt(df, kg_per_kwh=0.23):
    """Carbon Debt when target/residual are already interval energy in Wh (UCI Appliances data)."""
    import numpy as np
    out = df.copy()
    predicted_wh = np.maximum(0, out["residual"].to_numpy(dtype=float))
    true_wh = np.maximum(0, out.get("true_wasted_energy_wh", 0))
    out["predicted_avoidable_kwh"] = predicted_wh / 1000.0
    out["true_avoidable_kwh"] = true_wh / 1000.0
    out["carbon_intensity_kg_per_kwh"] = float(kg_per_kwh)
    out["predicted_avoidable_co2_kg"] = out["predicted_avoidable_kwh"] * kg_per_kwh
    out["true_avoidable_co2_kg"] = out["true_avoidable_kwh"] * kg_per_kwh
    out["predicted_carbon_debt_kg"] = out["predicted_avoidable_co2_kg"].cumsum()
    out["true_carbon_debt_kg"] = out["true_avoidable_co2_kg"].cumsum()
    return out
