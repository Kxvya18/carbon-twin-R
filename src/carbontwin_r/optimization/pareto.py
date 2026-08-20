import pandas as pd

def pareto_front(df: pd.DataFrame, maximize=("co2_reduction",), minimize=("cost","risk")):
    keep = []
    rows = df.reset_index(drop=True)
    for i, a in rows.iterrows():
        dominated = False
        for j, b in rows.iterrows():
            if i == j:
                continue
            no_worse = all(b[c] >= a[c] for c in maximize) and all(b[c] <= a[c] for c in minimize)
            strictly = any(b[c] > a[c] for c in maximize) or any(b[c] < a[c] for c in minimize)
            if no_worse and strictly:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return rows.iloc[keep].reset_index(drop=True)

def demo_interventions():
    return pd.DataFrame([
        {"intervention":"HVAC setpoint +1C","co2_reduction":0.082,"cost":0.10,"risk":0.15},
        {"intervention":"Schedule optimization","co2_reduction":0.061,"cost":0.03,"risk":0.08},
        {"intervention":"Fan control retuning","co2_reduction":0.074,"cost":0.08,"risk":0.12},
        {"intervention":"Equipment replacement","co2_reduction":0.193,"cost":0.85,"risk":0.35},
    ])
