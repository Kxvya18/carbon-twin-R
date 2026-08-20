from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def population_stability_index(expected, actual, bins=10):
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e = e[np.isfinite(e)]; a = a[np.isfinite(a)]
    if len(e) < bins or len(a) < bins:
        return np.nan
    cuts = np.unique(np.quantile(e, np.linspace(0, 1, bins+1)))
    if len(cuts) < 3:
        return 0.0
    e_hist, _ = np.histogram(e, bins=cuts)
    a_hist, _ = np.histogram(a, bins=cuts)
    e_pct = np.maximum(e_hist / max(1, e_hist.sum()), 1e-6)
    a_pct = np.maximum(a_hist / max(1, a_hist.sum()), 1e-6)
    return float(np.sum((a_pct-e_pct)*np.log(a_pct/e_pct)))


def drift_report(reference: pd.DataFrame, recent: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for c in columns:
        r = pd.to_numeric(reference[c], errors="coerce").dropna()
        q = pd.to_numeric(recent[c], errors="coerce").dropna()
        if len(r) < 20 or len(q) < 20:
            continue
        ks = ks_2samp(r, q)
        rows.append({"feature": c, "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
                     "psi": population_stability_index(r, q)})
    return pd.DataFrame(rows).sort_values("ks_stat", ascending=False)
