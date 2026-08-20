from __future__ import annotations
import pandas as pd
import numpy as np


class SeasonalMedianBaseline:
    """Time-of-week median baseline with global fallback."""
    def fit(self, df: pd.DataFrame, target="Appliances"):
        tmp = df.copy()
        ts = pd.to_datetime(tmp["date"])
        tmp["dow"] = ts.dt.dayofweek; tmp["hour"] = ts.dt.hour; tmp["minute"] = ts.dt.minute
        self.table_ = tmp.groupby(["dow","hour","minute"])[target].median()
        self.global_ = float(tmp[target].median())
        return self

    def predict(self, df: pd.DataFrame):
        ts = pd.to_datetime(df["date"])
        keys = list(zip(ts.dt.dayofweek, ts.dt.hour, ts.dt.minute))
        return np.asarray([self.table_.get(k, self.global_) for k in keys], dtype=float)
