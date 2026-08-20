from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


def validate_uci_frame(df: pd.DataFrame) -> dict:
    required = {"date", "Appliances", "lights", "T1", "RH_1", "T9", "RH_9", "To", "RH_out"}
    missing_columns = sorted(required - set(df.columns))
    ts = pd.to_datetime(df["date"], errors="coerce") if "date" in df else pd.Series(dtype="datetime64[ns]")
    diffs = ts.diff().dropna().dt.total_seconds().div(60)
    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_required_columns": missing_columns,
        "duplicate_timestamps": int(ts.duplicated().sum()) if len(ts) else 0,
        "invalid_timestamps": int(ts.isna().sum()) if len(ts) else len(df),
        "monotonic_timestamp": bool(ts.is_monotonic_increasing) if len(ts) else False,
        "median_interval_minutes": float(diffs.median()) if len(diffs) else None,
        "irregular_intervals": int((~np.isclose(diffs, 10.0)).sum()) if len(diffs) else 0,
        "missing_cells": int(df.isna().sum().sum()),
        "negative_target_rows": int((pd.to_numeric(df.get("Appliances", pd.Series(dtype=float)), errors="coerce") < 0).sum()),
        "excluded_negative_controls_present": [c for c in ["rv1", "rv2"] if c in df.columns],
    }
    report["passed"] = (
        not missing_columns
        and report["duplicate_timestamps"] == 0
        and report["invalid_timestamps"] == 0
        and report["negative_target_rows"] == 0
    )
    return report


def write_quality_report(report: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
