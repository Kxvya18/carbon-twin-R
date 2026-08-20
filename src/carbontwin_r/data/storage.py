from __future__ import annotations
from pathlib import Path
import pandas as pd


def write_table(df: pd.DataFrame, base_path: str | Path) -> Path:
    """Prefer Parquet; fall back to compressed CSV when a parquet engine is unavailable."""
    p = Path(base_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix != ".parquet":
        p = p.with_suffix(".parquet")
    try:
        df.to_parquet(p, index=False)
        return p
    except ImportError:
        q = p.with_suffix(".csv.gz")
        df.to_csv(q, index=False, compression="gzip")
        return q


def read_table(base_path: str | Path, parse_dates=None) -> pd.DataFrame:
    p = Path(base_path)
    candidates = [p, p.with_suffix(".csv.gz")] if p.suffix == ".parquet" else [p]
    for c in candidates:
        if c.exists():
            if c.suffix == ".parquet":
                return pd.read_parquet(c)
            return pd.read_csv(c, parse_dates=parse_dates)
    raise FileNotFoundError(f"No table found for {p}; checked {candidates}")
