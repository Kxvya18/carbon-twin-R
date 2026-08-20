from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import yaml

from .uci_appliances import download_uci_appliances, load_raw_uci_appliances
from .quality import validate_uci_frame, write_quality_report
from .lineage import sha256_file
from .storage import write_table, read_table


def prepare_real_uci(config_path: str = "configs/real_uci.yaml", download: bool = True) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    project = cfg["project"]
    raw_dir = Path(project["raw_dir"])
    silver_dir = Path(project["silver_dir"])
    gold_dir = Path(project["gold_dir"])
    silver_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / "energydata_complete.csv"
    if download and not raw_path.exists():
        raw_path = download_uci_appliances(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Real UCI data missing at {raw_path}. Run `carbontwin data-download --config {config_path}`."
        )

    raw = load_raw_uci_appliances(raw_path)
    quality = validate_uci_frame(raw)
    write_quality_report(quality, silver_dir / "quality_report.json")
    if not quality["passed"]:
        raise ValueError(f"Data quality checks failed: {quality}")

    # Silver: clean canonical frame. Drop UCI's intentionally random negative-control columns.
    drop_cols = [c for c in cfg["features"].get("exclude_columns", []) if c in raw.columns]
    silver = raw.drop(columns=drop_cols).copy()
    silver = silver.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    numeric_cols = silver.columns.difference(["date"])
    silver[numeric_cols] = silver[numeric_cols].apply(pd.to_numeric, errors="coerce")
    silver_path = write_table(silver, silver_dir / "clean.parquet")

    preparation = {
        "raw_sha256": sha256_file(raw_path),
        "silver_sha256": sha256_file(silver_path),
        "raw_rows": len(raw),
        "silver_rows": len(silver),
        "dropped_negative_controls": drop_cols,
        "target": cfg["features"]["target"],
        "timestamp_start": str(silver["date"].min()),
        "timestamp_end": str(silver["date"].max()),
    }
    (silver_dir / "preparation_manifest.json").write_text(json.dumps(preparation, indent=2), encoding="utf-8")
    return preparation


def load_silver(config_path: str = "configs/real_uci.yaml") -> pd.DataFrame:
    cfg = yaml.safe_load(Path(config_path).read_text())
    path = Path(cfg["project"]["silver_dir"]) / "clean.parquet"
    if not path.exists() and not path.with_suffix(".csv.gz").exists():
        prepare_real_uci(config_path=config_path, download=True)
    df = read_table(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])
    return df
