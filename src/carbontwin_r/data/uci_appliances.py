from __future__ import annotations

from pathlib import Path
import io
import shutil
import zipfile
import requests
import pandas as pd

from .lineage import make_manifest, write_manifest

UCI_ID = 374
UCI_SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/374/appliances%2Benergy%2Bprediction.zip"
CSV_NAME = "energydata_complete.csv"


def download_uci_appliances(raw_dir: str | Path = "data/raw/uci_appliances", force: bool = False) -> Path:
    """Download the real UCI Appliances Energy Prediction data set.

    Raw data are kept immutable. A source manifest with SHA-256 is written next to the file.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / CSV_NAME
    manifest_path = raw_dir / "source_manifest.json"
    if csv_path.exists() and manifest_path.exists() and not force:
        return csv_path

    response = requests.get(UCI_ZIP_URL, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        matches = [n for n in zf.namelist() if n.endswith(CSV_NAME)]
        if not matches:
            raise RuntimeError(f"{CSV_NAME} not found in UCI archive")
        with zf.open(matches[0]) as src, csv_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    manifest = make_manifest(
        source="UCI Machine Learning Repository",
        source_url=UCI_SOURCE_PAGE,
        dataset_id=str(UCI_ID),
        license="CC BY 4.0",
        raw_file=csv_path,
    )
    write_manifest(manifest_path, manifest)
    return csv_path


UCI_COLUMN_ALIASES = {
    "T_out": "To",
    "Press_mm_hg": "Pressure",
}


def canonicalize_uci_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rename = {
        source: target
        for source, target in UCI_COLUMN_ALIASES.items()
        if source in out.columns and target not in out.columns
    }

    return out.rename(columns=rename)


def load_raw_uci_appliances(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = canonicalize_uci_columns(df)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise"
    )

    return (
        df.sort_values("date")
        .reset_index(drop=True)
    )
