from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass
class DataManifest:
    source: str
    source_url: str
    dataset_id: str
    license: str
    downloaded_at_utc: str
    raw_file: str
    sha256: str
    bytes: int


def write_manifest(path: str | Path, manifest: DataManifest) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")


def make_manifest(*, source: str, source_url: str, dataset_id: str,
                  license: str, raw_file: str | Path) -> DataManifest:
    raw = Path(raw_file)
    return DataManifest(
        source=source,
        source_url=source_url,
        dataset_id=dataset_id,
        license=license,
        downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
        raw_file=str(raw),
        sha256=sha256_file(raw),
        bytes=raw.stat().st_size,
    )
