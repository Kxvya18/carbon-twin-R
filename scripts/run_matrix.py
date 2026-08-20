"""Run the real-data controlled-fault evaluation matrix without retraining for every scenario."""
from pathlib import Path
import itertools
import json
import pandas as pd
import yaml

from carbontwin_r.research_pipeline import train_real_models, simulate_real_scenario

CONFIG = "configs/real_uci.yaml"
MATRIX = "configs/research.yaml"

if __name__ == "__main__":
    cfg = yaml.safe_load(Path(MATRIX).read_text())
    train_real_models(CONFIG, budget=cfg.get("search_budget", "research"))
    rows = []
    combos = itertools.product(
        cfg["seeds"], cfg["fault_types"], cfg["severities"], cfg["start_fractions"], cfg["zones"]
    )
    for k, (seed, fault, severity, start, zone) in enumerate(combos, 1):
        name = f"{k:04d}_{fault}_sev{severity}_start{start}_z{zone}_seed{seed}"
        summary, _, _ = simulate_real_scenario(
            CONFIG, fault_type=fault, severity=severity, start_fraction=start,
            zone=zone, seed=seed, output_name=name,
        )
        rows.append(summary)
        print(k, name)
    out = Path("outputs/research/metrics/scenario_matrix.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)
