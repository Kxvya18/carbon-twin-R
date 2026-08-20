from pathlib import Path
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from carbontwin_r.research_pipeline import simulate_real_scenario

app = FastAPI(title="CarbonTwin-R Research API", version="0.2.0")

class SimulationRequest(BaseModel):
    fault_type: str = "hvac_efficiency_drift"
    severity: float = Field(0.18, ge=0.0, le=1.0)
    start_fraction: float = Field(0.65, ge=0.1, le=0.95)
    zone: int = Field(2, ge=1, le=9)
    seed: int = 42

@app.get("/")
def root():
    return {"service":"CarbonTwin-R", "mode":"research-simulator", "docs":"/docs"}

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/model/registry")
def registry():
    p = Path("outputs/research/metrics/training_registry.json")
    if not p.exists():
        raise HTTPException(404, "Train first with `carbontwin train --config configs/real_uci.yaml`.")
    return json.loads(p.read_text())

@app.post("/simulate")
def simulate(req: SimulationRequest):
    try:
        summary, _, _ = simulate_real_scenario(
            "configs/real_uci.yaml", fault_type=req.fault_type, severity=req.severity,
            start_fraction=req.start_fraction, zone=req.zone, seed=req.seed,
            output_name="api_latest",
        )
        return summary
    except FileNotFoundError as e:
        raise HTTPException(409, str(e)) from e
