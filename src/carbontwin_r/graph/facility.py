from dataclasses import dataclass
import numpy as np

NODES = [
    "cooling_power",
    "heating_power",
    "fan_power",
    "pump_power",
    "lighting_power",
    "plug_power",
]

@dataclass
class FacilityGraph:
    nodes: list[str]
    adjacency: np.ndarray

def build_engineering_prior_graph():
    # Cooling connects to fan and pump; heating connects to fan/pump;
    # lighting and plug are operationally coupled through occupancy.
    A = np.zeros((6,6), dtype=float)
    edges = [(0,2),(0,3),(1,2),(1,3),(2,3),(4,5)]
    for i,j in edges:
        A[i,j] = A[j,i] = 1.0
    return FacilityGraph(nodes=NODES, adjacency=A)
