from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.preprocessing import StandardScaler

ROOM_IDS = list(range(1, 10))


@dataclass
class LearnedRoomGraph:
    adjacency: np.ndarray
    partial_correlation: np.ndarray
    alpha: float
    room_names: list[str]
    converged: bool
    convergence_warnings: list[str]


def _room_matrix(df: pd.DataFrame) -> np.ndarray:
    blocks = []
    for i in ROOM_IDS:
        blocks.append(df[[f"T{i}", f"RH_{i}"]].to_numpy(dtype=float))
    # shape [n, room, 2], then summarize each room to two standardized dimensions later.
    return np.stack(blocks, axis=1)


def learn_room_graph(train_df: pd.DataFrame, edge_threshold: float = 0.08) -> LearnedRoomGraph:
    """Learn a sparse conditional-dependence graph from training data only.

    Graphical Lasso is fitted over 18 room temperature/humidity variables. Edge strength between two rooms
    is the maximum absolute partial correlation across their T/RH variable pairs. This avoids leaking test data
    into topology estimation and makes the graph a statistical object rather than a hand-drawn diagram.
    """
    cols = []
    for i in ROOM_IDS:
        cols += [f"T{i}", f"RH_{i}"]
    X = train_df[cols].astype(float).dropna().to_numpy()
    Xs = StandardScaler().fit_transform(X)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        gl = GraphicalLassoCV(alphas=np.logspace(-2, 0, 8), cv=4, max_iter=1000, tol=1e-4).fit(Xs)
    conv = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    precision = gl.precision_
    d = np.sqrt(np.diag(precision))
    partial = -precision / np.outer(d, d)
    np.fill_diagonal(partial, 1.0)

    A = np.zeros((9, 9), dtype=float)
    for i in range(9):
        for j in range(i+1, 9):
            block = partial[2*i:2*i+2, 2*j:2*j+2]
            w = float(np.max(np.abs(block)))
            if w >= edge_threshold:
                A[i,j] = A[j,i] = w
    return LearnedRoomGraph(A, partial, float(gl.alpha_), [f"zone_{i}" for i in ROOM_IDS], not bool(conv), conv)


def graph_laplacian(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    return np.diag(A.sum(axis=1)) - A


def zone_state_matrix(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> np.ndarray:
    """One robust standardized thermal state per room per time."""
    ref = reference if reference is not None else df
    states = []
    for i in ROOM_IDS:
        t_med = float(ref[f"T{i}"].median()); t_scale = float(ref[f"T{i}"].mad()) if hasattr(ref[f"T{i}"], "mad") else 0
        rh_med = float(ref[f"RH_{i}"].median()); rh_scale = float(ref[f"RH_{i}"].std())
        t_scale = float(ref[f"T{i}"].std()) if not np.isfinite(t_scale) or t_scale < 1e-6 else 1.4826*t_scale
        rh_scale = max(rh_scale, 1e-6)
        zt = (df[f"T{i}"].to_numpy() - t_med) / max(t_scale, 1e-6)
        zr = (df[f"RH_{i}"].to_numpy() - rh_med) / rh_scale
        states.append(np.sqrt(zt*zt + zr*zr))
    return np.stack(states, axis=1)


def graph_energy_series(states: np.ndarray, A: np.ndarray) -> np.ndarray:
    L = graph_laplacian(A)
    return np.einsum("bi,ij,bj->b", states, L, states)


def localize_zones(observed_df: pd.DataFrame, healthy_reference: pd.DataFrame, A: np.ndarray) -> pd.DataFrame:
    """Localize a zone with healthy-reference residuals plus graph-neighbor inconsistency.

    For each zone, two ridge models predict its temperature and humidity from outside conditions,
    clock features and the *other* zones. Standardized residual energy is then weighted by graph degree.
    This uses only the healthy reference to fit predictors; the fault period is inference-only.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    def context(frame, excluded_zone):
        ts = pd.to_datetime(frame["date"])
        out = pd.DataFrame(index=frame.index)
        out["To"] = frame["To"].to_numpy(); out["RH_out"] = frame["RH_out"].to_numpy()
        minute = ts.dt.hour*60 + ts.dt.minute
        out["tod_sin"] = np.sin(2*np.pi*minute/(24*60)); out["tod_cos"] = np.cos(2*np.pi*minute/(24*60))
        for j in ROOM_IDS:
            if j != excluded_zone:
                out[f"T{j}"] = frame[f"T{j}"].to_numpy()
                out[f"RH_{j}"] = frame[f"RH_{j}"].to_numpy()
        return out

    rows = []
    degree = np.asarray(A).sum(axis=1)
    for pos, i in enumerate(ROOM_IDS):
        Xh = context(healthy_reference, i); Xo = context(observed_df, i)
        total = 0.0
        for target in [f"T{i}", f"RH_{i}"]:
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=2.0))
            model.fit(Xh, healthy_reference[target])
            rh = healthy_reference[target].to_numpy() - model.predict(Xh)
            ro = observed_df[target].to_numpy() - model.predict(Xo)
            scale = max(float(np.std(rh)), 1e-6)
            z = ro/scale
            # Tail-sensitive score: controlled drift should dominate ordinary residual noise.
            total += float(np.sum(np.minimum(z*z, 100.0)))
        graph_weight = 1.0 + 0.15*degree[pos]
        rows.append({"zone": f"zone_{i}", "graph_anomaly_score": total*graph_weight})
    out = pd.DataFrame(rows).sort_values("graph_anomaly_score", ascending=False).reset_index(drop=True)
    denom = float(out["graph_anomaly_score"].sum()) or 1.0
    out["attribution_fraction"] = out["graph_anomaly_score"]/denom
    return out
