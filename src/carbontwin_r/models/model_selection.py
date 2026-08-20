from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class CandidateResult:
    model: str
    best_params: dict
    cv_mae_mean: float
    cv_mae_std: float
    selection_mae: float
    selection_rmse: float
    selection_r2: float
    fit_seconds: float


def _specs(seed=42):
    return {
        "elastic_net": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(max_iter=5000, random_state=seed)),
            ]),
            {
                "model__alpha": np.logspace(-3, 1, 30),
                "model__l1_ratio": np.linspace(0.05, 0.95, 19),
            },
        ),
        "random_forest": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(random_state=seed, n_jobs=-1)),
            ]),
            {
                "model__n_estimators": [100, 180, 260],
                "model__max_depth": [8, 14, 22, None],
                "model__min_samples_leaf": [1, 2, 5, 10],
                "model__max_features": [0.5, 0.75, 1.0],
            },
        ),
        "extra_trees": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesRegressor(random_state=seed, n_jobs=-1)),
            ]),
            {
                "model__n_estimators": [100, 180, 260],
                "model__max_depth": [8, 14, 22, None],
                "model__min_samples_leaf": [1, 2, 5],
                "model__max_features": [0.5, 0.75, 1.0],
            },
        ),
        "hist_gradient_boosting": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(random_state=seed)),
            ]),
            {
                "model__learning_rate": [0.02, 0.04, 0.06, 0.10],
                "model__max_iter": [150, 250, 400],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__l2_regularization": [0.0, 0.1, 0.5, 1.0],
                "model__min_samples_leaf": [10, 20, 40],
            },
        ),
    }


def metrics(y, p):
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
    }


def _evaluate_params(estimator, params, X, y, n_splits=4):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_mae = []
    for tr, va in tscv.split(X):
        m = clone(estimator).set_params(**params)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[va])
        fold_mae.append(mean_absolute_error(y.iloc[va], pred))
    return float(np.mean(fold_mae)), float(np.std(fold_mae))


def tune_and_select(X_train: pd.DataFrame, y_train: pd.Series,
                    X_selection: pd.DataFrame, y_selection: pd.Series,
                    candidates=None, *, seed=42, cv_splits=4, budget="quick",
                    n_iter_quick=6, n_iter_research=20, artifact_dir="outputs/research"):
    """Tune only on training folds, then select using a later untouched selection period.

    The final test set is not touched here. This is deliberate: test is reserved for one final estimate.
    """
    specs = _specs(seed)
    candidates = candidates or list(specs)
    n_iter = n_iter_quick if budget == "quick" else n_iter_research
    rng = np.random.RandomState(seed)
    results = []
    fitted = {}

    for name in candidates:
        if name not in specs:
            raise ValueError(f"Unknown model {name}")
        estimator, space = specs[name]
        sampled = list(ParameterSampler(space, n_iter=n_iter, random_state=rng))
        best = None
        start = perf_counter()
        for params in sampled:
            cv_mean, cv_std = _evaluate_params(estimator, params, X_train, y_train, n_splits=cv_splits)
            row = (cv_mean, cv_std, params)
            if best is None or row[0] < best[0]:
                best = row
        best_cv, best_std, best_params = best
        model = clone(estimator).set_params(**best_params)
        model.fit(X_train, y_train)
        pred_sel = model.predict(X_selection)
        sel = metrics(y_selection, pred_sel)
        seconds = perf_counter() - start
        results.append(CandidateResult(
            model=name,
            best_params=best_params,
            cv_mae_mean=best_cv,
            cv_mae_std=best_std,
            selection_mae=sel["mae"],
            selection_rmse=sel["rmse"],
            selection_r2=sel["r2"],
            fit_seconds=seconds,
        ))
        fitted[name] = model

    # Selection set chooses the family; temporal CV chooses its hyperparameters.
    results.sort(key=lambda r: (r.selection_mae, r.cv_mae_mean))
    best = results[0]
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame([r.__dict__ for r in results])
    report["best_params"] = report["best_params"].apply(json.dumps)
    report.to_csv(artifact_dir / "model_selection.csv", index=False)
    (artifact_dir / "best_model.json").write_text(json.dumps(best.__dict__, indent=2, default=str))
    return best, results, fitted[best.model]


def final_refit(model, X_train_selection, y_train_selection, path):
    model = clone(model)
    model.fit(X_train_selection, y_train_selection)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model
