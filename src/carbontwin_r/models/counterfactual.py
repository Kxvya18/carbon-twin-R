from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def build_counterfactual_model(random_state=42, max_iter=300):
    # Default uses a strong sklearn model so the core project installs cleanly.
    # Swap with XGBoost after installing requirements-advanced.txt.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=random_state,
        )),
    ])

def add_counterfactual_columns(df, prediction, lower=None, upper=None):
    out = df.copy()
    out["counterfactual_power"] = prediction
    out["residual"] = out["total_power"] - out["counterfactual_power"]
    if lower is not None:
        out["pred_lower"] = lower
    if upper is not None:
        out["pred_upper"] = upper
    return out
