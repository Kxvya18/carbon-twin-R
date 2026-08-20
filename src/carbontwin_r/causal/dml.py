import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.base import clone

def double_ml_ate(X, T, Y, random_state=42, n_splits=3):
    """
    Simple cross-fitted partially-linear Double ML estimator.
    theta = E[(T-m(X))(Y-g(X))] / E[(T-m(X))^2]
    """
    X = np.asarray(X)
    T = np.asarray(T).astype(float)
    Y = np.asarray(Y).astype(float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_res = np.zeros_like(Y, dtype=float)
    t_res = np.zeros_like(T, dtype=float)

    g_base = RandomForestRegressor(n_estimators=150, min_samples_leaf=8, random_state=random_state)
    m_base = RandomForestClassifier(n_estimators=150, min_samples_leaf=8, random_state=random_state)

    for train, test in kf.split(X):
        g = clone(g_base).fit(X[train], Y[train])
        m = clone(m_base).fit(X[train], T[train])
        y_res[test] = Y[test] - g.predict(X[test])
        t_res[test] = T[test] - m.predict_proba(X[test])[:,1]

    denom = np.sum(t_res*t_res)
    if denom <= 1e-12:
        raise ValueError("Treatment residual variance too small.")
    theta = float(np.sum(t_res*y_res) / denom)

    psi = t_res * (y_res - theta*t_res)
    se = float(np.sqrt(np.mean(psi**2) / (np.mean(t_res**2)**2 * len(X))))
    return {"ate": theta, "se": se, "ci_low": theta - 1.96*se, "ci_high": theta + 1.96*se}
