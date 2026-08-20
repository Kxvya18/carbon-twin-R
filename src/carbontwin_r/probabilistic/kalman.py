import numpy as np
from scipy.stats import norm

def kalman_filter_1d(observations, q=0.02, r=0.60, x0=0.0, p0=1.0):
    obs = np.asarray(observations, dtype=float)
    x = float(x0)
    p = float(p0)
    means, variances = [], []

    for z in obs:
        # Predict: F=H=1
        x_pred = x
        p_pred = p + q

        # Update
        innovation = z - x_pred
        s = p_pred + r
        k = p_pred / s
        x = x_pred + k * innovation
        p = (1 - k) * p_pred

        means.append(x)
        variances.append(p)

    return np.asarray(means), np.asarray(variances)

def probability_above_threshold(means, variances, threshold):
    sd = np.sqrt(np.maximum(variances, 1e-12))
    z = (threshold - np.asarray(means)) / sd
    return 1.0 - norm.cdf(z)
