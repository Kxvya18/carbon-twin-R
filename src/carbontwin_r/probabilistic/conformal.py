import numpy as np

def absolute_residual_quantile(y_cal, pred_cal, alpha=0.05):
    scores = np.abs(np.asarray(y_cal) - np.asarray(pred_cal))
    n = len(scores)
    if n == 0:
        raise ValueError("Calibration set is empty.")
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))

def intervals(pred, q):
    pred = np.asarray(pred)
    return pred - q, pred + q

def empirical_coverage(y, lower, upper):
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))
