import numpy as np

def cusum_positive(x, drift=0.15, threshold=6.0):
    x = np.asarray(x, dtype=float)
    s = 0.0
    scores = np.zeros_like(x)
    alarm = np.zeros(len(x), dtype=int)
    for i, value in enumerate(x):
        s = max(0.0, s + value - drift)
        scores[i] = s
        if s > threshold:
            alarm[i] = 1
            s = 0.0
    return scores, alarm

def bocpd_lite(x, hazard=1/200, sigma=1.5):
    """
    Lightweight Bayesian-style online change score.
    We compare a short recent mean against a long running mean and convert
    the standardized difference to a probability-like score.
    This is intentionally transparent; replace with full BOCPD for a paper.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    score = np.zeros(n)
    short = 24
    long = 192
    for t in range(long, n):
        a = x[t-short:t]
        b = x[t-long:t-short]
        denom = np.sqrt(np.var(a)/max(1,len(a)) + np.var(b)/max(1,len(b)) + 1e-8)
        z = abs(np.mean(a) - np.mean(b)) / denom
        likelihood_change = 1 - np.exp(-0.5 * z*z)
        score[t] = (1-hazard)*likelihood_change + hazard
    return np.clip(score, 0, 1)

def first_detection_index(binary_alarm):
    ids = np.flatnonzero(np.asarray(binary_alarm) > 0)
    return int(ids[0]) if len(ids) else None
