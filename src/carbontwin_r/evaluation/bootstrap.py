import numpy as np

def block_bootstrap_difference(a, b, block_size=32, n_boot=500, seed=42):
    """
    CI for mean(a-b) while sampling contiguous blocks.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("a and b must have same length")
    n = len(a)
    rng = np.random.default_rng(seed)
    diffs = []
    starts = np.arange(max(1, n-block_size+1))
    for _ in range(n_boot):
        sample_idx = []
        while len(sample_idx) < n:
            s = int(rng.choice(starts))
            sample_idx.extend(range(s, min(s+block_size, n)))
        sample_idx = np.asarray(sample_idx[:n])
        diffs.append(np.mean(a[sample_idx]-b[sample_idx]))
    return {
        "mean_difference": float(np.mean(a-b)),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
    }
