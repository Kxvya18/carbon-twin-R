import numpy as np

def gaussian_logpdf(x, mu, sigma):
    return -0.5*np.log(2*np.pi*sigma*sigma) - 0.5*((x-mu)/sigma)**2

def sprt(x, healthy_mean=0.0, degraded_mean=2.0, sigma=1.5, alpha=0.05, beta=0.10):
    """
    Returns cumulative LLR and state:
    0=continue/healthy reset, 1=degraded evidence.
    """
    x = np.asarray(x, dtype=float)
    upper = np.log((1-beta)/alpha)
    lower = np.log(beta/(1-alpha))
    llr = 0.0
    history = np.zeros(len(x))
    state = np.zeros(len(x), dtype=int)
    for i, value in enumerate(x):
        llr += gaussian_logpdf(value, degraded_mean, sigma) - gaussian_logpdf(value, healthy_mean, sigma)
        history[i] = llr
        if llr >= upper:
            state[i] = 1
            llr = 0.0
        elif llr <= lower:
            llr = 0.0
    return history, state
