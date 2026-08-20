import numpy as np
from carbontwin_r.probabilistic.conformal import absolute_residual_quantile, intervals, empirical_coverage

def test_conformal_interval_basic():
    y = np.array([0,1,2,3,4], dtype=float)
    p = y + np.array([0.1,-0.1,0.2,-0.2,0.0])
    q = absolute_residual_quantile(y,p,alpha=0.2)
    lo,hi = intervals(p,q)
    assert empirical_coverage(y,lo,hi) >= 0.8
