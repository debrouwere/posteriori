"""
An alternative way to get ML estimates for gamma distributions.
Compared to `scipy.stats.distributions.gamma.fit`, it generally
leads to worse estimates near the mass but better estimates 
in the fat tail.
"""

import numpy as np
import scipy as sci

def gamma(lower, mid, upper):
    # correction for small sample bias
    # correction for calculating skew from extreme quantiles only
    # correction to avoid dividing by zero
    skew = 0.01 + 4/3 * sci.stats.skew([lower, mid, upper], bias=False)
    deviation = abs(upper - lower) / 4
    shape = 4 / (skew ** 2)
    scale = deviation / np.sqrt(shape)
    mu = shape * scale
    mean = mid * (3 * shape + 0.2) / (3 * shape - 0.8)
    shift = mean - mu
    return sci.stats.distributions.gamma(a=shape, loc=shift, scale=scale)
