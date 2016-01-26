# the first step in a probabilistic reasoning tool is figuring out
# the most appropriate distribution given certain quantiles

import numpy as np
import scipy
from scipy.stats import distributions


N = 10000


def polygon(*quantiles, bounds=(0.05, 0.95)):
    """
    Construct a polygon distribution from quantile estimates
    spaced evenly between a lower and upper bound.

    Triangle and polygon distributions are a useful pedagogical tool.

    Here, they are mainly useful because they allow for a more 
    accurate estimate of skew: we wish to calculate the skew 
    of the distribution which produced these various quantiles, 
    not the skew of the censored distribution which includes
    only extreme quantiles.
    """

    quantiles = np.array(quantiles)
    lower, upper = np.array(bounds) * 1000
    knots = np.round(np.linspace(lower, upper, num=len(quantiles)))
    
    rises = quantiles[1:] - quantiles[:-1]
    runs = knots[1:] - knots[:-1]
    slopes = rises / runs
    
    runs = [25] + list(runs) + [25]
    slopes = [slopes[0]] + list(slopes) + [slopes[-1]]
    
    intercept = quantiles[0] - runs[0] * slopes[0]
    poly = [intercept]
    for slope, run in zip(slopes, runs):
        start = poly[-1] + slope
        stop = poly[-1] + run * slope
        poly.extend(np.linspace(start, stop, num=run))
    
    return poly


METHODS = [
    'rvs', 'pdf', 'cdf', 'sf', 'ppf', 'isf',
    'moment', 'median', 'mean', 'std', 'var', 'interval'
    ]

# Does the job, but we need to better control what happens when
# doing calculations -- ideally the result would still be a random
# variable but the methods above would use Monte Carlo simulation
# either directly or -- in simple cases -- by calculating the 
# parameters of the new distribution (e.g. normal + normal)
#
# So we need implementations of rvs, pdf, cdf, sf, ppf and isf
# (which should be easy enough.)
# 
# cf. http://docs.scipy.org/doc/numpy-1.10.1/user/basics.subclassing.html
class RandomVariable(np.ndarray):
    def __new__(cls, distribution):
        sample = distribution.rvs(N)
        obj = sample.view(cls)
        obj.distribution = distribution
        for method in METHODS:
            setattr(obj, method, getattr(distribution, method))
        return obj

    def __array_finalize__(self, obj):
        self.distribution = getattr(obj, 'distribution', None)

    # Gamma parameters are not very informative, perhaps it's
    # better to report one or more of mean, median, sd, 
    # MAD and skew instead
    def __repr__(self):
        dist = self.distribution
        parameters = [str(round(arg, 2)) for arg in dist.args]
        return "<{name}({parameters})>".format(
            name=dist.dist.name.title(),
            parameters=', '.join(parameters)
            )


def between(*quantiles):
    parameters = distributions.gamma.fit(polygon(*quantiles))
    distribution = distributions.gamma(*parameters)
    return RandomVariable(distribution)
