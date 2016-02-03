# the first step in a probabilistic reasoning tool is figuring out
# the most appropriate distribution given certain quantiles

import numpy as np
import scipy
from scipy.stats import describe, distributions
from scipy.stats import mstats


from utils import proxy, vectorize, hpd


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


proxied = proxy('distribution')

class RandomVariable(np.ndarray):
    # cf. http://docs.scipy.org/doc/numpy-1.10.1/user/basics.subclassing.html
    # for details on subclassing an ndarray
    def __new__(cls, distribution):
        sample = distribution.rvs(N)
        obj = sample.view(cls)
        obj.distribution = distribution
        return obj

    @proxied
    def rvs(self, n):
        return np.random.choice(self, size=n, replace=True)

    @vectorize
    @proxied
    def pdf(self, quantile):
        raise NotImplementedError()

    @vectorize
    @proxied
    def cdf(self, quantile):
        return np.mean(self <= quantile)
        
    @vectorize
    @proxied
    def sf(self, quantile):
        return 1 - self.cdf(quantile)        

    @vectorize
    @proxied
    def ppf(self, prob):
        return mstats.mquantiles(self, prob)[0]

    @vectorize
    @proxied
    def isf(self, prob):
        return mstats.mquantiles(self, 1 - prob)[0] 

    @proxied
    def moment(self, order):
        return mstats.moment(self, order)

    @proxied
    def interval(self, alpha):
        return tuple(hpd(self, 1 - alpha))

    @proxied
    def mean(self, *vargs, **kwargs):
        return np.asarray(self).mean(*vargs, **kwargs)

    @proxied
    def median(self, *vargs, **kwargs):
        return np.asarray(self).median(*vargs, **kwargs)

    @proxied
    def std(self, *vargs, **kwargs):
        return np.asarray(self).std(*vargs, **kwargs)

    @proxied
    def var(self, *vargs, **kwargs):
        return np.asarray(self).var(*vargs, **kwargs)

    # TODO
    # Gamma parameters are not very informative, perhaps it's
    # better to report one or more of mean, median, sd, 
    # MAD and skew instead (using scipy.stats.describe or
    # the population ML estimates for original distributions),
    # and/or the original between arguments used to create it.
    def __repr__(self):
        if hasattr(self, 'distribution'):
            name = self.distribution.dist.name.title()
            parameters = [str(round(arg, 2)) for arg in self.distribution.args]
        else:
            name = 'Transformed'
            parameters = [str(round(arg, 2)) for arg in describe(self)[2:]]
        
        return "<{name}({parameters})>".format(
            name=name,
            parameters=', '.join(parameters)
            )


def between(*quantiles):
    parameters = distributions.gamma.fit(polygon(*quantiles))
    distribution = distributions.gamma(*parameters)
    return RandomVariable(distribution)
