import numpy as np
from scipy.stats.distributions import gamma, norm, uniform


# TODO: ideally, this function should be able to deal with an 
# arbitrary map of [(quantile, cumulative_probability)]
def interpolate(*quantiles, bounds=(0.05, 0.95)):
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
