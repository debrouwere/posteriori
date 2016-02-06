import builtins
import functools
import collections
import numpy as np


def replace(string, mapping):
    for match, replacement in mapping.items():
        string = string.replace(match, replacement)
    return string


def bound(value, lower, upper):
    return max(lower, min(upper, value))


def round(value, ndigits=2):
    value = builtins.round(value, ndigits)
    if builtins.round(value) == value:
        return int(value)
    else:
        return value


def vectorize(fn):
    """
    Allows a method to accept one or more values, 
    but internally deal only with a single item.
    """

    @functools.wraps(fn)
    def vectorized_method(self, values, *vargs, **kwargs):
        scalar = not isinstance(values, collections.Iterable)
        
        if scalar:
            values = [values]

        results = [fn(self, value, *vargs, **kwargs) for value in values]

        if scalar:
            results = results[0]

        return results

    return vectorized_method


def proxy(attribute):
    def generic_proxy(fallback_method):
        @functools.wraps(fallback_method)
        def proxied_method(self, *vargs, **kwargs):
            if hasattr(self, attribute):
                obj = getattr(self, attribute)
                proxy_method = getattr(obj, fallback_method.__name__)
                return proxy_method(*vargs, **kwargs)
            else:
                return fallback_method(self, *vargs, **kwargs)
        return proxied_method
    return generic_proxy


def calc_min_interval(x, alpha):
    """
    Internal method to determine the minimum interval of
    a given width.
    
    Extracted from PyMC 3 (Apache license).
    """

    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max


def hpd(x, alpha=0.05):
    """
    Calculate highest posterior density (HPD) of array for given alpha. The HPD is the
    minimum width Bayesian credible interval (BCI).

    Extracted from PyMC 3 (Apache license).
    """

    # Make a copy of trace
    x = x.copy()

    # For multivariate node
    if x.ndim > 1:

        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:]+[0])
        dims = np.shape(tx)

        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1]+(2,))

        for index in make_indices(dims[:-1]):

            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])

            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)

        # Transpose back before returning
        return np.array(intervals)

    else:
        # Sort univariate node
        sx = np.sort(x)

        return np.array(calc_min_interval(sx, alpha))
