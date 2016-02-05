# the first step in a probabilistic reasoning tool is figuring out
# the most appropriate distribution given certain quantiles

from itertools import product, repeat
import numpy as np
import scipy
from scipy.stats import mstats, describe
from distributions import gamma, norm, uniform
from distributions import interpolate

from utils import vectorize, hpd


N = 10000


def distribute(method):
    def distributed_method(self, *vargs, **kwargs):
        if hasattr(self, 'distributions'):
            results = {key: getattr(distribution, method.__name__)(*vargs, **kwargs)
                for key, distribution in self.distributions.items()}
        else:
            results = {method(self[option], *vargs, **kwargs)
                for option in self.set}

        if len(results) > 1:
            return results
        else:
            return results.values()[0]

    return distributed_method


class RandomVector(np.ndarray):
    # cf. http://docs.scipy.org/doc/numpy-1.10.1/user/basics.subclassing.html
    # for details on subclassing an ndarray
    def __new__(cls, distributions):
        samples = np.vstack([distribution.rvs(N)
            for name, distribution in distributions.items()])
        obj = samples.view(cls)
        obj.distributions = distributions
        obj.factors = [tuple(distributions.keys())]
        return obj

    def __array_finalize__(self, obj):
        self.factors = getattr(obj, 'factors', [])

    def index(self, *items):
        ixs = list(repeat(slice(None), self.ndim - 1))
        for dim, categories in enumerate(self.factors):
            for row, category in enumerate(categories):
                if category in items:
                    ixs[dim] = row

        return ixs

    @property
    def set(self):
        if self.factors:
            return [self.index(*option) for option in product(*self.factors)]
        else:
            return [slice(None)]

    def __getitem__(self, name):
        if isinstance(name, str):
            category = self.index(name)
            dim, row = next((dim, row) for dim, row in enumerate(cat) if row != slice(None))
            if dim is None:
                raise KeyError(name)
            subset = np.squeeze(self[category])
            subset.factors = self.factors[:]
            subset.factors.pop(dim)
            return subset
        else:
            return super().__getitem__(name)

    @distribute
    def rvs(self, n):
        return np.random.choice(self, size=n, replace=True)

    @vectorize
    @distribute
    def pdf(self, quantile):
        raise NotImplementedError()

    @vectorize
    @distribute
    def cdf(self, quantile):
        return np.mean(self <= quantile)
        
    @vectorize
    @distribute
    def sf(self, quantile):
        return 1 - self.cdf(quantile)        

    @vectorize
    @distribute
    def ppf(self, prob):
        return mstats.mquantiles(self, prob)[0]

    @vectorize
    @distribute
    def isf(self, prob):
        return mstats.mquantiles(self, 1 - prob)[0] 

    @distribute
    def moment(self, order):
        return mstats.moment(self, order)

    @distribute
    def interval(self, alpha):
        return tuple(hpd(self, 1 - alpha))

    @distribute
    def mean(self, *vargs, **kwargs):
        return np.asarray(self).mean(*vargs, **kwargs)

    @distribute
    def median(self, *vargs, **kwargs):
        return np.asarray(self).median(*vargs, **kwargs)

    @distribute
    def std(self, *vargs, **kwargs):
        return np.asarray(self).std(*vargs, **kwargs)

    @distribute
    def var(self, *vargs, **kwargs):
        return np.asarray(self).var(*vargs, **kwargs)

    def plot(self, cumulative=False):
        import seaborn as sns
        import pandas as pd
        #ax = sns.kdeplot(x)
        #sns.kdeplot(y, ax=ax)
        data = {' × '.join(ix): self[ix] for ix in self.set}
        return pd.DataFrame(data).plot(kind='hist', cumulative=cumulative)

    def __add__(self, obj):
        # for now, we're assuming fully orthogonal categories; 
        # but we might have to allow for partial or full
        # overlap (is partial overlap possible or should we
        # error on it?)
        factors = self.factors + obj.factors

        # ndim - 1 because we always leave the last dimension
        # (the MC samples) untouched
        for i in range(obj.ndim - 1):
            self = np.expand_dims(self, axis=self.ndim - 1)

        # this is to enable us to add and multiply with 1-dimensional arrays
        # dunno if we should solve this here or elsewhere in the code path
        if obj.ndim < 2:
            obj = np.expand_dims(obj, axis=1)
        
        f = np.ndarray.__add__(self, obj).view(RandomVector)
        f.factors = factors
        return f

    # TODO
    # Gamma parameters are not very informative, perhaps it's
    # better to report one or more of mean, median, sd, 
    # MAD and skew instead (using scipy.stats.describe or
    # the population ML estimates for original distributions),
    # and/or the original between arguments used to create it.
    def ___repr___(self):
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


def between(*quantiles, **alternatives):
    if quantiles:
        alternatives['default'] = quantiles

    parameters = {name: gamma.fit(interpolate(*q)) for name, q in alternatives.items()}
    distributions = {name: gamma(*p) for name, p in parameters.items()}

    return RandomVector(distributions)
