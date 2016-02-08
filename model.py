import collections
from itertools import cycle, islice
from functools import partial

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.stats.distributions import norm, gamma

from posteriori import RandomVariable


N = 10000


def wrap(value):
    if isinstance(value, collections.Iterable):
        return value
    else:
        return [value]


# TODO: consider using inverse prediction so we we can reuse the same model
# to generate any (partially or fully) conditional distribution
# TODO: consider imputation (MICE-like) to ascertain conditional distributions
# of missing parameters (in lieu of their marginal distributions) when we are 
# asked to return a partially conditional / partially marginal distribution
# TODO: add interactions and possibly transformations
class Distribution(object):
    def __init__(self, data):
        self.data = data
        self.variables = {name: RandomVariable(gamma(*gamma.fit(series))) for name, series in data.iteritems()}

    def __getattr__(self, name):
        if name in self.variables:
            return partial(self.conditional, self, name)

    def marginal(self, name):
        return self.variables[name]

    def conditional(self, name, **conditions):
        # specify columns in the same order as they exist in the dataset
        # (`statsmodels` isn't always scrupulous about checking for column names)
        columns = [column for column in self.data.columns if column in conditions.keys()]
        predictors = self.data[columns]
        outcome = self.data[name]
        model = OLS(outcome, add_constant(predictors, has_constant='add'))
        fit = model.fit()
        # TODO: consider using LASSO regression
        # fit = model.fit_regularized(alpha=1)

        # reshape all conditions to N observations
        observations = {}
        for name, value in conditions.items():
            values = wrap(value)
            np.random.shuffle(values)
            observations[name] = list(islice(cycle(values), N))

        df = add_constant(pd.DataFrame(observations), has_constant='add')

        # NOTE: wls_prediction_std recalculates the prediction first before
        # calculating the standard error, so this procedure is a bit wasteful;
        # we might want to reimplement this ourselves and also memoize the
        # function (considering that in many cases conditions are constants)
        predictions = fit.predict(df)
        error = norm.rvs(size=N, loc=0, scale=wls_prediction_std(fit, df)[0])
        return (predictions + error).view(RandomVariable)


def test():
    # in reality samples are often much smaller, but we want to 
    # test whether the results align with what's expected, 
    # and that's much easier to check with large samples
    S = 1000

    from scipy.stats.distributions import norm
    a = norm.rvs(size=S)
    b = norm.rvs(size=S)
    c = norm.rvs(size=S)
    e = norm.rvs(size=S)
    y = a + 2 * b + 3 * c + e
    x = np.squeeze(np.dstack((a, b, c)))

    data = pd.DataFrame(dict(a=a, b=b, c=c, y=y))
    dist = Distribution(data)
    m = dist.marginal('a')
    m.mean()
    m.std()
    len(m)

    first = dist.conditional('y', a=[1,2,3], b=1, c=1)
    # should be 7, because y = 2 + 2 * 1 + 3 * 1
    print(first.mean())

    second = dist.conditional('y', a=1)
    print(second.mean())


test()
