import pandas as pd
import numpy as np
from scipy.stats.distributions import expon, gamma, rayleigh, norm, t, uniform

from posteriori import between


def RMSE(predicted, expected):
    return np.linalg.norm(predicted - expected) / np.sqrt(len(predicted))

distributions = [
    norm(),
    t(df=5),
    gamma(a=2),
    gamma(a=4),
    gamma(a=8),
    expon(scale=1/0.5),
    expon(scale=1/1),
    expon(scale=1/2),
    rayleigh(),
    uniform(),
]

errors = []

for distribution in distributions:
    parameters = [k + '=' + str(v) for k, v in distribution.kwds.items()]
    name = "{name}({parameters})".format(
        name=distribution.dist.name,
        parameters=', '.join(parameters)
        )
    l, lm, lt, m, ut, um, u = distribution.ppf([0.05, 0.2625, 0.342, 0.5, 0.658, 0.7375, 0.95])
    candidates = [
        between(l, u),
        between(l, m, u),
        between(l, lt, ut, u),
        between(l, lm, m, um, u),
        ]

    percentiles = np.linspace(0.01, 0.99, num=99)
    expected = distribution.ppf(percentiles)

    for ix, candidate in enumerate(candidates):
        points = ix + 2
        observed = candidate.ppf(percentiles)
        error = RMSE(observed, expected)
        errors.append([name, points, round(error, 2)])

errors = pd.DataFrame(errors, columns=('distribution', 'points', 'RMSE'))
print(errors)
