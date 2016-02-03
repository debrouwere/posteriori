"""
# TODO #

1. compare analytical and Monte Carlo estimates with `np.isclose`

from posteriori import between
a = between(5, 10)
b = 2 * a

print(a.cdf(7.5))
print(b.cdf(7.5))

print(a.ppf(0.5))
print(b.ppf(0.5))

print(a.isf(0.2))
print((1 * a).isf(0.2))

print(a.mean())
print(b.mean())

print(a.interval(0.9))
print((1 * a).interval(0.9))
print(b.interval(0.9))

2. do various operations on random variables to see if they work

from posteriori import between
import numpy as np

# we expect to get between 100 and 100,000 pageviews
# but on average we think we'd get about 1000 views
pageviews = between(100, 1000, 100000)
# what percentage of pageviews will be higher than 10,000?
pageviews.sf(10000)
# but there's fewer visitors on weekends
weekend = between(0.3, 0.9)
(pageviews * weekend).mean()
(pageviews * weekend).sf(10000)
# and we don't have to use distribution functions if we don't want to
np.mean(pageviews * weekend > 10000)

3. incorporate benchmark to guarantee that we don't regress on accuracy

cf. benchmark.py
"""