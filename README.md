# Posteriori

Making probabilistic calculations as straightforward as `2 + 2 = 4`. Just messing around,
you probably don't want to use this.

```python
>>> from posteriori import between
>>> import numpy as np
>>> 
>>> # we expect to get between 100 and 100,000 pageviews
>>> # but on average we think we'd get about 1000 views
>>> pageviews = between(100, 1000, 100000)
>>> # what percentage of pageviews will be higher than 10,000?
>>> pageviews.sf(10000)
0.41
>>> # but there's fewer visitors on weekends
>>> weekend = between(0.3, 0.9)
>>> (pageviews * weekend).mean()
7873.99
>>> (pageviews * weekend).sf(10000)
0.26
# and we don't have to use distribution functions if we don't want to
>>> np.mean(pageviews * weekend > 10000)
0.26
```

Give Posteriori an upper and lower boundary (aim for the 5th and 95th percentile) and it will fit a statistical distribution around it. Give it a median or a couple of equally spaced percentiles in between and it will even take into account skew. Then, do any kind of calculation you want with these random variables and ask for the percentage of outcomes lower than a certain value (`cdf`), higher than that value (`sf`), the 73th percentile (`ppf(0.73)`), the median (`median()`) and so on.

Under the hood, Posteriori fits everything to a Gamma distribution and calculates outcomes using Monte Carlo simulations. It works great across a range of different scenarios, see `error.txt` for benchmarks.

Inspired by Frank Krueger's [Calca](http://calca.io/), Douglas Hubbard's 
[How To Measure Anything](http://www.amazon.com/How-Measure-Anything-Intangibles-Business-ebook/dp/B00INUYS2U/)
and Ozzie Gooen's [Guesstimate](http://www.getguesstimate.com/).