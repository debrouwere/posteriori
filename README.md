# Posteriori

Making probabilistic calculations as straightforward as `2 + 2 = 4`.

Just messing around, you might not want to use this yet.

```python
>>> from posteriori import between
>>> import numpy as np
>>> 
>>> # we expect to get between 100 and 100,000 pageviews
>>> # but on average we think we'd get about 1000 views
>>> pageviews = between(100, 1000, 100000)
>>> # what percentage of pageviews will be higher than 10,000?
>>> pageviews.sf(10000)
0.34
>>> # but there's fewer visitors on weekends
>>> weekend = between(0.3, 0.9)
>>> (pageviews * weekend).mean()
6447.69
>>> (pageviews * weekend).sf(10000)
0.21
# and we don't have to use distribution functions if we don't want to
>>> np.mean(pageviews * weekend > 10000)
0.21
```

Give Posteriori an upper and lower boundary (aim for the 5th and 95th percentile) and it will fit a statistical distribution around it. Give it a median or a couple of equally spaced percentiles in between and it will even take into account skew. Then, do any kind of calculation you want with these random variables and ask for the percentage of outcomes lower than a certain value (`cdf`), higher than that value (`sf`), the 73th percentile (`ppf(0.73)`), the median (`median()`) and so on.

Under the hood, Posteriori fits everything to a Gamma distribution and calculates outcomes using Monte Carlo simulations. It works great across a range of different scenarios, see `benchmark.txt` for benchmarks.

Inspired by Frank Krueger's [Calca][1], Douglas Hubbard's 
[How To Measure Anything][2], the [Sheffield Elicitation Framework][3] and and Ozzie Gooen's [Guesstimate][4].

[1]: http://calca.io/
[2]: http://www.amazon.com/How-Measure-Anything-Intangibles-Business-ebook/dp/B00INUYS2U/
[3]: http://www.tonyohagan.co.uk/shelf/
[4]: http://www.getguesstimate.com/
