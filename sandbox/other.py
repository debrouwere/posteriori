

# if we only have quantile point estimates, straight-up optimization and Monte Carlo simulation
# works best
# 
# if we do have data, we can take the sample mean and interpolate the sample quantile


# TODO: work with any number of summary statistics, using
# scipy.stats.mstats.mquantiles to interpolate to the
# quantiles we care about and assuming they are equally spaced
# between 0.975 and 0.025


quantiles = [
    # test skew
    (10, 30, 50),
    (10, 25, 50),
    (10, 15, 50),
    # test invariance to location
    (10, 25, 50),
    (0, 15, 40),
    (-10, 5, 30),
    # test invariance to scale
    (10, 25, 50),
    (0.1, 0.25, 0.5),    
    (1000, 2500, 5000),
]

N = 1000000

def delta(a, b, s):
    return round(abs(a - b) / s, 2)

# good performance in the fat tail, 
# reasonable but degraded performance in the skinny tail near the mass
# (squared skew larger than 4 translates into shape <= 1, and the gamma
# distribution degenerates into an exponential distribution which is
# discontinuous from the left)

"""
for lower, mid, upper in quantiles:
    sample = gamma(lower, mid, upper).rvs(N)
    std = scipy.std(sample)
    l, m, u = mquantiles(sample, (0.025, 0.5, 0.975))
    print(delta(lower, l, std), delta(mid, m, std), delta(upper, u, std))

# ML fitting built into scipy does much better on the lower tail and 
# mean, but not on the fat tail -- what gives? This is even when 
for lower, mid, upper in quantiles:
    fit = distributions.gamma.fit(polygon(lower, mid, upper))
    sample = distributions.gamma(*fit).rvs(N)
    std = scipy.std(sample)
    l, m, u = mquantiles(sample, (0.025, 0.5, 0.975))
    print(delta(lower, l, std), delta(mid, m, std), delta(upper, u, std), upper, u)

r = distributions.gamma(a=5, )

# Monte Carlo using numpy so we can do fast vectorized computations
N = 100000
a = gamma(0, 10, 20).rvs(N)
b = gamma(100, 150, 200).rvs(N)
print(np.mean(a * b))
print(round(100 * np.mean(a**2 / b > 10)))


# random multinomial (note: the random element really only matters if n is small,
# because then even equal probabilities can lead to very unequal outcomes, so
# maybe add an optional `n` parameter)
weights = {
    'a': 1,
    'b': 1,
    'c': 3
}
labels = weights.keys()
total = sum(weights.values())
probabilities = {key: weight/total for key, weight in weights.items()}
cumulative_probabilities = np.cumsum(list(probabilities.values()))
uniform.rvs(size=N)

@np.vectorize
def label(p):
    for ix, bound in enumerate(cumulative_probabilities):
        if p <= bound:
            return ix


# not sure yet how to integrate categorical distributions
# - naive Bayes (1/2 chance of raining, 1/2 chance of me going out, 1/4 chance of getting wet)
# - different distributions conditional on a category (indicators)
#   e.g. x * categories.eq('label') + y * categories.eq('other')
#
# binomial is easier, because true/false naturally has that link to 0 and 1
# but again this is mainly a small sample vs. large sample thing, whereas
# the normal distribution is about intrinsic variability, 
# e.g. if you believe one product is vastly more profitable, but there's a 
# huge range of uncertainty, then you'd model that with different normal
# distributions for each product, not with a multinomial, which can easily
# be encoded as cat_a * dist_a + cat_b * dist_b + ... and needs no sugar
"""