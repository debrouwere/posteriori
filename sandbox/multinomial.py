"""
Not sure yet how to integrate categorical distributions

- naive Bayes (1/2 chance of raining, 1/2 chance of me going out, 1/4 chance of getting wet)
- different distributions conditional on a category (indicators)
  e.g. x * categories.eq('label') + y * categories.eq('other')

binomial is easier, because true/false naturally has that link to 0 and 1
but again this is mainly a small sample vs. large sample thing, whereas
the normal distribution is about intrinsic variability, 
e.g. if you believe one product is vastly more profitable, but there's a 
huge range of uncertainty, then you'd model that with different normal
distributions for each product, not with a multinomial, which can easily
be encoded as cat_a * dist_a + cat_b * dist_b + ... and needs no sugar
"""

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
