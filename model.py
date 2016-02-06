import numpy as np
from scipy.stats.distributions import norm
from sklearn import linear_model

# TODO: preprocess data and include first-order interactions

a = norm.rvs(size=100)
b = norm.rvs(size=100)
y = a + 2 * b
x = np.squeeze(np.dstack((a,b)))


# TODO: use LASSO instead
model = linear_model.LinearRegression()
model.fit(x, y)
coefs = dict(zip(('a', 'b'), model.coef_))

predictors = ('a', 'b')
outcome = 'y'
means = x.mean(axis=0)


# NOTE: IIRC it's possible to use the covariance matrix
# of the full model to get parameter estimates for
# reduced models -- that might be worth looking into, 
# either for imputation or just to work with the reduced
# model as-is
# (Ideally we'd do multiple imputation / MICE, but let's
# keep things simple for now.)
# NOTE: if it makes things easier, let's just fit a 
# separate regression taking each variable as outcome in
# turn -- it's more important to get the interface right
# than to have the code be elegant and/or fast
def predict(variable, values):
    p = zip(range(len(model.coef_)), model.coef_, values)
    # replace any missing variables with their mean
    # (ideally, replace them with their conditional mean
    # given the subset of other variables)
    v = [c * (x or means[ix]) for ix, c, x in p]
    eq = sum(v)

    # TODO: put this in a distribution instead (norm or t)
    # and then pass it on to RandomVariable which will
    # generate a sample using the prediction distribution
    if variable == outcome:
        return eq
    else:
        ix = predictors.index(variable)
        beta = model.coef_[ix]
        return (values[-1] - eq) / beta


predict('y', (2, 3))
predict('a', (None, 3, 8))
