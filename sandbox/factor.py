from itertools import repeat
import numpy as np


def first(l):
    for ix, el in enumerate(l):
        if el is not None:
            return el, ix


class Factor(np.ndarray):
    def __new__(cls, data, categories):
        obj = np.array(data).view(cls)
        obj.factors = [categories]
        obj.index()     
        return obj

    def index(self):
        self.categories = {}
        for dim, categories in enumerate(self.factors):
            for row, category in enumerate(categories):
                ix = list(repeat(None, self.ndim - 1))
                ix[dim] = row
                self.categories[category] = ix

    def __getitem__(self, name):
        if isinstance(name, str) and name in self.categories:
            category = self.categories[name]
            subset = self[category]
            subset = np.squeeze(subset)
            value, index = first(category)
            subset.factors = self.factors[:]
            subset.factors.pop(index)
            subset.index()
            return subset
        else:
            return super().__getitem__(name)

    def __array_finalize__(self, obj):
        self.factors = getattr(obj, 'factors', [])
        self.categories = getattr(obj, 'categories', {})

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
        
        # print('self', self.shape, self.__class__, 'obj', obj.shape, obj.__class__)
        f = np.ndarray.__add__(self, obj).view(Factor)
        f.factors = factors
        f.index()
        return f


left = Factor([[1, 2, 3, 4],[21, 22, 23, 24], [51, 52, 53, 54]], ('a', 'b', 'c'))
right = Factor([[4, 5, 6, 7], [101, 102, 103, 103]], ('x', 'y'))

# test whether categories and factors are being properly maintained
out = left + right

# test whether dimensions continue to properly expand when we keep adding or
# multiplying
left + right + Factor([1,2], ('v', 'w'))

# test whether we're actually getting the right subset, and whether this works
# when subsetting a subset (or a subset of a subset of a subset)
out['b']
out['b']['x']

# for constants, we'd like to be able to do something like
# res['a'] + 55
# or 
# res + between(a=55, b=100, c=500)
# (where in this case each option is just a constant, but it could also be
# N samples of a distribution)
