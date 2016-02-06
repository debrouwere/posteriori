import inspect
import textwrap

name = 'name of variable'
argument = 2.33

doc = """
      Cumulative distribution function: P(X ≤ x)

      Additional documentation goes here.
      """

doc = textwrap.dedent(doc).strip().split('\n')
fn, representation = doc[0].split(': ')
representation \
    .replace('X', name) \
    .replace('x', str(argument))

doc = """
      Interval: P(X < .) = p/2, P(X > .) = p/2

      Additional documentation goes here.
      """


def squeeze(l):
    if len(l) == 0:
        return None
    elif len(l) == 1:
        return l[0]
    else:
        return l


def label(*templates, **transformations):
    def labeler(method):
        keys = inspect.getargspec(method).args
        def generate(*vargs, **kwargs):
            context = dict(zip(keys, vargs))
            context.update(kwargs)
            for key, transformation in transformations.items():
                context[key] = transformation(*vargs, **kwargs)
            labels = [template.format(**context) for template in templates]
            return squeeze(labels)
        method.label = generate
        return method
    return labeler


@label('{a} and {b} and {c}', c=lambda a, b: a * b)
def m(a, b):
    return True

m.label(2, 4)


class Blob(object):
    def __init__(self):
        self.name = 'object name'

    @label('P({name} < .) = {p}', 'P({name} > .) = {p}', p=lambda alpha, name: round((1 - alpha)/2, 3))
    def interval(self, alpha):
        return True


obj = Blob()
obj.interval.label(0.95, name='object name')
