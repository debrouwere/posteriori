# NOTE: it might be a good idea to extract this into its own Python package; 
# there's so many different situations where I could have used this in
# the past.

# NOTE: the statistics module was added in Python 3.4, which makes 3.4
# the minimum requirement for now (if we do put this into its own package, 
# we could provide mean and median functions as part of the package)

from utils import replace, bound, round
from math import log, floor
from statistics import mean, median
from decimal import Decimal, Context


def e(exponent):
    value = str(abs(exponent))
    if exponent < 0:
        return 'E-' + value
    elif exponent > 0:
        return 'E+' + value
    else:
        return ''


# TODO: just have this be an array and then zip this with 
# range(-len(SI)//2*3, len(SI)//2*3, 3)
SI = {
     12: 'T',
      9: 'G',
      6: 'M',
      3: 'K',
      0: '',
     -3: 'm',
     -6: 'µ',
     -9: 'n',
    -12: 'p',
}


def E(value, precision=3, prefix=True, prefixes=SI):
    """ Convert a number to engineering notation. """

    display = Context(prec=precision)
    string = Decimal(value).normalize(context=display).to_eng_string()

    if prefix:
        prefixes = {e(exponent): prefix for exponent, prefix in prefixes.items()}
        return replace(string, prefixes)
    else:
        return string


# TODO: add a thousands separator
def B(values, precision=3, prefix=True, prefixes=SI, separator=' ', statistic=median):
    """
    Convert a list of numbers to the engineering notation appropriate to a 
    reference point like the minimum, the median or the mean -- 
    think of it as "business notation".

    Any number will have at most the amount of significant digits of the 
    reference point, that is, the function will round beyond the 
    decimal point.

    For example, if the reference is `233K`, this function will turn turn 
    1,175,125 into `1180K` and 11,234 into `11K` (instead of 1175K and
    11.2K respectively.) This can help enormously with readability.

    If the reference point is equal to or larger than E15 or
    equal to or smaller than E-15, E12 and E-12 become the
    reference point instead. (Petas and femtos are too
    unfamiliar to people to be easily comprehended.)
    """

    exponent = floor(log(statistic(values), 10))
    d = precision - exponent % precision - 1
    e = bound(exponent - exponent % precision, -12, 12)
    prefix = prefixes[e]

    strings = []
    for value in values:
        over = floor(log(value, 10)) - exponent
        decimals = min(d - over, d)
        strings.append(str(round(value / 10 ** e, decimals)) + prefix)
    
    return strings

"""
Examples: 

    E(1111.23)
    a = [111, 1111.23, 1175125.234]
    b = [11234.22, 233000.55, 1175125.2]
    B(a)
    B(b)

"""