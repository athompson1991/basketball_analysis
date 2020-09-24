import numpy as np
from pandas import Series


def get_implied_probability(ml):
    fn = lambda x: 100 / (x + 100) if x > 0 else -x / (-x + 100)
    if isinstance(ml, Series):
        ml = ml.astype(float)
        return ml.apply(fn)
    elif isinstance(ml, np.ndarray):
        ml = ml.astype(float)
        return np.vectorize(fn)(ml)
    elif isinstance(ml, list):
        ml = [float(l) for l in ml]
        return np.vectorize(fn)(ml)
    else:
        ml = float(ml)
        return fn(ml)


def zero_to_one(x):
    return np.exp(x) / (1 + np.exp(x))


def to_infinity(x):
    return np.log(x / (1 - x))

