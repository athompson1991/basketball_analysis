import numpy as np


def get_implied_probability(ml):
    ml = float(ml)
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return -ml / (-ml + 100)


def zero_to_one(x):
    return np.exp(x) / (1 + np.exp(x))


def to_infinity(x):
    return np.log(x / (1 - x))


def get_implied_probability_vec(ml_vec):
    return np.array([get_implied_probability(ml) for ml in ml_vec])
