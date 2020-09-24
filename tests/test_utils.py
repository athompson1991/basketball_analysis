import numpy as np
import pandas as pd
from nose.tools import assert_equal, assert_almost_equal
from numpy.testing import assert_array_almost_equal

from core.utils import get_implied_probability, zero_to_one, to_infinity


def test_get_implied_probability():
    p = get_implied_probability(-300)
    assert_equal(0.75, p)
    p = get_implied_probability(300)
    assert_equal(0.25, p)
    p = get_implied_probability('-300')
    assert_equal(0.75, p)


def test_zero_to_one():
    x = zero_to_one(2)
    assert_almost_equal(0.88079708, x, places=7)
    x = zero_to_one(-2)
    assert_almost_equal(0.11920292, x, places=7)


def test_to_infinity():
    x = to_infinity(0.1)
    assert_almost_equal(-2.19722458, x)
    x = to_infinity(0.9)
    assert_almost_equal(2.19722458, x)


def test_implied_probability_vec():
    expected = np.array([0.52380952, 0.33333333, 0.25, 0.71428571])
    vec = np.array([-110, 200, 300, -250])
    p = get_implied_probability(vec)
    assert_array_almost_equal(expected, p)
    p = get_implied_probability(pd.Series(vec))
    assert_array_almost_equal(expected, p)
    p = get_implied_probability(list(vec))
    assert_array_almost_equal(expected, p)
    p = get_implied_probability(['-110', '200', '300', '-250'])
    assert_array_almost_equal(expected, p)
