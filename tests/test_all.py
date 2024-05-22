import pytest
from portrait import coverage, period_match
import numpy as np


def test_coverage():
    H = 4
    D = 20
    dt = 20 / 60 / 24

    times = np.hstack([np.arange(0, H / 24, dt) + i for i in np.arange(D)])

    P_max = 10
    periods = np.arange(0, P_max, 0.01)
    _ = coverage(times)(periods)


def test_period_match():
    expected = 1.234
    np.random.seed(42)
    t0s = np.random.choice(np.arange(50) * expected, size=8)
    periods = np.linspace(0.2, 5, 1000000)
    _, computed = period_match(t0s, periods)
    np.testing.assert_allclose(computed, expected, atol=1e-4)
