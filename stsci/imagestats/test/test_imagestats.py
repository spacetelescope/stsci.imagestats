import numpy as np

from stsci.imagestats import ImageStats


def test_imagestats():
    mean = 1.854
    stddev = 0.333
    a = np.random.normal(loc=mean, scale=stddev, size=(100, 200))

    result = ImageStats(a)

    np.testing.assert_almost_equal(result.mean, mean, decimal=2)
    np.testing.assert_almost_equal(result.stddev, stddev, decimal=2)
