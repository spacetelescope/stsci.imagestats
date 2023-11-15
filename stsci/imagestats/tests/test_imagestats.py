import pytest
import numpy as np

from stsci.imagestats import ImageStats


def test_imagestats():
    mean = 1.854
    stddev = 0.333
    a = np.random.normal(loc=mean, scale=stddev, size=(100, 200))

    result = ImageStats(a)

    np.testing.assert_almost_equal(result.mean, mean, decimal=2)
    np.testing.assert_almost_equal(result.stddev, stddev, decimal=2)


def test_gaussian(gaussian_image):
    binwidth = 0.05
    result1 = ImageStats(
        gaussian_image,
        "npix,min,max,mean,midpt,median,mode,stddev",
        binwidth=binwidth,
    )
    result2 = ImageStats(gaussian_image.astype(np.float32))
    mean = gaussian_image.meta['mean']
    stddev = gaussian_image.meta['stddev']

    atol = 10 * np.finfo(np.float32).eps
    n = np.sqrt(gaussian_image.size)
    atol_population = stddev / n
    atol_binned = binwidth * stddev

    assert np.allclose(result1.mean, result2.mean, rtol=0, atol=atol)

    assert np.allclose(result1.mean, mean, rtol=0, atol=atol_population)
    assert np.allclose(result1.mode, mean, rtol=0, atol=atol_binned)
    assert np.allclose(result1.midpt, mean, rtol=0, atol=atol_binned)
    assert np.allclose(result1.stddev, stddev, rtol=0.01, atol=0.0)


def test_uniform(uniform_image):
    result1 = ImageStats(
        uniform_image,
        "npix,min,max,mean,midpt,median,stddev",
    )
    result2 = ImageStats(uniform_image.astype(np.float32))
    mean = uniform_image.meta['mean']
    stddev = uniform_image.meta['stddev']

    atol = 10 * np.finfo(np.float32).eps
    n = np.sqrt(uniform_image.size)
    a = uniform_image.meta['max'] - uniform_image.meta['min']
    atol_population = 10 * a * np.sqrt(n / ((n + 1)**2 * (n + 2)))

    assert np.allclose(result1.mean, result2.mean, rtol=0, atol=atol)

    assert np.allclose(result1.mean, mean, rtol=0, atol=atol_population)
    assert np.allclose(result1.median, mean, rtol=0, atol=atol_population)
    assert np.allclose(result1.midpt, mean, rtol=0, atol=atol_population)

    assert np.allclose(result1.stddev, stddev, rtol=0.01, atol=0.0)


def test_gaussian_clipping(gaussian_image):
    binwidth = 0.05
    result = ImageStats(
        gaussian_image,
        "npix,min,max,mean,midpt,median,mode,stddev",
        binwidth=binwidth,
        nclip=5,
    )
    mean = gaussian_image.meta['mean']
    stddev = gaussian_image.meta['stddev']

    atol_binned = binwidth * result.stddev

    assert np.allclose(result.mean, mean, rtol=0, atol=atol_binned)
    assert np.allclose(result.mode, mean, rtol=0, atol=atol_binned)
    assert np.allclose(result.median, mean, rtol=0, atol=atol_binned)
    assert np.allclose(result.midpt, mean, rtol=0, atol=atol_binned)
    assert result.stddev < stddev


@pytest.mark.parametrize('nclip', [0, 1, 5])
def test_all_values_in_one_bin(constant_image, nclip, capsys):
    result = ImageStats(
        constant_image,
        "npix,min,max,mean,median,mode,midpt,stddev",
        nclip=nclip,
    )
    captured = capsys.readouterr()
    assert captured.out == "! WARNING: Clipped data falls within 1 histogram bin\n"

    mean = constant_image.meta['mean']
    stddev = constant_image.meta['stddev']

    atol = 10 * np.finfo(np.float32).eps

    assert np.allclose(result.mean, mean, rtol=0, atol=atol)
    assert np.allclose(result.mode, mean, rtol=0, atol=atol)
    assert np.allclose(result.midpt, mean, rtol=0, atol=atol)
    assert np.allclose(result.median, mean, rtol=0, atol=atol)
    assert np.allclose(result.stddev, stddev, rtol=0, atol=atol)
