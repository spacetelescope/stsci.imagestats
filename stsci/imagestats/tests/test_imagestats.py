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


def test_limits(gaussian_image):
    binwidth = 0.05
    q = 0.2
    minv = gaussian_image.meta['min']
    maxv = gaussian_image.meta['max']
    r = maxv - minv

    lower = minv + q * r
    upper = maxv - q * r

    result = ImageStats(
        gaussian_image,
        "npix,min,max,mean,midpt,median,mode,stddev",
        binwidth=binwidth,
        lower=lower,
        upper=upper,
    )
    mean = gaussian_image.meta['mean']
    stddev = gaussian_image.meta['stddev']

    n = np.sqrt(result.npix)
    atol_population = stddev / ((1.0 - 2 * q) * n)
    atol_binned = binwidth * stddev

    assert np.allclose(result.mean, mean, rtol=0, atol=atol_population)
    assert np.allclose(result.mode, mean, rtol=0, atol=atol_binned)
    assert np.allclose(result.midpt, mean, rtol=0, atol=atol_binned)
    assert result.min >= lower
    assert result.max <= upper


def test_print(gaussian_image, capsys):
    result = ImageStats(gaussian_image, "npix,min,max,mean,mode,midpt,median,stddev")
    result.printStats()
    captured = capsys.readouterr()
    assert captured.out.startswith("--- Imagestats Results ---")


def test_no_data_after_clipping():
    data = np.array([0, 10], dtype=np.float32)
    with pytest.raises(ValueError):
        ImageStats(data, nclip=1, lsig=0.1, usig=0.1)


def test_get_centers(uniform_image):
    binwidth = 0.1
    minv = uniform_image.meta['min']
    maxv = uniform_image.meta['max']
    stddev = uniform_image.meta['stddev']
    result = ImageStats(uniform_image, "midpt", nclip=0, binwidth=binwidth)
    assert (
        result.getCenters().size == int((maxv - minv) / (stddev * binwidth)) + 1
    )
    result = ImageStats(uniform_image, nclip=0, binwidth=binwidth)
    assert result.getCenters() is None


def test_invalid_args():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError):
        ImageStats(data, nclip=-1)

    with pytest.raises(ValueError):
        ImageStats(data, lsig=0.0)

    with pytest.raises(ValueError):
        ImageStats(data, usig=-1.0)

    with pytest.raises(ValueError):
        ImageStats(data, binwidth=-0.1)

    with pytest.raises(ValueError):
        ImageStats(data, lower=6.0)

    with pytest.raises(ValueError):
        ImageStats(data, upper=-1.0)

    with pytest.raises(ValueError):
        ImageStats(data, lower=4.0, upper=2.0)

    with pytest.raises(ValueError):
        ImageStats(data, lower=6.0)

    with pytest.raises(ValueError):
        ImageStats(data, upper=0.0)


def test_no_data():
    with pytest.raises(ValueError) as e:
        ImageStats([])
    assert e.value.args[0] == "Not enough data points to compute statistics."


def test_invalid_data():
    with pytest.raises(ValueError) as e:
        ImageStats([np.nan])
    assert e.value.args[0] == "Not enough data points to compute statistics."


def test_no_data_after_clip():
    with pytest.raises(ValueError) as e:
        ImageStats(
            [0.0, 0.1, 0.2],
            lower=0.05,
            upper=0.05,
            nclip=3,
            lsig=0.0001,
            usig=0.0001
        )
    assert e.value.args[0] == "Not enough data points to compute statistics."


def test_mode_2_bins():
    eps = np.finfo(np.float32).eps

    data = [0.0, 0.2]
    stddev = np.std(data) * np.sqrt(len(data) / (len(data) - 1.0))
    h = ImageStats(
        data,
        fields='mode,midpt',
        binwidth=0.1 * (1 + eps) / stddev,
    )
    assert abs(h.mode - 0.1) < 2.0 * eps
    assert abs(h.midpt - 0.1) < 2.0 * eps

    data = [0.0, 0.0, 0.2]
    stddev = np.std(data) * np.sqrt(len(data) / (len(data) - 1.0))
    h = ImageStats(
        data,
        fields='mode,midpt',
        binwidth=0.1 * (1 + eps) / stddev,
    )
    assert abs(h.mode - 0.05) < 2.0 * eps
    assert abs(h.midpt - 0.075) < 2.0 * eps

    data = [0.0, 0.2, 0.2]
    stddev = np.std(data) * np.sqrt(len(data) / (len(data) - 1.0))
    h = ImageStats(
        data,
        fields='mode,midpt',
        binwidth=0.1 * (1 + eps) / stddev,
    )
    assert abs(h.mode - 0.15) < 2.0 * eps
    assert abs(h.midpt - 0.125) < 2.0 * eps


def test_mode_at_edges():
    eps = np.finfo(np.float32).eps

    data = [0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2]
    stddev = np.std(data) * np.sqrt(len(data) / (len(data) - 1.0))
    h = ImageStats(
        data,
        fields='mode,midpt',
        binwidth=0.05 * (1 + eps) / stddev,
    )
    assert abs(h.mode - 0.025) < 2.0 * eps
    assert abs(h.mode - 0.025) < 2.0 * eps

    data = [0.0, 0.05, 0.1, 0.15, 0.19, 0.19, 0.19]
    stddev = np.std(data) * np.sqrt(len(data) / (len(data) - 1.0))
    h = ImageStats(
        data,
        fields='mode,midpt',
        binwidth=0.045 * (1 + eps) / stddev,
    )
    assert abs(h.mode - 0.2025) < 2.0 * eps
    assert abs(h.midpt - 0.1575) < 2.0 * eps


@pytest.mark.skip(reason="improve detection of _peakindex for mode - see TODO")
def test_mode_uniform():
    eps = np.finfo(np.float32).eps

    data = [0.0, 0.08, 0.08, 0.08, 0.13, 0.13, 0.13, 0.19, 0.19, 0.19, 0.20001]
    stddev = np.std(data) * np.sqrt(len(data) / (len(data) - 1.0))
    h = ImageStats(
        data,
        fields='mode,midpt',
        binwidth=0.05 * (1 + eps) / stddev,
    )
    assert abs(h.mode - 0.075) < 2.0 * eps


def test_large_bin(gaussian_image):
    stddev = gaussian_image.meta['stddev']
    minv = gaussian_image.meta['min']
    maxv = gaussian_image.meta['max']

    binwidth = 2.0 * (maxv - minv) / stddev
    ImageStats(
        gaussian_image,
        "midpt,median,mode",
        binwidth=binwidth,
    )


def test_computemean_exception(gaussian_image):
    s = ImageStats(gaussian_image)
    s.image = np.float32(0.0)
    with pytest.raises(RuntimeError):
        s._computeStats()
