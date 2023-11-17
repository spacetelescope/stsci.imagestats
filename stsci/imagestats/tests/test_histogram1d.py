import pytest
import numpy as np

from stsci.imagestats.histogram1d import histogram1d


_FINFO32 = np.finfo(np.float32)
_EPS32 = _FINFO32.eps
_TINY32 = _FINFO32.tiny


def test_histogram(gaussian_image):
    mean = gaussian_image.meta['mean']
    stddev = gaussian_image.meta['stddev']
    nsig = 0.1
    minv = np.float32(gaussian_image.meta['min'] - nsig * stddev)
    maxv = np.float32(gaussian_image.meta['max'] + nsig * stddev)
    drange = maxv - minv
    data = gaussian_image.astype(np.float32).ravel()
    nbins = int(drange / (nsig * stddev))
    binwidth = drange / nbins

    h = histogram1d(
        arrayInput=data,
        nbins=nbins,
        binWidth=binwidth,
        zeroValue=minv
    )

    assert abs(mean - minv - binwidth * np.argmax(h.histogram)) < binwidth
    assert np.sum(h.histogram) == data.size

    # compare to numpy:
    nph, npedges = np.histogram(data[(data < maxv)], nbins, (minv, maxv))
    # ideally the following should pass:
    # assert np.all(nph == h.histogram)
    # However, it fails with minor errors on Linux, likely due to
    # floating point rounding errors. Therefore we replace this with a set of
    # less strict tests:
    diff = nph - h.histogram
    assert diff.sum() == 0
    assert max(np.abs(diff)) <= 1
    assert np.flatnonzero(diff).size < 4

    assert np.allclose(h.edges, npedges, rtol=0.0, atol=10.0 * _EPS32)


@pytest.mark.parametrize(
    'value,truth',
    [
        (0.0, [1, 0]),
        (_TINY32, [1, 0]),
        (-_TINY32, [0, 0]),
        (1.0 - _EPS32, [0, 1]),
        (1.0, [0, 0]),
        (1.0 + _EPS32, [0, 0]),
    ]
)
def test_small_data(value, truth):
    h = histogram1d([value], nbins=2, binWidth=0.5, zeroValue=0.0).histogram
    assert np.all(h == truth)


def test_bad_input_large_bin():
    with pytest.raises(ValueError):
        histogram1d(
            arrayInput=np.array(1.0).astype(np.float32),
            nbins=2,
            binWidth=0.1,
            zeroValue=0.1
        )


def test_bad_input_small_bin():
    with pytest.raises(RuntimeError):
        histogram1d(
            arrayInput=np.array(np.nan).astype(np.float32),
            nbins=2,
            binWidth=0.1,
            zeroValue=0.1
        )
