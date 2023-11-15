import pytest
import numpy as np

from stsci.imagestats.histogram1d import histogram1d


def test_histogram(gaussian_image):
    mean = gaussian_image.meta['mean']
    stddev = gaussian_image.meta['stddev']
    nsig = 0.1
    minv = gaussian_image.meta['min'] - nsig * stddev
    maxv = gaussian_image.meta['max'] + nsig * stddev
    drange = maxv - minv
    data = gaussian_image.astype(np.float32)
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


def test_bad_input_large_bin():
    with pytest.raises(ValueError) as e:
        histogram1d(
            arrayInput=np.array(1.0).astype(np.float32),
            nbins=2,
            binWidth=0.1,
            zeroValue=0.1
        )


def test_bad_input_small_bin():
    with pytest.raises(SystemError) as e:
        histogram1d(
            arrayInput=np.array(np.nan).astype(np.float32),
            nbins=2,
            binWidth=0.1,
            zeroValue=0.1
        )
