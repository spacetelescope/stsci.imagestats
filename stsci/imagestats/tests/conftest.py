import pytest
import numpy as np


class NDArrayMeta(np.ndarray):
    def __new__(cls, input_array, meta=None):
        obj = np.asarray(input_array).view(cls)
        obj.meta = {}
        if meta:
            obj.meta.update(meta)
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.meta = getattr(obj, 'meta', {})


@pytest.fixture(scope='module')
def gaussian_image():
    np.random.seed(1)
    gauss2d = np.random.normal(loc=np.pi, scale=0.3, size=(1024, 1024))
    meta = {
        'mean': np.pi,
        'stddev': 0.3,
        'min': gauss2d.min(),
        'max': gauss2d.max(),
    }
    return NDArrayMeta(gauss2d, meta)


@pytest.fixture(scope='module')
def uniform_image():
    np.random.seed(2)
    uniform2d = 2.0 * np.pi * np.random.random(size=(1024, 1024))
    meta = {
        'mean': np.pi,
        'stddev': np.pi * np.sqrt(1.0 / 3.0),
        'min': uniform2d.min(),
        'max': uniform2d.max(),
    }
    return NDArrayMeta(uniform2d, meta)


@pytest.fixture(scope='module')
def constant_image():
    zeros2d = np.zeros((1024, 1024))
    meta = {
        'mean': 0.0,
        'stddev': 0.0,
        'min': 0.0,
        'max': 0.0,
    }
    return NDArrayMeta(zeros2d, meta)
