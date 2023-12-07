"""
A module that provides functionality to construct a 1-dimentional histogram
from an array object.

For help, contact `HST Help Desk <https://hsthelp.stsci.edu>`_.

"""
import numpy as np

from . import buildHistogram

__all__ = ["histogram1d"]


class histogram1d:
    """
    Populate a 1-dimensional histogram from 1D `numpy.nddata` array.

    Parameters
    ----------

    arrayInput : numpy.nddata
        2D array object

    nbins : int
        Number of bins in the histogram.

    binWidth : float
        Width of 1 bin in desired units

    zeroValue : float
        Zero value for the histogram range

    """
    def __init__(self, arrayInput, nbins, binWidth, zeroValue):
        # Initialize Object Attributes
        self._data = np.asanyarray(arrayInput, dtype=np.float32)
        self.nbins = nbins
        self.binWidth = np.float32(binWidth)
        self.minValue = np.float32(zeroValue)

        # Compute the maximum value the histogram will take on
        self.maxValue = self.minValue + (self.binWidth * nbins)
        self._centers = None
        self._edges = None

        # Allocate the memory for the histogram.
        self.histogram = np.zeros([nbins], dtype=np.uint32)

        # Populate the histogram
        try:
            buildHistogram.populate1DHist(
                self._data,
                self.histogram,
                self.minValue,
                self.maxValue,
                self.binWidth
            )
        except Exception:
            if ((self._data.max() - self._data.min()) < self.binWidth):
                raise ValueError(
                    "In histogram1d class, the binWidth is greater than the "
                    "data range of the array object."
                )
            else:
                raise RuntimeError(
                    "An error processing the array object information occured "
                    "in the buildHistogram module of histogram1d."
                )

    @property
    def edges(self):
        """ Compute the array of bin center values """
        if self._edges is None:
            self._edges = self.get_edges()
        return self._edges

    @property
    def centers(self):
        """ Compute the array of bin center values """
        if self._centers is None:
            self._centers = self.getCenters()
        return self._centers

    def get_edges(self):
        """ Returns histogram's bin edges (including left edge of the first bin
            and right edge of the last bin).
        """
        nedge = self.histogram.size + 1
        return (
            np.arange(nedge, dtype=np.float32) * self.binWidth + self.minValue
        )

    def getCenters(self):
        """ Returns histogram's centers. """
        nbins = self.histogram.size
        return (
            (0.5 + np.arange(nbins, dtype=np.float32)) * self.binWidth +
            self.minValue
        )
