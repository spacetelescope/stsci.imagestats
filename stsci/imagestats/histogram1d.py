"""
A module that provides functionality to construct a 1-dimentional histogram
from an array object.

:Author: Christopher Hanley (for help, contact `HST Help Desk <https://hsthelp.stsci.edu>`_)

:License: :doc:`../LICENSE`

"""
from __future__ import division # confidence high

import numpy as np
from . import buildHistogram
from .version import *

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
        self._data = arrayInput.astype(np.float32)
        self.nbins = nbins
        self.binWidth = binWidth
        self.minValue = zeroValue

        # Compute the maximum value the histogram will take on
        self.maxValue = self.minValue + (binWidth * nbins)

        # Compute the array of bin center values
        #   This should be done lazily using the newer-style class definition
        #   for this class.
        self.centers = np.array([self.minValue, self.maxValue, binWidth])

        # Allocate the memory for the histogram.
        self.histogram = np.zeros([nbins], dtype=np.uint32)

        # Populate the histogram
        self._populateHistogram()

    def _populateHistogram(self):
        """Call the C-code that actually populates the histogram"""
        try :
            buildHistogram.populate1DHist(self._data, self.histogram,
                self.minValue, self.maxValue, self.binWidth)
        except:
            if ((self._data.max() - self._data.min()) < self.binWidth):
                raise ValueError("In histogram1d class, the binWidth is "
                                 "greater than the data range of the array "
                                 "object.")
            else:
                raise SystemError("An error processing the array object "
                                  "information occured in the buildHistogram "
                                  "module of histogram1d.")

    def getCenters(self):
        """ Returns histogram's centers. """
        return np.arange(self.histogram.size) * self.binWidth + self.minValue
