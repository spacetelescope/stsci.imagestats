#   Program:    histogram1d.py
#   Author:     Christopher Hanley
#   Purpose:    Construct a 1 dimentional histogram from an array object
#

from __future__ import division # confidence high

import numpy as N
from . import buildHistogram

__version__ = '1.0'

class histogram1d:
    """Populate a 1 dimensional histogram from array object"""

    def __init__(self,
        arrayInput,         # 2D array object
        nbins,              # Number of bins in the histogram
        binWidth,           # Width of 1 bin in desired units
        zeroValue           # Zero value for the histogram range
        ):

        # Initialize Object Attributes
        self.__arrayInput = arrayInput.astype(N.float32)
        self.nbins = nbins
        self.binWidth = binWidth
        self.minValue = zeroValue

        # Compute the maximum value the histogram will take on
        self.maxValue = self.minValue + (self.binWidth * self.nbins)

        # Compute the array of bin center values
        #   This should be done lazily using the newer-style class definition
        #   for this class.
        self.centers = N.array([self.minValue, self.maxValue, self.binWidth])

        # Allocate the memory for the histogram.
        self.histogram = N.zeros([self.nbins],dtype=N.uint32)

        # Populate the histogram
        self.__populateHistogram()

    def __populateHistogram(self):
        """Call the C-code that actually populates the histogram"""
        try :
            buildHistogram.populate1DHist(self.__arrayInput, self.histogram,
                self.minValue, self.maxValue, self.binWidth)
        except:
            if ( (self.__arrayInput.max() - self.__arrayInput.min() ) < self.binWidth ):
                raise ValueError("In histogram1d class, the binWidth is greater than the data \
                range of the array object.")
            else:
                raise SystemError("An error processing the array object information occured \
                in the buildHistogram module of histogram1d.")

    def getCenters(self):
        return N.arange(len(self.histogram)) * self.binWidth + self.minValue
