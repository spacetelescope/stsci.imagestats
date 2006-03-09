#   Program:    histogram1d.py
#   Author:     Christopher Hanley
#   Purpose:    Construct a 1 dimentional histogram from a numarray object
#
#   Version:
#       Version 0.1.0, 15-Jan-2004: Created -- CJH --
#       Version 0.1.1, 01-Apr-2004: Placed the call to the C function populate1DHist in a try/except block
#                                   for better error handling.  --  CJH
#       Version 0.1.2, 02-Nov-2005: Added 'centers' as an attribute for use
#                                   with matplotlib.  -- WJH
#                                    
#

import numarray as N
import buildHistogram

__version__ = '0.1.2'

class histogram1d:
    """Populate a 1 dimensional histogram from numarray object"""

    def __init__(self,
        arrayInput,         # 2D numarray object
        nbins,              # Number of bins in the histogram
        binWidth,           # Width of 1 bin in desired units
        zeroValue           # Zero value for the histogram range
        ):

        # Initialize Object Attributes
        self.__arrayInput = arrayInput
        self.nbins = nbins
        self.binWidth = binWidth
        self.minValue = zeroValue

        # Compute the maximum value the histogram will take on
        self.maxValue = self.minValue + (self.binWidth * self.nbins)
        
        # Compute the array of bin center values
        #   This should be done lazily using the newer-style class definition
        #   for this class.
        self.centers = N.array(self.minValue, self.maxValue, self.binWidth)

        # Allocate the memory for the histogram.
        self.histogram = N.zeros([self.nbins],type=N.UInt32)

        # Populate the histogram
        self.__populateHistogram()

    def __populateHistogram(self):
        """Call the C-code that actually populates the histogram"""
        try :
            buildHistogram.populate1DHist(self.__arrayInput, self.histogram,
                self.minValue, self.maxValue, self.binWidth)
        except:
            if ( (self.__arrayInput.max() - self.__arrayInput.min() ) < self.binWidth ):
                raise ValueError, "In histogram1d class, the binWidth is greater than the data range of the numarray object."
            else:
                raise SystemError, "An error processing the numarray object information occured in the buildHistogram module of histogram1d."
                
