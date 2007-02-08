# PROGRAM: imagestats.py
# AUTHOR:  Warren Hack and Christopher Hanley
# PURPOSE: Compute desired statistics values for input numarray objects.
#
#
# Version: 0.1.0 -- 17-Sep-2003: Created  -- CHanley
# Version: 0.1.1 -- 14-Oct-2003: Modified to use 'ravel' efficiently. -- WJH
# Version: 0.1.2 -- 04-Nov-2003: Added histogram as return value, and added
#                                'getCenters' method to return histogram bin
#                                 centers as an array. -- WJH
# Version: 0.2.0 -- 17-Nov-2003: Modified imagestats to compute only the minimal
#                                 set of requested statistical values. -- CJH
# Version: 0.2.1 -- 20-Nov-2003: Now raises an Exception for cases where no pixels
#                                 are found within clipping region. -- WJH
# Version: 0.2.2 -- 15-Jan-2004: Improved performance of the mode and median
#                                 computation by using the histogram1d class to build
#                                 the histogram instead of the current Python code.  --CJH
# Version: 0.2.3 -- 23-Feb-2004: C-API for computation of mean, stddev, min, and max. -- CJH
# Version: 0.2.4 -- 01-Apr-2004: Placed computeMEAN in a try except block for better error handling.
#                                Also, before building a histogram, we make sure that the entire 
#                                data range doesn't fit into a single bin.  If that condition exists,
#                                an exception is raised. -- CJH
# Version: 0.2.5 -- 19-Jul-2004: Added code to ensure that the hwidth in the histogram is always 
#                                equal to self.binwidth -- CJH
# Version: 1.0.0 -- 01-Jun-2005: Added an error condition to the clipping loop to throw an exception if
#                                the number of pixels in the region of interest is equal to 0. -- CJH
import numpy as N
from histogram1d import histogram1d
import time
from computeMean import computeMean

__version__ = '1.1.0'

class ImageStats:
    """ Class to compute desired statistics from numarray objects."""

    def __init__(self,
                image,
                fields="npix,min,max,mean,stddev",
                lower=None,
                upper=None,
                nclip=0,
                lsig=3.0,
                usig=3.0,
                binwidth=0.1
                ):

        #Initialize the start time of the program
        self.startTime = time.time()

        # Input Value
        self.image = image
        self.lower = lower
        self.upper = upper
        self.nclip = nclip
        self.lsig = lsig
        self.usig = usig
        self.binwidth = binwidth
        self.fields = fields.lower()

        # Initialize some return value attributes
        self.npix = None
        self.stddev = None
        self.mean = None
        self.mode = None
        self.bins = None
        self.median = None

        # Compute Global minimum and maximum
        self.min = N.minimum.reduce(N.ravel(image))
        self.max = N.maximum.reduce(N.ravel(image))

        # Apply initial mask to data: upper and lower limits
        if self.lower == None:
            self.lower = self.min

        if self.upper == None:
            self.upper = self.max

        # Compute the image statistics
        self._computeStats()

        # Initialize the end time of the program
        self.stopTime = time.time()
        self.deltaTime = self.stopTime - self.startTime

    def _computeStats(self):
        """ Compute all the basic statistics from the numarray object. """

        # Initialize the local max and min
        _clipmin = self.lower
        _clipmax = self.upper

        # Compute the clipped mean iterating the user specified numer of iterations
        for iter in xrange(self.nclip+1):

            try:
                _npix,_mean,_stddev,_min,_max = computeMean(self.image,_clipmin,_clipmax)
                #print "_npix,_mean,_stddev,_min,_max = ",_npix,_mean,_stddev,_min,_max 
            except:
                raise SystemError, "An error processing the numarray object information occured in the computeMean module of imagestats."
            
            if _npix <= 0:
                # Compute Global minimum and maximum
                errormsg =  "\n##############################################\n"
                errormsg += "#                                            #\n"
                errormsg += "# ERROR:                                     #\n"
                errormsg += "#  Unable to compute image statistics.  No   #\n"
                errormsg += "#  valid pixels exist within the defined     #\n"
                errormsg += "#  pixel value range.                        #\n"
                errormsg += "#                                            #\n"
                errormsg += "  Image MIN pixel value: " + str(self.min) + '\n'
                errormsg += "  Image MAX pixel value: " + str(self.max) + '\n\n'
                errormsg += "# Current Clipping Range                     #\n"
                errormsg += "       for iteration " + str(iter) + '\n' 
                errormsg += "       Excluding pixel values above: " + str(_clipmax) + '\n'
                errormsg += "       Excluding pixel values below: " + str(_clipmin) + '\n'
                errormsg += "#                                            #\n"
                errormsg += "##############################################\n"
                print errormsg
                raise ValueError

            if iter < self.nclip:
                # Re-compute limits for iterations
                _clipmin = _mean - self.lsig * _stddev
                _clipmax = _mean + self.usig * _stddev

        if ( (self.fields.find('mode') != -1) or (self.fields.find('median') != -1) ):
            # Populate the historgram
            _hwidth = self.binwidth * _stddev
            
            # Special Case:  We never want the _hwidth to be smaller than the bin width.  If it is,
            # we set the hwidth to be equal to the binwidth.
            if _hwidth < self.binwidth:
                _hwidth = self.binwidth
            
            _nbins = int( (_max - _min) / _hwidth ) + 1
            _dz = float(_nbins - 1) / max(self.binwidth,float(_max - _min))
            if (_dz == 0):
                print "! WARNING: Clipped data falls within 1 histogram bin"
                _dz = 1 / self.binwidth
            _hist = histogram1d(self.image,_nbins,1/_dz,_min)
            _bins = _hist.histogram

            if (self.fields.find('mode') != -1):
                # Compute the mode, taking into account special cases
                if _nbins == 1:
                    _mode = _min + 0.5 *_hwidth
                elif _nbins == 2:
                    if _bins[0] > _bins[1]:
                        _mode = _min + 0.5 *_hwidth
                    elif _bins[0] < _bins[1]:
                        _mode = _min + 1.5 *_hwidth
                    else:
                        _mode = _min + _hwidth
                else:
                    _peakindex = N.where(_bins == N.maximum.reduce(_bins))[0].tolist()[0]

                    if _peakindex == 0:
                        _mode = _min + 0.5 * _hwidth
                    elif _peakindex == (_nbins - 1):
                        _mode = _min + (_nbins - 0.5) * _hwidth
                    else:
                        _dh1 = _bins[_peakindex] - _bins[_peakindex - 1]
                        _dh2 = _bins[_peakindex] - _bins[_peakindex + 1]
                        _denom = _dh1 + _dh2
                        if _denom == 0:
                            _mode = _min + (_peakindex + 0.5) * _hwidth
                        else:
                            _mode = _peakindex + 1 + (0.5 * (long(_dh1) - long(_dh2))/_denom)
                            _mode = _min + ((_mode - 0.5) * _hwidth)
                # Return the mode
                self.mode = _mode

            if (self.fields.find('median') != -1):
                # Compute Median Value
                _binSum = N.cumsum(_bins).astype(N.float32)
                _binSum = _binSum/_binSum[-1]
                _lo = N.where(_binSum >= 0.5)[0][0]
                _hi = _lo + 1

                _h1 = _min + _lo * _hwidth
                if (_lo == 0):
                    _hdiff = _binSum[_hi-1]
                else:
                    _hdiff = _binSum[_hi-1] - _binSum[_lo-1]

                if (_hdiff == 0):
                    _median = _h1
                elif (_lo == 0):
                    _median = _h1 + 0.5 / _hdiff * _hwidth
                else:
                    _median = _h1 + (0.5 - _binSum[_lo-1])/_hdiff * _hwidth
                self.median = _median


            # These values will only be returned if the histogram is computed.
            self.hmin = _min + 0.5 * _hwidth
            self.hwidth = _hwidth
            self.histogram  = _bins

        #Return values
        self.stddev = _stddev
        self.mean = _mean
        self.npix = _npix
        self.min = _min
        self.max = _max

    def getCenters(self):
        """ Compute the array of bin center positions."""
        return N.arange(len(self.histogram)) * self.hwidth + self.hmin

    def printStats(self):
        """ Print the requested statistics values. """
        print "--- Imagestats Results ---"

        if (self.fields.find('npix') != -1 ):
            print "Number of pixels  :  ",self.npix
        if (self.fields.find('min') != -1 ):
            print "Minimum value     :  ",self.min
        if (self.fields.find('max') != -1 ):
            print "Maximum value     :  ",self.max
        if (self.fields.find('stddev') != -1 ):
            print "Standard Deviation:  ",self.stddev
        if (self.fields.find('mean') != -1 ):
            print "Mean              :  ",self.mean
        if (self.fields.find('mode') != -1 ):
            print "Mode              :  ",self.mode
        if (self.fields.find('median') != -1 ):
            print "Median            :  ",self.median
