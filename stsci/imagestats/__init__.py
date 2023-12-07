"""
Compute desired statistics values for input array objects.
"""
import time

import numpy as np

from .histogram1d import histogram1d
from .computeMean import computeMean
from ._version import version as __version__  # noqa F401


__all__ = ["histogram1d", "computeMean", "ImageStats"]


class ImageStats:
    """
    Class to compute desired statistics from array objects.


    Examples
    --------
    This class can can be instantiated using the following syntax::

        >>> import stsci.imagestats as imagestats
        >>> i = imagestats.ImageStats(image,
                    fields="npix,min,max,mean,stddev",
                    nclip=3,
                    lsig=3.0,
                    usig=3.0,
                    binwidth=0.1
                    )
        >>> i.printStats()
        >>> i.mean

    The statistical quantities specified by the parameter *fields* are
    computed and printed for the input *image* array. The results are available
    as attributes of the class object as well.


    Parameters
    ----------
    image : str
        input image data array.

    fields : str
        comma-separated list of values to be computed. The following
        are the available fields:

        ======    ======
        ======    ======
        image     image data array
        npix      the number of pixels used to do the statistics
        mean      the mean of the pixel distribution
        midpt     estimate of the median of the pixel distribution
        mode      the mode of the pixel distribution
        stddev    the standard deviation of the pixel distribution
        min       the minimum pixel value
        max       the maximum pixel value
        ======    ======

        **WARNING**
            Only those fields specified upon instantiation will be computed
            and available as an output value.

    lower : float
        Lowest valid value in the input array to be used for computing the
        statistical values

    upper : float
        Largest valid value in the input array to be used in computing the
        statistical values

    nclip : int
        Number of clipping iterations to apply in computing the results

    lsig : float
        Lower sigma clipping limit (in sigma)

    usig : float
        Upper sigma clipping limit (in sigma)

    binwidth : float
        Width of bins (in sigma) to use in generating histograms for computing
        median-related values

    NOTES
    -----
    The mean, standard deviation, min and max are computed in a
    single pass through the image using the expressions listed below.
    Only the quantities selected by the fields parameter are actually computed.
    ::

            mean = sum (x1,...,xN) / N
               y = x - mean
        variance = sum (y1 ** 2,...,yN ** 2) / (N-1)
          stddev = sqrt (variance)

    The midpoint and mode are computed in two passes through the image. In the
    first pass the standard deviation of the pixels is calculated and used
    with the *binwidth* parameter to compute the resolution of the data
    histogram. The midpoint is estimated by integrating the histogram and
    computing by interpolation the data value at which exactly half the
    pixels are below that data value and half are above it. The mode is
    computed by locating the maximum of the data histogram and fitting the
    peak by parabolic interpolation.

    **Warning**
        This data will be promoted down to float32 if provided as 64-bit
        datatype.

    """
    def __init__(self, image, fields="npix,min,max,mean,stddev",
                 lower=None, upper=None, nclip=0, lsig=3.0, usig=3.0,
                 binwidth=0.1):
        # Initialize the start time of the program
        self.startTime = time.time()
        self._hist = None

        image = np.asanyarray(image)

        if image.size == 0:
            raise ValueError(
                "Not enough data points to compute statistics."
            )

        # Input Value
        if image.dtype != np.float32:
            # Warning: Input array is being downcast to a float32 array
            image = image.astype(np.float32)

        if nclip < 0:
            raise ValueError("'nclip' must be a non-negative integer.")

        if lsig <= 0.0:
            raise ValueError("'lsig' must be a positive number.")

        if usig <= 0.0:
            raise ValueError("'usig' must be a positive number.")

        if binwidth <= 0.0:
            raise ValueError("'binwidth' must be a positive number.")

        self.image = image
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
        self.median = None  # numpy computed median with clipping
        self.midpt = None  # IRAF-based pseudo-median using bins

        # Compute Global minimum and maximum
        self.min = np.minimum.reduce(np.ravel(image))
        self.max = np.maximum.reduce(np.ravel(image))

        # Apply initial mask to data: upper and lower limits
        if lower is None:
            lower = self.min

        elif lower > self.max:
            raise ValueError(
                "Lower data cutoff is larger than maximum pixel value.\n"
                "Not enough data points to compute statistics."
            )

        elif upper is not None and lower > upper:
            raise ValueError(
                "Lower data cutoff must be smaller than upper cutoff limit."
            )

        if upper is None:
            upper = self.max

        elif upper < self.min:
            raise ValueError(
                "Upper data cutoff is smaller than minimum pixel value.\n"
                "Not enough data points to compute statistics."
            )

        self.lower = lower
        self.upper = upper

        # Compute the image statistics
        self._computeStats()

        # Initialize the end time of the program
        self.stopTime = time.time()
        self.deltaTime = self.stopTime - self.startTime

    def _error_no_valid_pixels(self, clipiter, minval, maxval, minclip, maxclip):
        errormsg = "\n##############################################\n"
        errormsg += "#                                            #\n"
        errormsg += "# ERROR:                                     #\n"
        errormsg += "#  Unable to compute image statistics.  No   #\n"
        errormsg += "#  valid pixels exist within the defined     #\n"
        errormsg += "#  pixel value range.                        #\n"
        errormsg += "#                                            #\n"
        errormsg += "  Image MIN pixel value: " + str(minval) + '\n'
        errormsg += "  Image MAX pixel value: " + str(maxval) + '\n\n'
        errormsg += "# Current Clipping Range                     #\n"
        errormsg += "       for iteration " + str(clipiter) + '\n'
        errormsg += "       Excluding pixel values above: " + str(maxclip) + '\n'
        errormsg += "       Excluding pixel values below: " + str(minclip) + '\n'
        errormsg += "#                                            #\n"
        errormsg += "##############################################\n"
        return errormsg

    def _computeStats(self):
        """ Compute all the basic statistics from the array object. """

        # Initialize the local max and min
        _clipmin = self.lower
        _clipmax = self.upper

        # Compute the clipped mean iterating the user specified numer of iterations
        for iter in range(self.nclip + 1):
            try:
                _npix, _mean, _stddev, _min, _max = computeMean(
                    self.image,
                    _clipmin,
                    _clipmax
                )
            except Exception:
                raise RuntimeError(
                    "An error processing the array object information occured "
                    "in the computeMean module of imagestats."
                )

            if _npix <= 0:
                # Compute Global minimum and maximum
                errormsg = self._error_no_valid_pixels(
                    iter, self.min, self.max,
                    _clipmin, _clipmax
                )
                print(errormsg)
                raise ValueError(
                    "Not enough data points to compute statistics."
                )

            if iter < self.nclip:
                # Re-compute limits for iterations
                _clipmin = max(self.lower, _mean - self.lsig * _stddev)
                _clipmax = min(self.upper, _mean + self.usig * _stddev)

        if self.fields.find('median') != -1:
            # Use the clip range to limit the data before computing
            #  the median value using numpy
            if self.nclip > 0:
                _image = self.image[(self.image <= _clipmax) & (self.image >= _clipmin)]
            else:
                _image = self.image
            self.median = np.median(_image)
            # clean-up intermediate product since it is no longer needed
            del _image

        if ((self.fields.find('mode') != -1) or (self.fields.find('midpt') != -1)):
            # Populate the historgram
            _hwidth = self.binwidth * _stddev

            _drange = _max - _min
            _minfloatval = 10.0 * np.finfo(dtype=np.float32).eps
            if _hwidth < _minfloatval or abs(_drange) < _minfloatval or \
               _hwidth > _drange:
                _nbins = 1
                _dz = _drange
                print("! WARNING: Clipped data falls within 1 histogram bin")
            else:
                _nbins = int((_max - _min) / _hwidth) + 1
                _dz = float(_max - _min) / float(_nbins - 1)

            _hist = histogram1d(self.image, _nbins, _dz, _min)
            self._hist = _hist
            _bins = _hist.histogram

            if (self.fields.find('mode') != -1):
                # Compute the mode, taking into account special cases
                if _nbins == 1:
                    _mode = _min + 0.5 * _hwidth
                elif _nbins == 2:
                    if _bins[0] > _bins[1]:
                        _mode = _min + 0.5 * _hwidth
                    elif _bins[0] < _bins[1]:
                        _mode = _min + 1.5 * _hwidth
                    else:
                        _mode = _min + _hwidth
                else:
                    # TODO: perform a better analysis and pick the middle when
                    # there are multiple picks:
                    _peakindex = np.where(
                        _bins == np.maximum.reduce(_bins)
                    )[0].tolist()[0]
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
                            _mode = _peakindex + 1 + (
                                0.5 * (int(_dh1) - int(_dh2)) / _denom
                            )
                            _mode = _min + ((_mode - 0.5) * _hwidth)
                # Return the mode
                self.mode = _mode

            if (self.fields.find('midpt') != -1):
                # Compute a pseudo-Median Value using IRAF's algorithm
                if _bins.size > 1:
                    _binSum = np.cumsum(_bins).astype(np.float32)
                    _binSum = _binSum / _binSum[-1]
                    _lo = np.where(_binSum >= 0.5)[0][0]
                    _hi = _lo + 1

                    _h1 = _min + _lo * _hwidth
                    if (_lo == 0):
                        _hdiff = _binSum[_hi - 1]
                    else:
                        _hdiff = _binSum[_hi - 1] - _binSum[_lo - 1]

                    if (_hdiff == 0):
                        _midpt = _h1
                    elif (_lo == 0):
                        _midpt = _h1 + 0.5 / _hdiff * _hwidth
                    else:
                        _midpt = _h1 + (
                            (0.5 - _binSum[_lo - 1]) / _hdiff * _hwidth
                        )
                    self.midpt = _midpt
                else:
                    self.midpt = _bins[0]

            # These values will only be returned if the histogram is computed.
            self.hmin = _min + 0.5 * _hwidth
            self.hwidth = _hwidth
            self.histogram = _bins

        # Return values
        self.stddev = _stddev
        self.mean = _mean
        self.npix = _npix
        self.min = _min
        self.max = _max

    def getCenters(self):
        """ Compute the array of bin center positions."""
        if self._hist is not None:
            return self._hist.centers

    def printStats(self):
        """ Print the requested statistics values for those fields specified
            on input.
        """
        print("--- Imagestats Results ---")

        if (self.fields.find('npix') != -1):
            print("Number of pixels  :  ", self.npix)
        if (self.fields.find('min') != -1):
            print("Minimum value     :  ", self.min)
        if (self.fields.find('max') != -1):
            print("Maximum value     :  ", self.max)
        if (self.fields.find('stddev') != -1):
            print("Standard Deviation:  ", self.stddev)
        if (self.fields.find('mean') != -1):
            print("Mean              :  ", self.mean)
        if (self.fields.find('mode') != -1):
            print("Mode              :  ", self.mode)
        if (self.fields.find('median') != -1):
            print("Median            :  ", self.median)
        if (self.fields.find('midpt') != -1):
            print("Midpt            :  ", self.midpt)
