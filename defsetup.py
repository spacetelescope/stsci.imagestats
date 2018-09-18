from __future__ import division # confidence high

import sys
import distutils
import distutils.core
import distutils.sysconfig

try:
    import numpy
except:
    raise ImportError("NUMPY was not found. It may not be installed or it may not be on your PYTHONPATH")

pythoninc = distutils.sysconfig.get_python_inc()
numpyinc = numpy.get_include()

ext = [ distutils.core.Extension('stsci.imagestats.buildHistogram',['src/buildHistogram.c'],
                 include_dirs = [pythoninc,numpyinc]),
         distutils.core.Extension('stsci.imagestats.computeMean', ['src/computeMean.c'],
                 include_dirs = [pythoninc,numpyinc])
    ]



pkg = "stsci.imagestats"

setupargs = {
        'version' :		"1.4.3",
        'description' :	"Compute desired statistics values for array objects",
        'author' :		"Warren Hack, Christopher Hanley",
        'author_email' :	"https://hsthelp.stsci.edu",
        'license' :		"LICENSE.txt",
        'platforms' :		["Linux","Solaris","Mac OS X", "Windows"],
        'data_files' :	[('stsci/imagestats',['LICENSE.txt'])],
        'ext_modules' :   ext,
        'package_dir' :   { 'stsci.imagestats':'lib/stsci/imagestats', },

}
