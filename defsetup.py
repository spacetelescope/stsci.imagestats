import sys
import distutils
import distutils.core

try:
    import numpy
except:
    raise ImportError("NUMPY was not found. It may not be installed or it may not be on your PYTHONPATH")

pythoninc = distutils.sysconfig.get_python_inc()
numpyinc = numpy.get_include()

ext = [ distutils.core.Extension('imagestats.buildHistogram',['src/buildHistogram.c'],
                 include_dirs = [pythoninc,numpyinc]),
         distutils.core.Extension('imagestats.computeMean', ['src/computeMean.c'],
                 include_dirs = [pythoninc,numpyinc])
    ]




pkg = "imagestats"

setupargs = { 
              'version' :		"1.1.0",
              'description' :	"Compute desired statistics values for array objects",
              'author' :		"Warren Hack, Christopher Hanley",
              'author_email' :	"help@stsci.edu",
              'license' :		"http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
              'platforms' :		["Linux","Solaris","Mac OS X", "Windows"],
              'data_files' :	[('imagestats',['lib/LICENSE.txt'])],
              'ext_modules' :   ext,
}
