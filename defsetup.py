import sys
import distutils
import distutils.core


if not hasattr(sys, 'version_info') or sys.version_info < (2,3,0,'alpha',0):
    raise SystemExit, "Python 2.3 or later required to build imagestats."

try:
    import numpy
except:
    raise ImportError("NUMPY was not found. It may not be installed or it may not be on your PYTHONPATH")

if sys.platform != 'win32':
    imagestats_libraries = ['m']
else:
    imagestats_libraries = ['']


pythoninc = distutils.sysconfig.get_python_inc()
numpyinc = numpy.get_include()

ext = [ distutils.core.Extension('imagestats.buildHistogram',['src/buildHistogram.c'],
                 include_dirs = [pythoninc,numpyinc],
                 libraries = imagestats_libraries),
         distutils.core.Extension('imagestats.computeMean', ['src/computeMean.c'],
                 include_dirs = [pythoninc,numpyinc],
                 libraries = imagestats_libraries)
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
