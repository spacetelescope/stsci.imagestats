from distutils.core import setup, Extension
import numarray
from numarray.numarrayext import NumarrayExtension
import sys

if not hasattr(sys, 'version_info') or sys.version_info < (2,2,0,'alpha',0):
    raise SystemExit, "Python 2.2 or later required to build numarray."

setup(name = "buildHistogram",
      version = "0.1",
      description = "",
      packages=[""],
      package_dir={"":""},
      ext_modules=[NumarrayExtension("buildHistogram",['buildHistogram.c'],\
                   include_dirs=["./"],
                   library_dirs=["./"],
                   libraries=['m'])]
      )
setup(name = "computeMean",
      version = "0.1",
      description = "",
      packages=[""],
      package_dir={"":""},
      ext_modules=[NumarrayExtension("computeMean",['computeMean.c'],\
                   include_dirs=["./"],
                   library_dirs=["./"],
                   libraries=['m'])]
      )
