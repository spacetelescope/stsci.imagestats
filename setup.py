from distutils.core import setup, Extension
import numarray
from numarray.numarrayext import NumarrayExtension
import sys

if not hasattr(sys, 'version_info') or sys.version_info < (2,2,0,'alpha',0):
    raise SystemExit, "Python 2.2 or later required to build imagestats."

def dolocal():
    """Adds a command line option --local=<install-dir> which is an abbreviation for
    'put all of imagestats in <install-dir>/imagestats'."""
    if "--help" in sys.argv:
        print >>sys.stderr
        print >>sys.stderr, " options:"
        print >>sys.stderr, "--local=<install-dir>    same as --install-lib=<install-dir>"
    for a in sys.argv:
        if a.startswith("--local="):
            dir = a.split("=")[1]
            sys.argv.extend([
                "--install-lib="+dir,
                ])
            sys.argv.remove(a)

def getExtensions():
    ext = [NumarrayExtension('imagestats/buildHistogram',['src/buildHistogram.c'],
                             libraries = ['m']),
           NumarrayExtension('imagestats/computeMean', ['src/computeMean.c'],
                             libraries = ['m'])]
                             
    return ext


def dosetup(ext):
    r = setup(name = "imagestats",
              version = "0.2.4",
              description = "Compute desired statistics values for numarray objects",
              author = "Warren Hack, Christopher Hanley",
              author_email = "help@stsci.edu",
              license = "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
              platforms = ["Linux","Solaris","Mac OS X"],
              packages=['imagestats'],
              package_dir={'imagestats':'lib'},
              ext_modules=ext)
    return r


def main():
    args = sys.argv
    dolocal()
    ext = getExtensions()
    dosetup(ext)


if __name__ == "__main__":
    main()

