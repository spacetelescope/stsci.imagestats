from distutils.core import setup, Extension
from distutils import sysconfig
import sys, os.path

if not hasattr(sys, 'version_info') or sys.version_info < (2,2,0,'alpha',0):
    raise SystemExit, "Python 2.2 or later required to build imagestats."

#pythonlib = sysconfig.get_python_lib(plat_specific=1)
pythoninc = sysconfig.get_python_inc()
try:
    import numarray
    from numarray.numarrayext import NumarrayExtension
except:
    raise ImportError("Numarray was not found. It may not be installed or it may not be on your PYTHONPATH\n")

if numarray.__version__ < "1.1":
    raise SystemExit, "Numarray 1.1 or later required to build imagestats."

if sys.platform != 'win32':
    imagestats_libraries = ['m']
else:
    imagestats_libraries = ['']

args = sys.argv[:]

for a in args:
    if a.startswith('--local='):
        dir = os.path.abspath(a.split("=")[1])
        sys.argv.append('--install-lib=%s' % dir)
        #remove --local from both sys.argv and args
        args.remove(a)
        sys.argv.remove(a)


def getExtensions(args):
    ext = [NumarrayExtension('imagestats.buildHistogram',['src/buildHistogram.c'],
                             include_dirs = [pythoninc],
                             libraries = imagestats_libraries),
           NumarrayExtension('imagestats.computeMean', ['src/computeMean.c'],
                             include_dirs = [pythoninc],
                             libraries = imagestats_libraries)]

    return ext


def dosetup(ext):
    r = setup(name = "imagestats",
              version = "0.2.5",
              description = "Compute desired statistics values for numarray objects",
              author = "Warren Hack, Christopher Hanley",
              author_email = "help@stsci.edu",
              license = "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
              platforms = ["Linux","Solaris","Mac OS X", "Windows"],
              packages=['imagestats'],
              package_dir={'imagestats':'lib'},
              ext_modules=ext)
    return r


if __name__ == "__main__":
    ext = getExtensions(args)
    dosetup(ext)


