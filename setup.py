from distutils.core import setup, Extension
import numarray
from numarray.numarrayext import NumarrayExtension
import sys, os.path, string

if not hasattr(sys, 'version_info') or sys.version_info < (2,2,0,'alpha',0):
    raise SystemExit, "Python 2.2 or later required to build imagestats."

if numarray.__version__ < "1.1":
    raise SystemExit, "Numarray 1.1 or later required to build imagestats."
ver = sys.version_info
python_exec = 'python' + str(ver[0]) + '.' + str(ver[1])

def dolocal():
    """Adds a command line option --local=<install-dir> which is an abbreviation for
    'put all of imagestats in <install-dir>/imagestats'."""
    if "--help" in sys.argv:
        print >>sys.stderr
        print >>sys.stderr, " options:"
        print >>sys.stderr, "--local=<install-dir>    same as --install-lib=<install-dir>"
    for a in sys.argv:
        if a.startswith("--local="):
            dir = os.path.abspath(a.split("=")[1])
            sys.argv.extend([
                "--install-lib="+dir,
                "--install-data="+os.path.join(dir,"imagestats")
                ])
            sys.argv.remove(a)

def getExtensions(args):
    numarrayIncludeDir = './'
    for a in args:
        if a.startswith('--home='):
            numarrayIncludeDir = os.path.abspath(os.path.join(a.split('=')[1], 'include', 'python', 'numarray'))
        elif a.startswith('--prefix='):
            numarrayIncludeDir = os.path.abspath(os.path.join(a.split('=')[1], 'include','python2.3', 'numarray'))
        elif a.startswith('--local='):
            numarrayIncludeDir = os.path.abspath(a.split('=')[1])

    ext = [NumarrayExtension('imagestats/buildHistogram',['src/buildHistogram.c'],
                             include_dirs = [numarrayIncludeDir],
                             libraries = ['m']),
           NumarrayExtension('imagestats/computeMean', ['src/computeMean.c'],
                             include_dirs = [numarrayIncludeDir],
                             libraries = ['m'])]
                             
    return ext

def getDataDir(args):
    for a in args:
        if string.find(a, '--home=') == 0:
            dir = os.path.abspath(string.split(a, '=')[1])
            data_dir = os.path.join(dir, 'lib/python/imagestats')
        elif string.find(a, '--prefix=') == 0:
            dir = os.path.abspath(string.split(a, '=')[1])
            data_dir = os.path.join(dir, 'lib', python_exec, 'site-packages/imagestats')
        elif a.startswith('--install-data='):
            dir = os.path.abspath(string.split(a, '=')[1])
            data_dir = dir
        else:
            data_dir = os.path.join(sys.prefix, 'lib', python_exec, 'site-packages/imagestats')
    return data_dir


def dosetup(data_dir, ext):
    r = setup(name = "imagestats",
              version = "0.2.5",
              description = "Compute desired statistics values for numarray objects",
              author = "Warren Hack, Christopher Hanley",
              author_email = "help@stsci.edu",
              license = "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
              platforms = ["Linux","Solaris","Mac OS X"],
              packages=['imagestats'],
              package_dir={'imagestats':'lib'},
              data_files = [(data_dir,['lib/LICENSE.txt'])],
              ext_modules=ext)
    return r


def main():
    args = sys.argv
    print args
    dolocal()
    data_dir = getDataDir(args)
    ext = getExtensions(args)
    dosetup(data_dir, ext)


if __name__ == "__main__":
    main()

