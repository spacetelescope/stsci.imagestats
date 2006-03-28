#!/usr/bin/env python

import os, os.path, sys
from distutils.core import setup
from distutils.sysconfig import *
from distutils.command.install_data import install_data

from cfg_pyraf import PYRAF_DATA_FILES, PYRAF_SCRIPTS, PYRAF_EXTENSIONS, PYRAF_CLCACHE
from cfg_pydrizzle import PYDRIZZLE_EXTENSIONS
from cfg_modules import PYFITS_MODULES, PYTOOLS_MODULES
from cfg_imagestats import IMAGESTATS_EXTENSIONS

#py_includes = get_python_inc(plat_specific=1)
py_libs =  get_python_lib(plat_specific=1)
ver = get_python_version()
pythonver = 'python' + ver

args = sys.argv[2:]
#data_dir = py_libs

PACKAGES = ['pyraf','numdisplay', 'imagestats', 'multidrizzle', 'saaclean', 'pydrizzle', 'pydrizzle.traits102', 'puftcorr']



PACKAGE_DIRS = {'pyraf':'pyraf/lib','numdisplay':'numdisplay', 'imagestats':'imagestats/lib', 'multidrizzle':'multidrizzle/lib', 'saaclean':'saaclean/lib', 'pydrizzle':'pydrizzle/lib', 'pydrizzle.traits102':'pydrizzle/traits102', 'puftcorr':'puftcorr/lib'}

PYMODULES = PYFITS_MODULES + PYTOOLS_MODULES

for a in args:
    if a.startswith('--local='):
        dir = os.path.abspath(a.split("=")[1])
	sys.argv.extend([
                "--install-lib="+dir,
                "--install-scripts=%s" % os.path.join(dir,"pyraf"),
                ])
	sys.argv.remove(a)
        args.remove(a)
    elif a.startswith('--clean_dist'):
        for f in PYMODULES:
            print "cleaning distribution ..."
            file = f + '.py'
            try:
                os.unlink(file)
            except OSError: pass
        sys.argv.remove(a)
        sys.exit(0)

    else:
        print "Invalid argument  %s", a

class smart_install_data(install_data):
    def run(self):
        #need to change self.install_dir to the library dir
        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
        return install_data.run(self)


PYRAF_DATA_DIR = os.path.join('pyraf')
PYRAF_CLCACHE_DIR = os.path.join('pyraf', 'clcache')

IMAGESTATS_DATA_DIR = os.path.join('imagestats')
IMAGESTATS_DATA_FILES = ['imagestats/lib/LICENSE.txt']

MULTIDRIZZLE_DATA_DIR = os.path.join('multidrizzle')
MULTIDRIZZLE_DATA_FILES = ['multidrizzle/lib/LICENSE.txt']

NUMDISPLAY_DATA_DIR = os.path.join('numdisplay')
NUMDISPLAY_DATA_FILES = ['numdisplay/imtoolrc', 'numdisplay/LICENSE.txt']

PUFTCORR_DATA_DIR = os.path.join('puftcorr')
PUFTCORR_DATA_FILES = ['puftcorr/lib/LICENSE.txt']

PYDRIZZLE_DATA_DIR = os.path.join('pydrizzle')
PYDRIZZLE_DATA_FILES = ['pydrizzle/lib/LICENSE.txt']

SAACLEAN_DATA_FILES = ['saaclean/lib/SP_LICENSE']
SAACLEAN_DATA_DIR = os.path.join('saaclean')


DATA_FILES = [(PYRAF_DATA_DIR, PYRAF_DATA_FILES), (PYRAF_CLCACHE_DIR, PYRAF_CLCACHE), (NUMDISPLAY_DATA_DIR, NUMDISPLAY_DATA_FILES), (SAACLEAN_DATA_DIR, SAACLEAN_DATA_FILES), (IMAGESTATS_DATA_DIR, IMAGESTATS_DATA_FILES), (MULTIDRIZZLE_DATA_DIR, MULTIDRIZZLE_DATA_FILES), (PUFTCORR_DATA_DIR,PUFTCORR_DATA_FILES), (PYDRIZZLE_DATA_DIR, PYDRIZZLE_DATA_FILES)  ]

EXTENSIONS = PYRAF_EXTENSIONS + PYDRIZZLE_EXTENSIONS + IMAGESTATS_EXTENSIONS

if sys.platform == 'win32':
    PACKAGES.remove('pyraf')
    del(PACKAGE_DIRS['pyraf'])
    PACKAGES.remove('saaclean')
    del(PACKAGE_DIRS['saaclean'])
    PACKAGES.remove('puftcorr')
    del(PACKAGE_DIRS['puftcorr'])
    #remove pyraf's data files
    DATA_FILES = [(NUMDISPLAY_DATA_DIR, NUMDISPLAY_DATA_FILES), (IMAGESTATS_DATA_DIR, IMAGESTATS_DATA_FILES), (MULTIDRIZZLE_DATA_DIR, MULTIDRIZZLE_DATA_FILES), (PYDRIZZLE_DATA_DIR, PYDRIZZLE_DATA_FILES)]

    EXTENSIONS = PYDRIZZLE_EXTENSIONS + IMAGESTATS_EXTENSIONS
    
class smart_install_data(install_data):
    def run(self):
        #need to change self.install_dir to the library dir
        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
        return install_data.run(self)

setup(name="STScI Python Software",
      version="2.2",
      description="",
      author="Science Software Branch, STScI",
      maintainer_email="help@stsci.edu",
      url="http://www.stsci.edu/resources/software_hardware/index_html?category=Data_Analysis",
      packages = PACKAGES,
      py_modules = PYMODULES,
      package_dir = PACKAGE_DIRS,
      cmdclass = {'install_data':smart_install_data},
      data_files = DATA_FILES,
      scripts = ['pyraf/lib/pyraf'],
      ext_modules = EXTENSIONS,
      )






