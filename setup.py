#!/usr/bin/env python
import relic.release
from glob import glob
from numpy import get_include as np_include
from setuptools import setup, find_packages, Extension


version = relic.release.get_info()
relic.release.write_template(version, 'lib/stsci/imagestats')

setup(
    name = 'stsci.imagestats',
    version = version.pep386,
    author = 'Warren Hack, Christopher Hanley',
    author_email = 'help@stsci.edu',
    description = 'Compute various useful statistical values for array objects',
    url = 'https://github.com/spacetelescope/stsci.imagestats',
    classifiers = [
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires = [
        'nose',
        'numpy',
        'sphinx',
        'stsci.sphinxext',
    ],
    package_dir = {
        '':'lib',
    },
    namespace_packages = ['stsci'],
    packages = find_packages(),
    package_data = {
        '': ['LICENSE.txt'],
    },
    ext_modules=[
        Extension('stsci.imagestats.buildHistogram',
            ['src/buildHistogram.c'],
            include_dirs=[np_include()]),
        Extension('stsci.imagestats.computeMean',
            ['src/computeMean.c'],
            include_dirs=[np_include()]),
    ],
)
