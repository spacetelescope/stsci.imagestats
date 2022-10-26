#!/usr/bin/env python
import sys

import numpy
from setuptools import setup, Extension

# Setup C module include directories
include_dirs = [numpy.get_include()]

# Setup C module macros
define_macros = [('NUMPY', '1')]

# Handle MSVC `wcsset` redefinition
if sys.platform == 'win32':
    define_macros += [
        ('_CRT_SECURE_NO_WARNING', None),
        ('__STDC__', 1)
    ]

ext_modules = [
    Extension('stsci.imagestats.buildHistogram',
              ['src/buildHistogram.c'],
              include_dirs=include_dirs,
              define_macros=define_macros),
    Extension('stsci.imagestats.computeMean',
              ['src/computeMean.c'],
              include_dirs=include_dirs,
              define_macros=define_macros),
]

setup(
    ext_modules=ext_modules,
)
