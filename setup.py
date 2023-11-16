#!/usr/bin/env python
import numpy
import sys
from setuptools import setup, Extension
import sysconfig

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

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-Wall", "-Wextra"]
extra_compile_args += ["-DNDEBUG", "-O2"]

setup(
    ext_modules=[
        Extension(
            'stsci.imagestats.buildHistogram',
            ['src/buildHistogram.c'],
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ),
        Extension(
            'stsci.imagestats.computeMean',
            ['src/computeMean.c'],
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ),
    ],
)
