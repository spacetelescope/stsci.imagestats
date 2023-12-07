#!/usr/bin/env python
import numpy
import sys
from setuptools import Extension, find_namespace_packages, setup
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

cflags = sysconfig.get_config_var('CFLAGS')
if cflags:
    extra_compile_args = cflags.split()
    extra_compile_args += ["-Wall", "-Wextra"]
    extra_compile_args += ["-DNDEBUG", "-O2"]
else:
    extra_compile_args = None

setup(
    packages=find_namespace_packages(
        where='.',
        include=['stsci', 'stsci.imagestats']
    ),
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
