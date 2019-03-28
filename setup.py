#!/usr/bin/env python
import numpy
import pkgutil
import sys
from os import path, listdir
from subprocess import check_call, CalledProcessError
from configparser import ConfigParser
from setuptools import setup, find_packages, Extension
from setuptools.command.test import test as TestCommand

# get some config values

conf = ConfigParser()
conf.read(['setup.cfg'])
metadata = dict(conf.items('metadata'))
PACKAGENAME = metadata.get('name', 'stsci.imagestats')
DESCRIPTION = metadata.get('description',
                           'Compute sigma-clipped statistics on data arrays')
LONG_DESCRIPTION = metadata.get('long_description', 'README.rst')
LONG_DESCRIPTION_CONTENT_TYPE = metadata.get('long_description_content_type',
                                             'text/x-rst')
AUTHOR = metadata.get('author', 'Warren Hack, Christopher Hanley')
AUTHOR_EMAIL = metadata.get('author_email', 'help@stsci.edu')
URL = metadata.get('url', 'https://www.stsci.edu/')
LICENSE = metadata.get('license', 'BSD-3-Clause')

# load long description
this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, LONG_DESCRIPTION), encoding='utf-8') as f:
    long_description = f.read()

if not pkgutil.find_loader('relic'):
    relic_local = path.exists('relic')
    relic_submodule = (relic_local and
                       path.exists('.gitmodules') and
                       not listdir('relic'))

    try:
        if relic_submodule:
            check_call(['git', 'submodule', 'update', '--init', '--recursive'])
        elif not relic_local:
            check_call(['git', 'clone', 'https://github.com/spacetelescope/relic.git'])
        sys.path.insert(1, 'relic')

    except CalledProcessError as e:
        print(e)
        exit(1)

import relic.release

version = relic.release.get_info()
if not version.date:
    default_version = metadata.get('version', '')
    default_version_date = metadata.get('version_date', '')
    version = relic.git.GitVersion(
        pep386=default_version,
        short=default_version,
        long=default_version,
        date=default_version_date,
        dirty=True,
        commit='',
        post='-1'
    )
relic.release.write_template(version,  path.join(*PACKAGENAME.split('.')))


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['tests']
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


# Install packages required for this setup to proceed:
INSTALL_REQUIRES = ['numpy>=1.13']

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

PACKAGE_DATA = {'': ['README.md', 'LICENSE.txt']}

setup(
    name=PACKAGENAME,
    version=version.pep386,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    url=URL,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Development Status :: 7 - Inactive',
    ],
    python_requires='>=3.5',
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    package_data=PACKAGE_DATA,
    ext_modules=[
        Extension('stsci.imagestats.buildHistogram',
                  ['src/buildHistogram.c'],
                  include_dirs=include_dirs,
                  define_macros=define_macros),
        Extension('stsci.imagestats.computeMean',
                  ['src/computeMean.c'],
                  include_dirs=include_dirs,
                  define_macros=define_macros),
    ],
    cmdclass={
        'test': PyTest,
    },
    project_urls={
        'Bug Reports': 'https://github.com/spacetelescope/stsci.imagestats/issues/',
        'Source': 'https://github.com/spacetelescope/stsci.imagestats/',
        'Help': 'https://hsthelp.stsci.edu/',
    },
)
