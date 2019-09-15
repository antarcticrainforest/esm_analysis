#!/usr/bin/env python
import os.path as osp
import re
from setuptools import setup, find_packages
import sys


def get_script_path():
    return osp.dirname(osp.realpath(sys.argv[0]))


def read(*parts):
    return open(osp.join(get_script_path(), *parts)).read()


def find_version(*parts):
    vers_file = read(*parts)
    match = re.search(r'^__version__ = "(\d+\.\d+\.\d+)"', vers_file, re.M)
    if match is not None:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(name="esm_analysis",
      version=find_version("esm_analysis", "__init__.py"),
      author="Max Planck Institute for Meteorology",
      maintainer="Martin Bergemann",
      description="Tools to read and analyse data from Earth System Models ",
      long_description=read("README.md"),
      long_description_content_type='text/markdown',
      license="BSD-3-Clause",
      packages=find_packages(),
      install_requires=[
          'cdo',
          'cloudpickle',
          'dask',
          'distributed',
          'dill',
          'f90nml',
          'h5netcdf',
          'ipywidgets',
          'ipython',
          'matplotlib',
          'numpy',
          'pandas',
          'toml',
          'tqdm',
          'seaborn',
          'xarray',
      ],
      extras_require={
       'test': [
              'pytest',
              'pytest-cov',
              'nbval',
              'h5netcdf',
              'testpath',
          ]
        },
      python_requires='>=3.5',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Data Analysis',
          'Topic :: Scientific/Engineering :: Earth Sciences',
      ]
)
