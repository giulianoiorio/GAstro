#cython: language_level=3, boundscheck=False
from __future__ import print_function
import pip
import time



from distutils.core import  setup, Extension
from Cython.Build import cythonize
import os
import sysconfig
import sys

cy=['GAstro/cutils.pyx']
cy_ext=Extension('GAstro.cutils',sources=cy)

ext_modules=cythonize(cy_ext)

setup(
		name='GAstro',
		version='0.8.1.dev0',
		author='Giuliano Iorio',
		author_email='',
		url='',
		packages=['GAstro','GAstro/gaia_src'],
		include_package_data=True,
		ext_modules=ext_modules,
		zip_safe=False
)
