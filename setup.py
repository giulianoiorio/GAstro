#cython: language_level=3, boundscheck=False
from __future__ import print_function
import pip
import time
'''
#Check Cython installation
print('Checking Ctyhon')
try:
    import Cython
    cyv=Cython.__version__
    print('OK! (Version %s)'%cyv)
except:
    print('Cython is not present, I will install it for you my lord')
    pip.main(['install','Cython'])

#Check CythonGSL installation
print('Checking CtyhonGSL')
try:
    import cython_gsl
    print('OK!')
except:
    print('Cython is not present, I will install it for you my lord')
    pip.main(['install','CythonGSL'])

#Check Scipy>1.0 installation
print('Checking Scipy>1.0')
try:
    import scipy
    scv=scipy.__version__
    scvl=scv.split('.')
    if int(scvl[0])>0 or int(scvl[1])>19:
        print('OK! (Version %s)'%scv)
    else:
        print('Version %s too old. I will install the lastest version' % scv)
        pip.main(['install','scipy'])
except:
    print('Scipy is not present, I will install it for you my lord')
    pip.main(['install','scipy'])
'''


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
		version='0.8.0.dev0',
		author='Giuliano Iorio',
		author_email='',
		url='',
		packages=['GAstro','GAstro/gaia_src'],
		include_package_data=True,
		ext_modules=ext_modules,
		zip_safe=False
)
