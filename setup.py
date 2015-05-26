#!/usr/bin/env python

from distutils.core import setup

setup( name='countIt',
       version='1.4.0',
       description="Python application to assist in state assignment of noisy data from a system displaying time-dependent switching between quantized states",
       author='Alex Pronschinske',
       url='https://www.github.com/ampron',
       packages=['countit',],
       scripts=[ 'scripts/run_countIt_here.py',
               ],
     )