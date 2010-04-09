'''
Created on Apr 9, 2010

@author: sonhua
'''

"""
from distutils.core import setup
import py2exe

setup(console=['./src/ee6901/ImageMosaick.py'])
"""

from distutils.core import setup
import py2exe

import matplotlib

setup(
    console=['./src/ee6901/ImageMosaick.py'],
    options={
             'py2exe': {
                        'packages' : ['matplotlib', 'pytz'],
                       }
            },
    data_files=matplotlib.get_py2exe_datafiles()
)
