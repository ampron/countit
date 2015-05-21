#!/usr/bin/python
# -*- encoding: UTF-8 -*-
'''Script or Module Title
    
    This section should be a summary of important information to help the editor
    understand the purpose and/or operation of the included code.
    
    List of classes: -none-
    List of functions:
        main
'''

# built-in modules
import re

# third-party modules
import numpy as np

#==============================================================================
def open_jv(file_name):
    Y = []
    with open(file_name) as f:
       for line in f:
           Y.append(float(line))
       # END for
    # END with
    return np.array(Y)
# END open_jv

#==============================================================================
def open_jv_autosave(file_name):
    Y = []
    lbls = []
    fall_back = False
    with open(file_name) as f:
       i = 1
       for line in f:
           try:
               y, a = re.search(r'^([\d.]+),(\w*)\n', line).groups()
               Y.append(float(y))
               lbls.append(a)
           except Exception:
               print 'autosaved file not valid'
               print 'line {}: {}'.format(i, repr(line))
               fall_back = True
               break
           # END try
           i += 1
       # END for
    # END with
    if fall_back:
        Y = open_jv( re.sub(r'\.sa\.csv~$', '', file_name) )
        lbls = []
    else:
        Y = np.array(Y)
    # END if
    return Y, lbls
# END open_jv_autosave