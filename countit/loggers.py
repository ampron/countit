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

# third-party modules

#==============================================================================
def passable_method(func):
    def func_wrapper(*args, **kwargs):
        if args[0].pass_all:
            return None
        else:
            return func(*args, **kwargs)
        # END if
    # END func_wrapper
    return func_wrapper
# END passable_method

#==============================================================================
class HardLogger(object):
    ''' '''
    
    def __init__(self, filename, pass_all=False):
        self.fn = filename
        self.pass_all = pass_all
        self.pre = {}
    # END __init__
    
    @passable_method
    def create_type(self, name, str_prefix):
        self.pre[name] = str_prefix
    # END create_type
    
    @passable_method
    def write(self, s, t=None, _hang=False):
        if t is not None: s = self.pre[t] + s
        f = open(self.fn, 'a')
        f.write(s)
        if _hang:
            return f
        else:
            f.close()
        # END if
    # END write
    
    @passable_method
    def writeln(self, *args, **kwargs):
        kwargs['_hang'] = True
        f = self.write(*args, **kwargs)
        f.write('\n')
        f.close()
    # END writeln
# END HardLogger