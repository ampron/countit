#!/usr/bin python2.7
# -*- encoding: UTF-8 -*-

# built-in modules
import sys
import traceback

# local modules
sys.path.insert(0, '../')
import countit

if __name__ == '__main__':
    app = countit.App()#debug_mode=True)
    app.run()
# END if