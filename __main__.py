#!/usr/bin python2.7
# -*- encoding: UTF-8 -*-

# built-in modules
import sys
import traceback

#==============================================================================
def main(*args):
    app = CountItApp()
    app.run()
# END main

#===============================================================================
if __name__ == '__main__':
    # The following try-except statement is a work-around to allow Windows users
    # to see output and errors when they run the script from the file browser
    # with a double-click
    try:
        main(*sys.argv[1:])
    except Exception as err:
        exc_type, exc_value, exc_tb = sys.exc_info()
        bad_file, bad_line, func_name, text = traceback.extract_tb(exc_tb)[-1]
        print 'Error in {}'.format(bad_file)
        print '{} on {}: {}'.format(type(err).__name__, bad_line, err)
        print ''
    finally:
        raw_input("press any key to exit")
    # END try
# END if