# -*- encoding: UTF-8 -*-
'''Count I(t)
    
    Super helpful application for quickly doing switching analysis!
'''

# built-in modules
import sys
# Bug fix for matplotlib 1.4.0; allows for interactive mode
sys.ps1 = 'SOMETHING'
sys.path.append('../../ampPy')
import os
import os.path
import time
import re
#from pprint import pprint

# third-party modules
from grouped_lists import GroupedList, ColoredList
import numpy as np
import matplotlib as mpl
mpl.rc( 'font',
        **{ 'sans-serif': [ 'Droid Sans', 'Helvetica Neue', 'Helvetica',
                            'Verdana', 'Bitstream Vera Sans', 'sans-serif'
                          ]
          }
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from sklearn.cluster import KMeans

# Remove needed keymapping
# defaults are...
# 'keymap.all_axes': 'a'
# 'keymap.back':       ['left', 'c', 'backspace']
# 'keymap.forward':    ['right', 'v']
# 'keymap.fullscreen': ('f', 'ctrl+f')
# 'keymap.grid':       'g'
# 'keymap.home':       ['h', 'r', 'home']
# 'keymap.pan':        'p'
# 'keymap.quit':       ('ctrl+w', 'cmd+w')
# 'keymap.save':       ('s', 'ctrl+s')
# 'keymap.xscale':     ['k', 'L']
# 'keymap.yscale':     'l'
# 'keymap.zoom':       'o'

plt.rcParams['keymap.save'] = ''
plt.rcParams['keymap.save'] = ['h', 'home']
plt.rcParams['keymap.zoom'] = 'z'
plt.rcParams['keymap.all_axes'] = ''

base_menu = '''Program Options:
  quit ... Quit
  open ... Open raw data
  save ... Save state assignments
  a ...... Select all
View Options:
  r ............... Refresh
  line ............ Toggle line display
  color [lbl] ..... Re-color state
Data Options:
  sm [w] [p] .... Smooth data, w: window (pnts),
                               p: polynomial order
Filter Options:
  Fc ........... Clear all filters
  Fa ........... Add filter, options: peak, valley, slope, g:[label]
                             negation operator: ![filter]
Assignment Options:
  ro ............... Re-order state labels from high to low
  z ................ Undo last state assignment
  l ................ Toggle assignment lock
  as [label]........ Assign single state from selection
  am ............... Assign multiple states from selection
  amc .............. Assign multiple states from smart chunks
  es [label]........ Expand state assignments w/ moving avg.
  sp [label]........ Expand state assignments w/ simple pick-up
  cat [l1] [l2] .... Concatenate states n & m
  u ................ Unassign selection
  uout [3.5] [20]... Unassign normal outliers: [zcut], [rnds]
  u1 [label] ....... Unassign isolated points from [label]
  del [label] ...... Delete state
  clearall ......... Clear all state assignments'''

#==============================================================================
def logging_method(func):
    def wrapped_func(*args, **kwargs):
        args[0].log.writeln(
            '{}(self, {}, {})'.format(func.func_name, args[1:], kwargs)
        )
        return func(*args, **kwargs)
    # END wrapped_func
    return wrapped_func
# END logging_method

#==============================================================================
class CountItApp(object):
    '''Name
    
    Main program for conducting state-switching analysis
    
    Instantiation Args:
        debug_mode (bool): switch for running in debug mode
    Instance Attributes:
        _dbm (bool): "DeBug Mode", private switch
        
        cwfile (str): "Current "Working FILE", i.e. name of open file
        Y (np.ndarray): raw data from I(t) spectrum, will be altered
        saY (ColoredList((i,y))): list of (i,y) tuples grouped into states
            (i.e. "s"tate "a"ssigned "y"-data)
        sqrerr (float): sum of square errors for current state assignments
        
        fig (plt.Figure): UI figure
        ax_main (plt.Axes): Main axes showing all data with state color-coding
        ax_hist (plt.Axes): axes with histogram of data in main axes
        ax_zoom (plt.Axes): axes with zoomed view
        ax_zhist (plt.Axes): axes with histogram of data in zoomed axes
        keymap (dict): keys are str representing keyboard presses and values
            are functions called on press
        
    Magic Methods: -none-
    Class Methods: -none-
    Object Methods:
        run, _root_menu, set_date, _toggle_selector, _on_span_select,
        _keyboard_press, assign_selection, new_assignments,
        refresh_graph
    '''
    
    __version__ = '1.0.0'
    __author__ = 'Dr. Alex Pronschinske'
    
    def __init__(self, debug_mode=False):
        self._dbm = debug_mode
        
        self.cwfile = ''
        self._zfilter = ''
        self._sm_w = 0
        self._sm_p = 1
        
        self._autonames_src = (
            None, '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
            'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R',
            'T', 'U', 'W', 'X', 'Y'
        )
        self._autonames = list(self._autonames_src)
        
        self._keymap = {}
    # END init
    
    def run(self):
        '''Run application
        '''
        # Create debugging log
        self.log = HardLogger(
            'count_it_v{}_log{}.txt'.format(
                CountItApp.__version__, int(time.time())
            ),
            pass_all=not self._dbm
        )
        self.log.create_type('ui', 'User Input>>> ')
        
        self._line_disp = True
        self._tmpout = ''
        
        # Ask user for signature
        if not self._dbm: os.system('cls' if os.name == 'nt' else 'clear')
        sig = raw_input('Enter user signature >>> ')
        self.log.writeln(sig, t='ui')
        self.sig = re.sub(r'[^\w]', '', sig)
        
        self.open_data_file()
        if not self.cwfile: return
        
        # Create GUI
        plt.ion()
        gs = gridspec.GridSpec(2, 2, width_ratios=[1.618, 1])
        self.fig = plt.figure(figsize=(16, 9))
        self.ax_main = self.fig.add_subplot(gs[0,:-1])
        self.ax_zoom = self.fig.add_subplot(gs[1,:-1])
        self.ax_hist = self.fig.add_subplot(gs[0,-1], sharey=self.ax_main)
        self.ax_zhist = self.fig.add_subplot(gs[1,-1], sharey=self.ax_zoom)
        self.fig.canvas.mpl_connect('key_press_event', self._keyboard_press)
        self.selector = RectangleSelector(
            self.ax_main, self._on_rect_select, useblit=True,
            rectprops={'facecolor':'none', 'edgecolor':'red'},
            button=3
        )
        plt.show()
        self.refresh_graph()
        self._root_menu()
    # END run
    
    # Terminal UI
    #------------
    def _root_menu(self, refresh=False):
        while True:
            try:
                self._keymap = {
                    '0': self.assign_selection,
                    'alt+0': self.assign_selection,
                    'l': self.saY.toggle_reassign,
                    'alt+l': self.saY.toggle_reassign,
                    'u': self.unassign_selection,
                    'alt+u': self.unassign_selection,
                    'a': self.select_all,
                    'alt+a': self.select_all,
                    'ctrl+z': self.undo_assignment,
                }
            except Exception:
                pass
            # END try
            
            if (not self._dbm) or refresh:
                os.system('cls' if os.name == 'nt' else 'clear')
                print 'Count I(t) ver. {}'.format(CountItApp.__version__)
                print 60*'='
            # END if
            print base_menu
            print 'Current working file: "{}"'.format(self.cwfile)
            print 'States: {} (lock is {})'.format(
                sorted(self.saY.keys()), self.saY.get_reassign()
            )
            print ( 'Current selection filter: '+
                    '[{}]'.format(self._zfilter) +
                    '{:>7d} / {} points'.format(self._N_zdata, len(self.Y))
                  )
            print 'Command history {}'.format(self.cmd_hist[-5:])
            print '----------'
            if self._tmpout:
                print self._tmpout
                self._tmpout = ''
            # END if
            print 60*'='
            response = raw_input('{}>>> '.format(self.sig)).strip()
            self.log.writeln(response, t='ui')
            self.cmd_hist.append(response)
            
            refresh = False
            if re.search(r'^quit$', response, re.I):
                self.log.writeln('User exited program')
                return None
            elif re.search(r'^save$', response, re.I):
                self.save_assignments()
            elif re.search(r'^open$', response, re.I):
                self.open_data_file()
            elif re.search(r'^r$', response, re.I): refresh=True
            elif re.search(r'^line$', response, re.I):
                self._line_disp = not self._line_disp
            elif re.search(r'^color \w+$', response, re.I):
                try:
                    self.saY.cycle_group_color(
                        re.search('\w+$', response).group(0)
                    )
                except Exception as err:
                    print '{} {}'.format(type(err), err)
                # try
            elif re.search(r'^d$', response, re.I):
                self.flash_deriv()
            elif re.search(r'^sm(?: \d+(?: \d)?)?$', response, re.I):
                args = [int(s) for s in re.findall(r'\d+', response)]
                try:
                    self._sm_w = args[0] + 1 - (args[0]%2)
                except IndexError:
                    self._sm_w = 0
                # END try
                try:
                    self._sm_p = args[1]
                except IndexError:
                    self._sm_p = 1
                # END try
            elif re.search(r'^a$', response, re.I):
                self.select_all()
            elif re.search(r'^fc$', response, re.I):
                self._zfilter = ''
            elif re.search(r'^fa', response, re.I):
                parts = response.split()
                for flt in parts[1:]: self._zfilter += flt + ','
            elif re.search(r'^ro$', response, re.I):
                self.reorder_labels()
            elif re.search(r'^z$', response, re.I):
                self.undo_assignment()
            elif re.search(r'^l|lock$', response, re.I):
                self.saY.toggle_reassign()
            elif re.search(r'^del \w+$', response, re.I):
                self.delete_state(
                    re.search('\w+$', response).group(0)
                )
            elif re.search(r'^clearall$', response, re.I):
                self.delete_all_states()
            elif re.search(r'^u$', response, re.I):
                self.unassign_selection()
            elif re.search(r'^cat(?: \w+){2,}$', response, re.I):
                response = re.sub(r'^cat ', '', response)
                stlbls = re.findall(r'\w+', response)
                self.concat_states(*stlbls)
            elif re.search(r'^as$', response, re.I):
                self.assign_selection()
            elif re.search(r'^as \w+$', response, re.I):
                self.assign_selection(
                    stlbl=re.search('\w+$', response).group(0)
                )
            elif re.search(r'^es \w+$', response, re.I):
                self.expand_assignment(
                    stlbl=re.search('\w+$', response).group(0)
                )
            elif re.search(r'^sp \w+$', response, re.I):
                self.simple_pickup(
                    stlbl=re.search('\w+$', response).group(0)
                )
            elif re.search(r'^am$', response, re.I):
                self.multiassign_1D_KMeans()
            elif re.search(r'^am2$', response, re.I):
                self.multiassign_raw_and_chunk()
            elif re.search(r'^amc$', response, re.I):
                self.multiassign_compound()
            elif re.search(r'^uout( [\d.]+( \d+)?)?$', response, re.I):
                parts = response.split()
                parts.pop(0)
                try:
                    parts[0] = float(parts[0])
                except:
                    pass
                # END try
                try:
                    parts[1] = int(parts[1])
                except:
                    pass
                # END try
                self.unassign_outliers(*parts)
            elif re.search(r'^u1 \w+$', response, re.I):
                self.unassign_singles( re.search('\w+$', response).group(0) )
            elif self._dbm and re.search(r'^ERROR$', response):
                raise RuntimeError
            else:
                print 'command not recognized'
            # END if
            
            self.refresh_graph()
            self._autosave()
        # END while
    # END _root_menu
    
    def _int_input(self, prompt, low_lim=None, up_lim=None):
        while True:
            response = raw_input(prompt)
            self.log.writeln(response, t='ui')
            try:
                n = int(response)
                if low_lim is not None and n < low_lim:
                    print '*** value must be >= {} ***'.format(low_lim)
                    continue
                if up_lim is not None and up_lim < n:
                    print '*** value must be <= {} ***'.format(up_lim)
                    continue
                break
            except ValueError:
                if response.lower() == 'x':
                    raise RuntimeError('User abort')
            # END try
            print '*** Input cannot be parsed as an int ***'
            self.log.writeln('rejected user input')
        # END while
        return n
    # END _int_input
    
    def _float_input(self, prompt, low_lim=None, up_lim=None):
        while True:
            response = raw_input(prompt)
            self.log.writeln(response, t='ui')
            try:
                y = float(response)
                if low_lim is not None and y < low_lim:
                    print '*** value must be >= {} ***'.format(low_lim)
                    continue
                if up_lim is not None and up_lim < y:
                    print '*** value must be <= {} ***'.format(up_lim)
                    continue
                break
            except ValueError:
                if response.lower() == 'x':
                    raise RuntimeError('User abort')
            # END try
            print '*** Input cannot be parsed as a float ***'
            self.log.writeln('rejected user input')
        # END while
        return y
    # END _int_input
    
    # Data management and manipulation
    #---------------------------------
    @logging_method
    def _autosave(self):
        fn = self.cwfile + '.sa.csv~'
        with open(fn, 'w') as f:
            for i in range(len(self.Y)):
                f.write(
                    '{},{}\n'.format( self.Y[i],
                                      self.saY.lkup_group(self.saY[i], '')
                                    )
                )
            # END for
        # END with
    # END _autosave
    
    def next_autoname(self, reset_enable=True):
        # Check for automatic reset
        if self.saY.N_groups == 0 and reset_enable:
            self._autonames = list(self._autonames_src)
        
        # first element of the list is always the last one that was returned
        # Begin by moving the last returned to the end of the list
        self._autonames.append(self._autonames.pop(0))
        # Now the first element is the candidate for the next offered name
        if self._autonames[0] is None:
            # update autoname list, all have been offered
            for i in range(1, len(self._autonames)):
                self._autonames[i] += self._autonames[i][0]
            return self.next_autoname()
        elif self._autonames[0] in self.saY.keys():
            # skip, it's already in use
            return self.next_autoname()
        else:
            # use
            return self._autonames[0]
        # END if
    # END next_autoname
    
    @logging_method
    def get_zdata(self):
        # Filter for peaks/valleys
        pnts = set(self.saY)
        filters = self._zfilter.split(',')
        self.log.writeln('filters split into '+repr(filters))
        for flt in filters:
            self.log.writeln('considering '+repr(flt))
            if not flt: continue
            negate = False
            if re.search(r'^!', flt):
                flt = flt[1:]
                negate = True
            # END if
            self.log.writeln('negate <-- '+repr(negate))
            try:
                if re.search(r'^g:', flt):
                    if not negate: pnts &= self.saY.get_group(flt[2:])
                    else: pnts -= self.saY.get_group(flt[2:])
                else:
                    if not negate: pnts &= self.Ypv.get_group(flt)
                    else: pnts -= self.Ypv.get_group(flt)
                # END if
            except KeyError:
                self.log.writeln('KeyError for '+repr(flt))
            # END try
        # END for
        self.log.writeln(
            'after user filters, len(pnts) = {}'.format(len(pnts))
        )
        
        # Filter out data outside the selection window
        for iy in set(pnts):
            if self.zrng[0][0] <= iy[0] and iy[0] <= self.zrng[1][0]:
                if self.zrng[0][1] <= iy[1] and iy[1] <= self.zrng[1][1]:
                    continue
                # END if
            # END if
            pnts.discard(iy)
        # END for
        self.log.writeln(
            'after selection area, len(pnts) = {}'.format(len(pnts))
        )
        
        # Apply smoothing
        try:
            Y = list( SG_smooth(self.Y, self._sm_w, self._sm_p) )
        except Exception:
            Y = list( self.Y )
        # END if
        
        Xout = np.zeros(len(pnts), dtype=np.int32)
        Yout = np.zeros(len(pnts))
        pnts = sorted(pnts, key=lambda iy: iy[0])
        for j in range(len(pnts)):
            Xout[j] = pnts[j][0]
            Yout[j] = Y[pnts[j][0]]
        # END for
        
        # cache the total number of selected points
        self._N_zdata = len(Xout)
        return Xout, Yout
    # END get_zdata
    
    @logging_method
    def open_data_file(self):
        '''UI Prompt for opening files'''
        # look up all valid files
        files = os.listdir('.')
        for _ in range(len(files)):
            fn = files.pop(0)
            # TODO: add compatibility for I(V)_mtrx files +"|I(V)_mtrx$"
            if re.search(r'^jv[^~]+$', fn): files.append(fn)
        # END for
        files.sort()
        
        for i in range(len(files)):
            print '({:<2d}) {}'.format(i+1, files[i])
        # END for
        print '(X ) abort'
        
        try:
            fnum = self._int_input('Select file >>> ', 1, len(files)+1) - 1
        except RuntimeError:
            print 'abort'
            return
        # END try
        fn = files[fnum]
        
        self.cwfile = fn
        self.log.writeln('cwfile set to "{}"'.format(self.cwfile))
        try:
            if re.search(r'^jv', fn):
                if os.path.exists(fn+'.sa.csv~'):
                    # open auto-saved file instead
                    Y, lbls = loadjvsa(fn+'.sa.csv~')
                    self._set_data(Y)
                    # set assignments
                    for i in range(len(self.Y)):
                        if lbls[i] != '':
                            self.saY.assign(lbls[i], self.saY[i])
                    # END for
                else:
                    # open original
                    self._set_data(loadjv(self.cwfile))
                # END if
            elif re.search(r'\.I(V)_mtrx$', fn):
                print 'not implemented, yet...'
            else:
                print 'unknown filetype'
                return
            # END if
        except Exception as err:
            print 'Error opening file'
            print '{} {}'.format(type(err), err)
            self.log.writeln(
                'Error opening file | {} {}'.format(type(err), err)
            )
        # END try
        
        # start time for speed analysis
        self.open_time = time.time()
        self.log.writeln('file opened at {} s'.format(int(self.open_time)))
        
        # create new command history
        self.cmd_hist = []
    # END open_data_file
    
    def _set_data(self, Y):
        self.Y = np.array(Y)
        self.Y -= np.min(self.Y)
        self.Y /= 1.01*np.max(self.Y)
        self._Y_orig = np.array(Y)
        self.saY = ColoredList( zip(range(len(Y)), self.Y) )
        self.zrng = ( (0, min(self.Y)), (len(self.Y)-1, max(self.Y)) )
        
        self.Ypv = GroupedList( self.saY )
        dY_f = np.diff(self.Y)
        for i in range(1, len(self.Y)-1):
            if 0 < dY_f[i-1] and dY_f[i] < 0:
                self.Ypv.assign('peak', self.Ypv[i])
            elif dY_f[i-1] < 0 and 0 < dY_f[i]:
                self.Ypv.assign('valley', self.Ypv[i])
            else:
                self.Ypv.assign('slope', self.Ypv[i])
            # END if
        # END for
    # END set_data
    
    @logging_method
    def save_assignments(self):
        # Create a list of just the assigned data points
        labels = [(i,lbl) for i,lbl in enumerate(self.saY.list_memberships())]
        for i in range(len(labels)):
            i_lbl = labels.pop(0)
            if i_lbl[1] is not None: labels.append(i_lbl)
        # END for
        
        # Create label re-mapping to sorted numbers
        st_means = [ (k, np.mean(np.array(list(self.saY.get_group(k)))[:,1]) )
                     for k in self.saY.keys()
                   ]
        if len(st_means) == 0:
            print 'No assignments to save'
            return
        # END if
        st_means.sort(key=lambda tup: tup[1])
        newlbl = {tup[0]: i for i, tup in enumerate(st_means)}
        
        # Get input arguments from user
        print 'Current working file: {}'.format(self.cwfile)
        save_name = raw_input('File name >>> ')
        self.log.writeln(save_name, t='ui')
        if save_name == '':
            print 'abort saving'
            return
        # END if
        if os.path.exists(save_name):
            response = raw_input(
                save_name+' already exists, overwrite file? [Y/n] >>> '
            )
            if re.search(r'y', response, re.IGNORECASE):
                pass
            else:
                print 'abort saving'
                return
            # END if
        # END if
        try:
            f = open(save_name, 'w')
        except OSError:
            print 'Filename is invalid'
            return
        # END try
        # write signature
        f.write('# Analyst: {}\n'.format(self.sig))
        f.write('# working time: {} s\n'.format(time.time()-self.open_time))
        # write entrance points
        f.write('# Entrance points\n')
        f.write('{}\n'.format(labels[0][0]))
        for i in range(1, len(labels)):
            if labels[i][1] != labels[i-1][1]:
                f.write('{}\n'.format(labels[i][0]))
        # END for
        # write exit points
        f.write('# Exit points\n')
        for i in range(len(labels)-1):
            if labels[i][1] != labels[i+1][1]:
                f.write('{}\n'.format(labels[i][0]))
        # END for
        f.write('{}\n'.format(labels[-1][0]))
        # write labels
        f.write('# State labels\n')
        f.write('{}\n'.format(labels[0][1]))
        for i in range(1, len(labels)):
            if labels[i][1] != labels[i-1][1]:
                f.write('{}\n'.format(newlbl[ labels[i][1] ]))
        # END for
        f.close()
        self.log.writeln('saved "{}"'.format(save_name))
        
        # Save a corresponding graph
        img_name = re.sub(r'\.[^.]+$', '', save_name) + '.png'
        self.fig.savefig(img_name, dpi=150)
        self.log.writeln('saved "{}"'.format(img_name))
    # END save_assignments
    
    # Analysis
    #---------
    @logging_method
    def calc_lifetimes(self):
        lifetimes = {lbl: [] for lbl in self.saY.keys()}
        labels = [(i,lbl) for i,lbl in enumerate(self.saY.list_memberships())]
        for i in range(len(labels)):
            i_lbl = labels.pop(0)
            if i_lbl[1] is not None: labels.append(i_lbl)
        # END for
        i0 = labels[0][0]
        for i in range(1, len(labels)):
            if labels[i-1][1] != labels[i][1]:
                lifetimes[labels[i-1][1]].append( labels[i-1][0]+1 - i0 )
                i0 = labels[i][0]
            # END if
        # END for
        lifetimes[labels[-1][1]].append( labels[-1][0]+1 - i0 )
        return lifetimes
    # END calc_lifetimes
    
    # State assignment methods
    #-------------------------
    @logging_method
    def reorder_labels(self):
        self.saY.commit()
        self._autonames = list(self._autonames_src)
        new_lbl_order = sorted(self.saY.keys())
        st_means = []
        for k in new_lbl_order:
            gXY = np.array(list(self.saY.get_group(k)))
            self.saY.overassign(k+'_', *list(self.saY.get_group(k)))
            st_means.append( (k,np.mean(gXY[:,1])) )
        # END for
        st_means.sort(key=lambda tup: tup[1])
        self.saY.reset_color_cycle()
        for k, _ in st_means:
            self.saY.overassign( self.next_autoname(),
                                 *list(self.saY.get_group(k+'_'))
                               )
        # END for
    # END reorder_labels
    
    @logging_method
    def delete_state(self, stlbl):
        if stlbl not in self.saY.keys(): return
        self.saY.commit()
        self.saY.disband(stlbl)
    # END delete_state
    
    @logging_method
    def delete_all_states(self):
        self.saY.commit()
        for k in self.saY.keys(): self.saY.disband(k)
    # END delete_state
    
    @logging_method
    def unassign_selection(self):
        zX, _ = self.get_zdata()
        if len(zX) == 0: return
        
        self.saY.commit()
        self.saY.unassign( *[self.saY[i] for i in zX] )
    # END unassign_selection
    
    @logging_method
    def unassign_singles(self, stlbl):
        if stlbl not in self.saY.keys(): return
        N_released = 0
        self.saY.commit()
        labels = [(i,lbl) for i,lbl in enumerate(self.saY.list_memberships())]
        for i in range(1, len(labels)-1):
            if ( labels[i][1] == stlbl and
                 labels[i-1][1] != labels[i][1] and
                 labels[i+1][1] != labels[i][1]
               ):
                self.saY.unassign(self.saY[labels[i][0]])
                N_released += 1
            # END if
        # END for
        if N_released == 0: self.saY.rollback()
        self._tmpout += '{} points unassigned from {}'.format( N_released,
                                                               stlbl
                                                             )
    # END unassign_singles
    
    @logging_method
    def unassign_outliers(self, zcutoff=3.5, rounds=20):
        '''Unassign points that are normal distribution outliers'''
        # drop save point in history before making assignments
        self.saY.commit()
        
        N_unassigned = 0
        for rnd in range(rounds):
            zX, _ = self.get_zdata()
            if len(zX) == 0: return
            
            # Calculate state distribution estimates (i.e. mean & stdev)
            ms = {}
            for lbl in self.saY.keys():
                stXY = np.array(
                    sorted(self.saY.get_group(lbl), key=lambda xy: xy[0])
                )
                ms[lbl] = ( np.median(stXY[:,1]), np.std(stXY[:,1]) )
            # END for
            
            N_u = 0
            for i in zX:
                lbl = self.saY.lkup_group( self.saY[i] )
                if lbl is None: continue
                if zcutoff < np.abs(self.Y[i] - ms[lbl][0]) / ms[lbl][1]:
                    self.saY.unassign( self.saY[i] )
                    N_u += 1
                # END if
            # END for
            
            self.log.writeln(
                'unassigned {} on round {}'.format(N_u, rnd)
            )
            if N_u == 0: return
            else: N_unassigned += N_u
        # END for
        
        if N_unassigned == 0: self.saY.rollback()
        self._tmpout += '{} points unassigned'.format(N_unassigned)
    # END unassign_outliers
    
    @logging_method
    def undo_assignment(self): self.saY.rollback()
    
    @logging_method
    def assign_selection(self, stlbl=None):
        zX, _ = self.get_zdata()
        if len(zX) == 0: return
        
        if stlbl is None:stlbl = self.next_autoname()
        self.saY.commit()
        for i in zX:
            self.saY.assign(stlbl, self.saY[i])
        # END for
    # END assign_selection
    
    @logging_method
    def concat_states(self, st1, *other_stlbls):
        if len(other_stlbls) == 0: return
        other_stlbls = list(other_stlbls)
        st2 = other_stlbls.pop(0)
        if st2 in self.saY.keys():
            self.saY.commit()
            former_members = self.saY.disband(st2)
            self.saY.overassign(st1, *former_members)
        # END if
        self.concat_states(st1, *other_stlbls)
    # END concat_states
    
    @logging_method
    def multiassign_selection(self):
        zX, zY = self.get_zdata()
        if len(zX) == 0: return
        
        print 'enter "X" at anytime to abort command'
        # Get input arguments from user
        while True:
            response = raw_input('Number of states >>> ')
            try:
                Nst = int(response)
                break
            except ValueError:
                if response.lower() == 'x': return
            # END try
            print '*** Invalid input, please try again ***'
        # END while
        if Nst < 2:
            print 'Cannot assign fewer than 2 states'
            return
        # END if
        
        guesses = []
        for i in range(Nst):
            while True:
                response = raw_input(
                    'approx. y-value of state {} >>> '.format(i+1)
                )
                try:
                    y0 = float(response)
                    guesses.append((y0, y0))
                    break
                except ValueError:
                    if response.lower() == 'x': return
                # END try
                print '*** Invalid input, please try again ***'
            # END while
        # END for
        guesses = np.array(guesses)
        
        # Fit a K-means model
        smY = SG_smooth(self.Y, 7, 1)
        classifier = KMeans(
            n_clusters=Nst, n_init=100,
            n_jobs=-1, max_iter=1000, init=guesses
        )
        #classifier = SpectralClustering(n_clusters=Nst)
        #classifier = AgglomerativeClustering(n_clusters=Nst, linkage='average')
        #classifier = DBSCAN(eps=0.0125, min_samples=30)
        data = np.array( [(self.Y[i],smY[i]) for i in zX] )
        #data = []
        #for j in range(len(zX)):
        #    if zX[j]-1 < 0: continue
        #    data.append( (zY[j], self.Y[zX[j]-1]) )
        ## END for
        data = np.array(data)
        classifier.fit(data)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(
            data[:,0], data[:,1], s=12, marker='o',
            c=classifier.labels_,
            cmap=plt.get_cmap('jet'),
            alpha=0.6, linewidths=0
        )
        fig.show()
        raw_input('press enter')
        plt.close(fig)
        
        # drop save point in history before making assignments
        self.saY.commit()
        
        # Assign based on model results
        new_st = {}
        for i in range(Nst): new_st[i] = self.next_autoname()
        for i in range(len(zX)):
            if classifier.labels_[i] < 0: continue
            self.saY.assign(
                new_st[classifier.labels_[i]], self.saY[zX[i]]
            )
        # END for
    # END multiassign_selection
    
    @logging_method
    def multiassign_1D_KMeans(self):
        '''Assign multiple states using only the displayed y-values'''
        zX, zY = self.get_zdata()
        if len(zX) == 0: return
        
        print 'enter "X" at anytime to abort command'
        try:
            Nst = self._int_input('Number of states >>> ', 2)
        except RuntimeError:
            print 'abort'
            return
        # END try
        
        # Fit a K-means model
        classifier = KMeans(
            n_clusters=Nst, n_init=100,
            n_jobs=-1, max_iter=1000, init='random'
        )
        data = np.array( [(y,) for y in zY] )
        classifier.fit(data)
        
        # drop save point in history before making assignments
        self.saY.commit()
        
        # Assign based on model results
        new_st = { i: self.next_autoname(reset_enable=False)
                   for i in range(Nst)
                 }
        for j in range(len(zX)):
            self.saY.assign(new_st[classifier.labels_[j]], self.saY[zX[j]])
        # END for
    # END multiassign_1D_KMeans
    
    @logging_method
    def multiassign_raw_and_chunk(self):
        '''Assign multiple states using only the displayed y-values'''
        zX, zY = self.get_zdata()
        if len(zX) == 0: return
        
        # Create "chunked data"
        Y_chunky = np.array(self.Y)
        i = 0
        while i < len(self.Y):
            # start lookahead when arriving at a point that is grouped
            if self.saY.is_grouped(self.saY[i]):
                curr_lbl = self.saY.lkup_group(self.saY[i])
                for j in range(i+1, len(self.Y)+1):
                    if j == len(self.Y): break
                    elif self.saY.lkup_group(self.saY[j]) != curr_lbl: break
                m = np.mean(self.Y[i:j])
                for i in range(i, j): Y_chunky[i] = m
            # END if
            i += 1
        # END while
        
        # show preview of chunky data and fitting space
        fig, ax = plt.subplots(2,1)
        ax[0].scatter(
            range(len(self.Y)), Y_chunky,
            s=12, marker='o', edgecolors='none', facecolor='black'
        )
        ax[0].plot(self.Y, '-k', linewidth=0.5, alpha=0.3)
        ax[1].scatter(
            self.Y, Y_chunky,
            s=12, marker='o', edgecolors='none', facecolor='black',
            alpha=0.6
        )
        fig.show()
        
        # User input
        print 'enter "X" at anytime to abort command'
        try:
            Nst = self._int_input('Number of states >>> ', 2)
        except RuntimeError:
            print 'abort'
            return
        # END try
        
        plt.close(fig)
        
        # Fit a K-means model
        classifier = KMeans(
            n_clusters=Nst, n_init=100,
            n_jobs=-1, max_iter=1000, init='random'
        )
        data = np.array( [(self.Y[i], Y_chunky[i]) for i in zX] )
        classifier.fit(data)
        
        # show preview of results
        cm = plt.get_cmap('Dark2')
        fig, ax = plt.subplots(2,1)
        ax[0].scatter(
            range(len(self.Y)), Y_chunky,
            s=12, marker='o', edgecolors='none',
            c=classifier.labels_, cmap=cm
        )
        ax[0].plot(self.Y, '-k', linewidth=0.5, alpha=0.3)
        ax[1].scatter(
            self.Y, Y_chunky,
            s=12, marker='o', edgecolors='none',
            c=classifier.labels_, cmap=cm
        )
        fig.show()
        raw_input('press enter to continue')
        plt.close(fig)
        
        # drop save point in history before making assignments
        self.saY.commit()
        
        # Assign based on model results
        new_st = { i: self.next_autoname(reset_enable=False)
                   for i in range(Nst)
                 }
        for j in range(len(zX)):
            self.saY.assign(new_st[classifier.labels_[j]], self.saY[zX[j]])
        # END for
    # END multiassign_1D_KMeans
    
    @logging_method
    def multiassign_compound(self):
        # Apply smoothing
        try:
            dY = SG_smooth(self.Y, self._sm_w, self._sm_p, 1)
        except Exception:
            dY = np.concatenate((np.diff(self.Y), [self.Y[-1]-self.Y[-2]]))
        # END if
        
        dY2 = abs(dY)
        fig, ax = plt.subplots(2, 1)
        #cm = plt.get_cmap('rainbow')
        cm = plt.get_cmap('YlOrRd')
        ax[0].set_title('Derivative')
        ax[0].scatter(
            range(len(dY)), dY,
            s=12, marker='o', edgecolors='none', c=dY2, cmap=cm, alpha=0.6
        )
        ax[1].set_title('Raw Data')
        ax[1].scatter(
            range(len(dY)), self.Y,
            s=12, marker='o', edgecolors='none', c=dY2, cmap=cm
        )
        ax[1].plot(self.Y, '-k', linewidth=0.5, alpha=0.3)
        fig.show()
        try:
            # TODO: will this break if the user enters 0?
            ycutoff = abs( self._float_input('Enter cuttoff value >>> ') )
        except RuntimeError:
            print 'abort'
            return
        # END try
        plt.close(fig)
        ssIY = []
        a = 0
        while a < len(dY):
            # start lookahead when arriving at a point that is "flat-enough"
            if abs(dY[a]) <= ycutoff:
                for b in range(a+1, len(dY)+1):
                    if b == len(dY): break
                    elif ycutoff < abs(dY[b]): break
                m = np.mean(self.Y[a:b])
                for a in range(a, b): ssIY.append( (a, m) )
            # END if
            a += 1
        # END while
        ssIY = np.array(ssIY)
        fig, ax = plt.subplots(1,2)
        cm = plt.get_cmap('Dark2')
        ax[0].scatter(
            ssIY[:,0], ssIY[:,1],
            s=12, marker='o', edgecolors='none', c=ssIY[:,1], cmap=cm
        )
        ax[0].plot(self.Y, '-k', linewidth=0.5, alpha=0.3)
        ax[1].hist(
            ssIY[:,1], bins=36, edgecolor='none', orientation='horizontal'
        )
        fig.show()
        try:
            Nst = self._int_input('Number of states >>> ', 2)
        except RuntimeError:
            print 'abort'
            return
        # END try
        plt.close(fig)
        
        # Use a K-means model to categorize by means
        classifier = KMeans(n_clusters=Nst)
        classifier.fit( np.array( [(y,) for i,y in ssIY] ) )
        
        # show preview of results
        fig, ax = plt.subplots(1,1)
        cm = plt.get_cmap('brg')
        ax.scatter(
            ssIY[:,0], ssIY[:,1],
            s=12, marker='o', edgecolors='none',
            c=classifier.labels_, cmap=cm
        )
        ax.plot(self.Y, '-k', linewidth=0.5, alpha=0.3)
        fig.show()
        raw_input('press enter to continue')
        plt.close(fig)
        
        self.saY.commit()
        
        max_lbl = max(classifier.labels_)
        remap = [ self.next_autoname(reset_enable=False)
                     for _ in range(max_lbl+1)
                   ]
        for j,lbl in enumerate(classifier.labels_):
            self.saY.assign(remap[lbl], self.saY[int(ssIY[j,0])])
        # END for
    # END multiassign_compound
    
    @logging_method
    def expand_assignment(self, stlbl):
        stXY = np.array(
            sorted(self.saY.get_group(stlbl), key=lambda xy: xy[0])
        )
        zchk = {i: True for i in self.get_zdata()[0]}
        
        print 'enter "X" at anytime to abort command'
        # Get input arguments from user
        try:
            N_buff = self._int_input( 'Size of point buffer >>> ',
                                      1, stXY.shape[0]
                                    )
        except RuntimeError:
            print 'abort'
            return
        # END try
        
        # plot stdev bars for user
        buff = stXY[-N_buff:,1]
        # END for
        m = np.mean(buff)
        std = np.std(buff)
        X = [0, len(self.Y)-1]
        M = np.array([m, m])
        self.ax_main.plot(X, M, '-k', linewidth=1)
        self.ax_main.plot(X, M+std, '--k', linewidth=1)
        self.ax_main.plot(X, M-std, '--k', linewidth=1)
        self.ax_main.plot(X, M+2*std, '--k', linewidth=0.5)
        self.ax_main.plot(X, M-2*std, '--k', linewidth=0.5)
        self.ax_main.plot(X, M+3*std, '--k', linewidth=0.5)
        self.ax_main.plot(X, M-3*std, '--k', linewidth=0.5)
        self.fig.canvas.draw()
        
        try:
            z_cuttoff = abs( self._float_input('z-score cutoff >>> ', 0) )
        except RuntimeError:
            print 'abort'
            return
        # END try
        
        # drop save point in history before making assignments
        self.saY.commit()
        N_assigned = 0
        
        # capture moving from right to left (<--)
        states = self.saY.list_memberships()
        buff = []
        for i in range(len(self.Y))[::-1]:
            if states[i] == stlbl:
                buff.insert(0, self.Y[i])
                if len(buff) > N_buff: buff.pop(-1)
            elif ( states[i] is None and len(buff) == N_buff and
                   abs(self.Y[i] - np.mean(buff))/std < z_cuttoff and
                   i in zchk
                 ):
                    self.saY.assign(stlbl, self.saY[i])
                    N_assigned += 1
                    buff.insert(0, self.Y[i])
                    buff.pop(-1)
                # END if
            # END if
        # END for
        
        # capture moving from left to right (-->)
        buff = []
        for i in range(len(self.Y)):
            if states[i] == stlbl:
                buff.append(self.Y[i])
                if len(buff) > N_buff: buff.pop(0)
            elif ( states[i] is None and len(buff) == N_buff and 
                   abs(self.Y[i] - np.mean(buff))/std < z_cuttoff and
                   i in zchk
                 ):
                    self.saY.assign(stlbl, self.saY[i])
                    N_assigned += 1
                    buff.append(self.Y[i])
                    buff.pop(0)
                # END if
            # END if
        # END for
        
        if N_assigned == 0: self.saY.rollback()
        self._tmpout += '{} points assigned to {}'.format(N_assigned, stlbl)
    # END expand_assignment
    
    @logging_method
    def simple_pickup(self, stlbl):
        stlbl_ = stlbl + '_'
        committed = False
        i = 0
        N_pickedup = 0
        if not self.saY.is_grouped(self.saY[i]):
            for i in range(len(self.Y)):
                if self.saY.is_grouped(self.saY[i]): break
        while i < len(self.Y):
            # start lookahead when arriving at a valid point
            if ( not self.saY.is_grouped(self.saY[i]) and
                 self.saY.lkup_group(self.saY[i-1]) == stlbl
               ):
                for j in range(i+1, len(self.Y)+1):
                    if j == len(self.Y):
                        return
                    elif self.saY.is_grouped(self.saY[j]):
                        if self.saY.lkup_group(self.saY[i-1]) == stlbl:
                            contained_x = True
                        else:
                            contained_x = False
                        # END if
                        break
                    # END if
                # END for
                contained_y = True
                if contained_x:
                    for y in self.Y[i:j]:
                        if self.Y[i] < self.Y[j]:
                            if y < self.Y[i] or self.Y[j] < y:
                                contained_y = False
                                break
                            # END if
                        else:
                            if y > self.Y[i] or self.Y[j] < y:
                                contained_y = False
                                break
                            # END if
                        # END if
                    # END for
                # END if
                if contained_x and contained_y:
                    if not committed:
                        self.saY.commit()
                        committed = True
                    self.saY.assign(stlbl, *self.saY[i:j])
                    N_pickedup += j-i
                elif contained_x and not contained_y:
                    if not committed:
                        self.saY.commit()
                        committed = True
                    self.saY.assign(stlbl_, *self.saY[i:j])
                # END if
                i = j
            # END if
            i += 1
        # END while
        self._tmpout += 'added {} points to state "{}"'.format( N_pickedup,
                                                                stlbl
                                                              )
        self.log.writeln('add {} points to "{}"'.format(N_pickedup, stlbl))
    # END simple_pickup
    
    # Graph UI
    #---------
    def _on_rect_select(self, eclick, erelease):
        # eclick and erelease are the press and release events
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        # END if
        if x2 < x1:
            x1, x2 = x2, x1
        if x1 < 0:
            x1 = 0
        if len(self.Y)-1 < x2:
            x2 = len(self.Y)-1
        if y2 < y1:
            y1, y2 = y2, y1
        
        self.zrng = ( (x1, y1), (x2, y2) )
        self.refresh_graph()
    # END _on_rect_select
    
    def _keyboard_press(self, event):
        try:
            self.Y
        except Exception:
            return
        # END try
        keypress = re.sub(r'^alt\+', '', event.key)
        if keypress in self._keymap:
            self.log.writeln('hotkey "{}"'.format(keypress), t='ui')
            self._keymap[keypress]()
            self.refresh_graph()
            if self.cwfile: self._autosave()
        elif re.search(r'^[1-9]$', keypress):
            self.log.writeln('hotkey "{}"'.format(keypress), t='ui')
            self.assign_selection(keypress)
            self.refresh_graph()
            if self.cwfile: self._autosave()
        else:
            return
        # END if
    # END _keyboard_press
    
    @logging_method
    def select_all(self):
        self.zrng = ( (0, min(self.Y)), (len(self.Y)-1, max(self.Y)) )
    # END select_all()
    
    @logging_method
    def refresh_graph(self):
        self.ax_main.clear()
        self.ax_zoom.clear()
        self.ax_hist.clear()
        self.ax_zhist.clear()
        #self.ax.set_axisbelow(True)
        
        # Apply smoothing
        try:
            Y = list( SG_smooth(self.Y, self._sm_w, self._sm_p) )
        except Exception:
            Y = self.Y
        # END if
        
        # Histogram bin edges
        histbins = np.linspace(min(self.Y), max(self.Y), 36+1)
        
        self.ax_hist.hist(self.Y, bins=histbins,
            color='white', edgecolor='black',
            orientation='horizontal', histtype='stepfilled'
        )
        lgd_handles = []
        lgd_labels = []
        for stlbl in sorted(self.saY.keys(), reverse=True):
            if len(self.saY.get_group(stlbl)) == 0: continue
            stXY = np.array(
                sorted(self.saY.get_group(stlbl), key=lambda xy: xy[0])
            )
            for i in range(stXY.shape[0]):
                stXY[i,1] = Y[int(stXY[i,0])]
            out = self.ax_main.scatter(
                stXY[:,0], stXY[:,1], s=12, marker='o',
                alpha=0.6, edgecolors='none', linewidths=1,
                facecolors=self.saY.lkup_group_color(stlbl),
                label=stlbl
            )
            lgd_handles.append(out)
            lgd_labels.append(stlbl)
            self.ax_hist.hist(
                stXY[:,1], bins=histbins, stacked=True,
                color=self.saY.lkup_group_color(stlbl), edgecolor='none',
                alpha=0.6, orientation='horizontal'
            )
        # END for
        
        if self._line_disp:
            lines = self.ax_main.plot(self.Y, '-k', linewidth=0.5, alpha=0.3)
            lines[0].zorder -= 10
        # END if
        if self.saY.N_ungrouped > 0:
            ugXY = np.array(
                sorted(self.saY.get_ungrouped(), key=lambda xy: xy[0])
            )
            for i in range(ugXY.shape[0]):
                ugXY[i,1] = Y[int(ugXY[i,0])]
            self.ax_main.scatter(
                ugXY[:,0], ugXY[:,1], s=12, marker='o',
                facecolor='none', alpha=0.6,
                edgecolors='black', linewidths=1
            )
        # END if
        
        self.ax_main.set_xlim(-20, len(self.Y)+19)
        ymin = min(self.Y)
        ymax = max(self.Y)
        self.ax_main.set_ylim(ymin-0.1*(ymax-ymin), ymax+0.1*(ymax-ymin))
        
        #self.ax_hist.set_ylim(ymin-0.1*(ymax-ymin), ymax+0.1*(ymax-ymin))
        
        # Draw box on main graph showing where the selection area is
        dx = self.zrng[1][0] - self.zrng[0][0]
        dy = self.zrng[1][1] - self.zrng[0][1]
        self.ax_main.add_patch(
            Rectangle(
                self.zrng[0], dx, dy,
                facecolor='none', edgecolor='red', linewidth=0.5
            )
        )
        
        # Plot data in zoom window
        zX, zY = self.get_zdata()
        self.log.writeln('{} points in selection window'.format(len(zX)))
        if self._dbm: print '{} points in selection window'.format(len(zX))
        self.ax_zoom.scatter( zX, zY, s=12, marker='o',
                              c='black', alpha=0.6, edgecolors='none'
                            )
        # line connecting raw data
        if self._line_disp:
            self.ax_zoom.plot(self.Y, '-k', linewidth=0.5, alpha=0.3)
        # set axis limits
        try:
            self.ax_zoom.set_xlim(zX[0], zX[-1])
            zYmin = min(zY)
            zYmax = max(zY)
            dzY = zYmax-zYmin
            self.ax_zoom.set_ylim(zYmin-0.1*dzY, zYmax+0.1*dzY)
            # plot corresponding histogram
            nbins = len(zY)/100
            if nbins < 12: nbins = len(zY)/20
            if nbins < 12: nbins = 12
            self.ax_zhist.hist(
                zY, bins=nbins,
                edgecolor='none',
                histtype='stepfilled', orientation='horizontal'
            )
            self.ax_zhist.set_ylim(zYmin-0.1*dzY, zYmax+0.1*dzY)
        except IndexError:
            self.ax_zoom.set_xlim(self.zrng[0][0], self.zrng[1][0])
            self.ax_zoom.set_ylim(self.zrng[0][1], self.zrng[1][1])
            self.ax_zhist.set_ylim(self.zrng[0][1], self.zrng[1][1])
        # END try
        
        self.ax_hist.xaxis.set_ticks([])
        self.ax_main.grid(color=(0.8,0.8,0.8), linewidth=1.5)
        self.ax_zoom.grid(color=(0.8,0.8,0.8), linewidth=1.5)
        #self.ax.minorticks_on()
        
        self.ax_hist.legend(lgd_handles, lgd_labels)
        
        self.ax_main.set_title(
            u'{:d}/{:d} ({:0.1%}) unassigned'.format(
                self.saY.N_ungrouped, len(self.Y),
                float(self.saY.N_ungrouped)/len(self.Y)
            )
        )
        self.ax_zoom.set_title( '{} points selected'.format(len(zX)) )
        
        self.fig.tight_layout()
        self.fig.canvas.draw()
    # END refresh_graph
# END CountItApp

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

#==============================================================================
def loadjvsa(file_name):
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
        Y = loadjv( re.sub(r'\.sa\.csv~$', '', file_name) )
        lbls = []
    else:
        Y = np.array(Y)
    # END if
    return Y, lbls
# END loadjv

#==============================================================================
def loadjv(file_name):
    Y = []
    with open(file_name) as f:
       for line in f:
           Y.append(float(line))
       # END for
    # END with
    return np.array(Y)
# END loadjv

#===============================================================================
def SG_smooth(y, window_size, order, deriv=0):
    '''Savitzky-Golay Smoothing & Differentiating Function
    
    Args:
        Y (list): Objective data array.
        window_size (int): Number of points to use in the local regressions.
                        Should be an odd integer.
        order (int): Order of the polynomial used in the local regressions.
                    Must be less than window_size - 1.
        deriv = 0 (int): The order of the derivative to take.
    Returns:
        (ndarray)  The resulting smoothed curve data. (or it's n-th derivative)
    Test:
        t = np.linnp.ce(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = sg_smooth(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
    '''
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    # END try
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    
    order_range = range(order+1)
    half_window = (window_size -1) / 2
    
    # precompute coefficients
    b = np.mat(
        [
            [k**i for i in order_range] for k in range(
                -half_window, half_window+1
            )
        ]
    )
    m = np.linalg.pinv(b).A[deriv]
    
    # pad the function at the ends with reflections
    left_pad = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    right_pad = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((left_pad, y, right_pad))
    
    return ((-1)**deriv) * np.convolve( m, y, mode='valid')
# END sg_smooth