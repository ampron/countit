# -*- encoding: UTF-8 -*-
'''Count I(t)
    
    Super helpful application for quickly doing switching analysis!
'''

# built-in modules
import sys
# Bug fix for matplotlib 1.4.0; allows for interactive mode
sys.ps1 = 'SOMETHING'
import os
import os.path
import time
import re
import math
from pprint import pprint

# local modules
from loggers import HardLogger
from numerical import SG_smooth
import file_importers as fimps

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
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from sklearn.cluster import KMeans, MiniBatchKMeans

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
  line ............ Toggle line display
  color [lbl] ..... Re-color state
  summ ............ View summary plot
Data Options:
  sm ............ Smooth data, algorithms: sg [w] [p], deriv, sa, res
Filter Options:
  fr ........... Reset filter
  fset [set] ... Set filter
  fin [set] .... Add intersection
  fun [set]..... Add union
  ( Valid sets: all, peak, valley, slope, g:[label] )
  ( Negation operator: ![set]                       )
Assignment Options:
  ro ..................... Re-order state labels from high to low
  z ...................... Undo last state assignment
  lock ................... Toggle assignment lock
  as [label].............. Assign single state from selection
  am [N] ................. Assign [N] states from selection
  am2d ................... Assign multiple states from selection in 2-D
  aflat .................. Assign flat areas
  es [label].............. Expand state assignments w/ moving avg.
  sp [label].............. Safe pick-up (global)
  cat [l1] [l2] .......... Concatenate states n & m
  u ...................... Unassign selection
  uout [l] [w] [z] [r] ... Unassign normal outliers from [l] (global)
  ushort [lim] [lbl] ..... Unassign segments shorter than [lim] from [lbl]
  del [label] ............ Delete state
  clearall ............... Clear all state assignments'''

#==============================================================================
def logging_method(func):
    def wrapped_func(*args, **kwargs):
        args[0].log.writeln(
            '{}(self, {}, {})'.format( func.func_name, str(args[1:])[1:-1],
                                       kwargs
                                     )
        )
        return func(*args, **kwargs)
    # END wrapped_func
    return wrapped_func
# END logging_method

#==============================================================================
class App(object):
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
    
    __version__ = '1.3.0'
    
    def __init__(self, debug_mode=False):
        self._dbm = debug_mode
        # Create debugging log
        self.log = HardLogger(
            'count_it_v{}_log{}.txt'.format(
                App.__version__, int(time.time())
            ),
            pass_all=not self._dbm
        )
        self.log.create_type('ui', 'User Input>>> ')
        
        self.sig = ''
        self.cwfile = ''
        self._fstk = 'all'
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
        self._line_disp = True
        self._tmpout = ''
        
        # Ask user for signature
        if not self._dbm: os.system('cls' if os.name == 'nt' else 'clear')
        sig = raw_input('Enter user signature >>> ')
        self.log.writeln(sig, t='ui')
        self.sig = re.sub(r'[^\w]', '', sig)
        
        self.open_data_file()
        if not self.cwfile: return
        
        # Connect terminal commands to object methods
        self.cmd_map = {
            'quit': self.quit,
            'open': self.open_data_file,
            'save': self.save_assignments,
            'a': self.select_all,
            'line': self.toggle_line_display,
            'color': self.cycle_state_color,
            'summ': self.view_summary_plot,
            'sm': self.set_smoothing_func,
            'fr': self.filter_reset,
            'fset': self.filter_set,
            'fin': self.filter_push_intersection,
            'fun': self.filter_push_union,
            'ro': self.reorder_labels,
            'z': self.undo_assignment,
            'lock': self.toggle_lock,
            'as': self.assign_selection,
            'am': self.multiassign_1D_KMeans,
            'am2d': self.multiassign_2D_KMeans,
            'aflat': self.multiassign_compound,
            'es': self.expand_assignment,
            'sp': self.simple_pickup,
            'cat': self.concat_states,
            'u': self.unassign_selection,
            'uout': self.unassign_outliers,
            'ushort': self.unassign_short_lives,
            'del': self.delete_state,
            'clearall': self.delete_all_states,
        }
        
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
        
        self._keymap = {
            '0': self.assign_selection,
            'alt+0': self.assign_selection,
            'u': self.unassign_selection,
            'alt+u': self.unassign_selection,
            'a': self.select_all,
            'alt+a': self.select_all,
            'ctrl+z': self.undo_assignment,
        }
        
        self.refresh_graph()
        self._idle()
        
        # Clean up
        plt.close('all')
    # END run
    
    def quit(self, *args):
        self.log.writeln('User exited program')
        raise KeyboardInterrupt()
    # END quit
    
    # Terminal UI
    #------------
    def _idle(self):
        while True:
            # refresh text interface menu
            if not self._dbm:
                os.system('cls' if os.name == 'nt' else 'clear')
            self.print_menu()
            
            # wait for user input
            cmd, args = self._input_cmd()
            
            # validate input
            if not cmd:
                self._tmpout += (
                    'No command entered'.format(cmd)
                )
                continue
            elif cmd not in self.cmd_map:
                self._tmpout += (
                    'No action; "{}" is not a recognized command'.format(cmd)
                )
                continue
            # END if
            
            # run subroutine that matches user input
            try:
                self.cmd_map[cmd](*args)
                self.cmd_hist.append('{}({})'.format(cmd, ','.join(args)))
                self._autosave()
            except RuntimeError:
                pass
            except KeyboardInterrupt:
                break
            # END try
            
            # refresh graph ui with results of subroutine
            self.refresh_graph()
        # END while
    # END _idle
    
    def print_menu(self):
        # Header
        print 'Count I(t) ver. {}'.format(App.__version__)
        print 60*'='
        
        # Command menu
        print base_menu
        
        # Status
        print 'Current working file: "{}"'.format(self.cwfile)
        print 'States: {} (lock is {})'.format(
            ', '.join( sorted(self.saY.keys()) ), not self.saY.get_reassign()
        )
        print ( 'Current selection filter: '+
                '{} '.format(re.sub(r'[\[\]\',]', '', str(self._fstk))) +
                '({:>7d} / {} points)'.format(self._N_zdata, len(self.Y))
              )
        print 'Command history: {}'.format(', '.join(self.cmd_hist[-5:]))
        print '----------'
        if self._tmpout:
            print self._tmpout
            self._tmpout = ''
        # END if
        print 60*'='
    # END print_menu
    
    def _input_cmd(self):
        response = raw_input('{}>>> '.format(self.sig)).strip()
        # prevent user from inputing reserved characters
        response = re.sub(r'[|&]', '', response) 
        parts = response.split()
        try:
            cmd = parts[0]
        except IndexError:
            cmd = []
        # END try
        try:
            args = parts[1:]
        except IndexError:
            args = []
        # END try
        self.log.writeln(response, t='ui')
        return cmd, args 
    # END _input_cmd
    
    def _yn_input(self, prompt):
        while True:
            response = raw_input(prompt)
            self.log.writeln(response, t='ui')
            if response.lower() == 'y':
                return True
            elif response.lower() == 'n':
                return False
            elif response.lower() == 'x':
                raise RuntimeError('User abort')
            # END if
            self.log.writeln('rejected user input')
        # END while
    # END _yn_input
    
    def _stlbl_input(self, prompt):
        while True:
            response = raw_input(prompt)
            self.log.writeln(response, t='ui')
            if response.lower() == 'x':
                raise RuntimeError('User abort')
            elif re.search(r'^\w+$', response):
                return response
            # END if
            self.log.writeln('rejected user input')
        # END while
    # END _stlbl_input
    
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
            self.cmd_hist.append('hotkey "{}"'.format(keypress))
            self._keymap[keypress]()
            self.refresh_graph()
            if self.cwfile: self._autosave()
        elif re.search(r'^[1-9]$', keypress):
            self.log.writeln('hotkey "{}"'.format(keypress), t='ui')
            self.cmd_hist.append('hotkey "{}"'.format(keypress))
            self.assign_selection(keypress)
            self.refresh_graph()
            if self.cwfile: self._autosave()
        else:
            return
        # END if
    # END _keyboard_press
    
    @logging_method
    def select_all(self, *args):
        self.zrng = ( (0, min(self.Y)), (len(self.Y)-1, max(self.Y)) )
    # END select_all()
    
    @logging_method
    def toggle_line_display(self, *args):
        self._line_disp = not self._line_disp
    # END toggle_line_display
    
    @logging_method
    def cycle_state_color(self, *args):
        if len(args) < 0: return
        try:
            self.saY.cycle_group_color(args[0])
        except Exception as err:
            print '{} {}'.format(type(err), err)
        # try
    # END cycle_state_color
    
    @logging_method
    def plot_summary(self):
        Y_chunky = self.calc_result_data()
        lifetimes = self.calc_lifetimes()
        # show preview of chunky data and fitting space
        nrows = int(2 + math.ceil(0.5*self.saY.N_groups))
        gs = gridspec.GridSpec(nrows, 2)
        fig = plt.figure(figsize=(16, nrows*9.0/4.0))
        ax = []
        ax.append( fig.add_subplot(gs[:2,:]) )
        lgd_handles = []
        lgd_labels = []
        ax[0].plot(self.Y, '-k', linewidth=0.5, alpha=0.3)
        if self.saY.N_ungrouped > 0:
            ugXY = np.array(
                sorted(self.saY.get_ungrouped(), key=lambda xy: xy[0])
            )
            ax[0].scatter(
                ugXY[:,0], ugXY[:,1], s=12, marker='o', alpha=0.3,
                facecolor='none', edgecolors='black', linewidths=1
            )
        # END if
        X_disp = []
        Y_chunky_disp = []
        for i in range(len(self.Y)):
            if self.saY.is_grouped(self.saY[i]):
                X_disp.append(i)
                Y_chunky_disp.append(Y_chunky[i])
        ax[0].plot(
            X_disp, Y_chunky_disp, '-', linewidth=2.5,
            color=(254.0/255, 209.0/255, 9.0/255), alpha=0.4
        )
        for stlbl in sorted(self.saY.keys(), reverse=True):
            if len(self.saY.get_group(stlbl)) == 0: continue
            stXY = np.array(
                sorted(self.saY.get_group(stlbl), key=lambda xy: xy[0])
            )
            for i in range(stXY.shape[0]):
                stXY[i,1] = Y_chunky[int(stXY[i,0])]
            out = ax[0].scatter(
                stXY[:,0], stXY[:,1], s=12, marker='o', edgecolors='none',
                facecolors=self.saY.lkup_group_color(stlbl),
                label=stlbl
            )
            lgd_handles.append(out)
            lgd_labels.append(stlbl)
            
            # add histogram of lifetimes
            n_hist = len(ax)-1
            row = n_hist/2 + 2
            col = n_hist%2
            ax.append( fig.add_subplot(gs[row,col]) )
            ax[-1].hist(
                lifetimes[stlbl], color=self.saY.lkup_group_color(stlbl)
            )
            mlife = np.mean(lifetimes[stlbl])
            ax[-1].text(
                0.5, 0.8,
                'Mean lifetime of "{}" is {:0.2f}'.format(stlbl, mlife),
                horizontalalignment='center', transform=ax[-1].transAxes
            )
        # END for
        ax[0].set_title(
            u'{:d}/{:d} ({:0.1%}) unassigned'.format(
                self.saY.N_ungrouped, len(self.Y),
                float(self.saY.N_ungrouped)/len(self.Y)
            )
        )
        ax[0].set_xlim(-0.025*len(self.Y), 1.1*len(self.Y))
        ax[0].set_ylim(-0.1, 1.1)
        ax[0].legend(lgd_handles, lgd_labels)
        fig.tight_layout()
        return fig
    # END plot_summary
    
    @logging_method
    def view_summary_plot(self, *args):
        fig = self.plot_summary()
        fig.show()
        raw_input('press Enter to continue')
        plt.close(fig)
    # END view_summary_plot
    
    @logging_method
    def refresh_graph(self):
        self.ax_main.clear()
        self.ax_zoom.clear()
        self.ax_hist.clear()
        self.ax_zhist.clear()
        #self.ax.set_axisbelow(True)
        
        smY = self._sm_func()
        
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
            for j in range(stXY.shape[0]): stXY[j,1] = smY[int(stXY[j,0])]
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
                ugXY[i,1] = smY[ugXY[i,0]]
            self.ax_main.scatter(
                ugXY[:,0], ugXY[:,1], s=12, marker='o',
                facecolor='none', alpha=0.6,
                edgecolors='black', linewidths=1
            )
        # END if
        
        self.ax_main.set_xlim(-0.05*len(self.Y), 1.05*len(self.Y))
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
        lbl_to_int = {}
        j = 1
        ordered_colors = []
        intlbls = []
        for i in zX:
            if not self.saY.is_grouped(self.saY[i]):
                intlbls.append(0)
                if ordered_colors and ordered_colors[0] != (0,0,0):
                    ordered_colors.insert(0, (0,0,0))
                continue
            # END if
            if self.saY.lkup_group(self.saY[i]) not in lbl_to_int:
                lbl_to_int[self.saY.lkup_group(self.saY[i])] = j
                j += 1
                ordered_colors.append( self.saY.lkup_elem_color(self.saY[i]) )
            # END if
            intlbls.append( lbl_to_int[self.saY.lkup_group(self.saY[i])] )
        # END for
        cmap = ListedColormap(ordered_colors)
            
        # END for
        self.log.writeln('{} points in selection window'.format(len(zX)))
        if 0 < len(zX):
            self.ax_zoom.scatter( zX, zY, s=12, marker='o',
                                  c=intlbls, cmap=cmap, alpha=0.6,
                                  edgecolors='none'
                                )
        # END if
        # line connecting raw data
        if self._line_disp:
            self.ax_zoom.plot(self.Y, '-k', linewidth=0.5, alpha=0.3)
        # set axis limits
        dzX = self.zrng[1][0] - self.zrng[0][0]
        self.ax_zoom.set_xlim(
            self.zrng[0][0]-0.01*dzX, self.zrng[1][0]+0.01*dzX
        )
        #zYmin = min(zY)
        #zYmax = max(zY)
        dzY = self.zrng[1][1] - self.zrng[0][1]
        self.ax_zoom.set_ylim(
            self.zrng[0][1]-0.01*dzY, self.zrng[1][1]+0.01*dzY
        )
        # plot corresponding histogram
        nbins = len(zY)/100
        if nbins < 12: nbins = len(zY)/20
        if nbins < 12: nbins = 12
        try:
            self.ax_zhist.hist(
                zY, bins=nbins,
                edgecolor='none',
                histtype='stepfilled', orientation='horizontal'
            )
        except ValueError:
            pass
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
    
    # Filter management
    #------------------
    def eval_fstk(self, a, b=None, op=None):
        if isinstance(a, list):
            a = self.eval_fstk(*a)
        else:
            a = self.parse_set(a)
        # END if
        if b is None: return a
        if isinstance(b, list):
            b = self.eval_fstk(*b)
        else:
            b = self.parse_set(b)
        # END if
        if op == '|':
            return a | b
        elif op == '&':
            return a & b
        elif op == '-':
            return a - b
        else:
            raise ValueError('{} is not a valid operator'.format(op))
        # END if
    # END eval_fstk
    
    @logging_method
    def filter_reset(self, *args): self._fstk = 'all'
    
    @logging_method
    def filter_set(self, *args):
        try:
            self.parse_set(args[0])
            self._fstk = args[0]
        except Exception:
            self._tmpout += 'Failed to set filter; {} not recognized'.format(
                args[0]
            )
        # END try
    # END filter_set
    
    @logging_method
    def filter_push_union(self, *args):
        for flt in args:
            try:
                self._fstk = [str(flt), self._fstk, '|']
            except Exception:
                pass
            # END try
        # END for
    # END add_filter
    
    @logging_method
    def filter_push_intersection(self, *args):
        for flt in args:
            try:
                self._fstk = [str(flt), self._fstk, '&']
            except Exception:
                pass
            # END try
        # END for
    # END add_filter
    
    def parse_set(self, s):
        if re.search(r'^!', s):
            return self.eval_fstk('all', s[1:], '-')
        elif re.search(r'^g:', s):
            return self.saY.get_group(s[2:])
        elif self.Ypv.contains_key(s):
            return self.Ypv.get_group(s)
        elif s == 'all':
            return set(self.saY)
        # END if
        raise ValueError("{} isn't a valid filter".format(s))
    # END parse_set
    
    # Data management and manipulation
    #---------------------------------
    @logging_method
    def _autosave(self):
        fn = self.cwfile + '.sa.csv'
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
    def set_smoothing_func(self, *args):
        if len(args) == 0:
            self._sm_func = lambda: self.Y
        elif args[0] == 'sa':
            self._sm_func = self.calc_assignment_smoothed_data
        elif args[0] == 'res':
            self._sm_func = self.calc_result_data
        elif args[0] == 'deriv':
            self.calc_derivative_chunked_data()
            self._sm_func = self.get_derivative_chunked_data
        elif args[0] == 'sg':
            try:
                w = int(args[1]) + 1 - (int(args[1])%2)
                p = int(args[2])
            except Exception:
                self._tmpout += 'Error: invalid input arguments'
                return
            # END try
            self._sm_func = lambda: SG_smooth(self.Y, w, p)
        else:
            self._tmpout += 'Error: not a vaild smoothing function'
        # END if
    # END set_smoothing_func
    
    @logging_method
    def get_zdata(self):
        # Get points with user filter applied
        self.log.writeln('filter_stack = '+repr(self._fstk))
        try:
            pnts = self.eval_fstk(self._fstk)
        except Exception:
            self.filter_reset()
            pnts = self.eval_fstk(self._fstk)
        # END try
        self.log.writeln(
            'after user filters, len(pnts) = {}'.format(len(pnts))
        )
        
        # Apply smoothing
        Y = self._sm_func()
        
        # Filter out data outside the selection window
        for iy in set(pnts):
            if self.zrng[0][0] <= iy[0] <= self.zrng[1][0]:
                if self.zrng[0][1] <= Y[iy[0]] <= self.zrng[1][1]:
                    continue
                # END if
            # END if
            pnts.discard(iy)
        # END for
        self.log.writeln(
            'after selection area, len(pnts) = {}'.format(len(pnts))
        )
        
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
    def get_isolated_unassigned_points(self):
        iso_pnts = []
        for pnt in sorted(self.saY.get_ungrouped(), key=lambda p: p[0]):
            if pnt[0] == 0 or pnt[0] == len(self.Y)-1: continue
            if ( self.saY.is_grouped(self.saY[pnt[0]-1]) and
                 self.saY.is_grouped(self.saY[pnt[0]+1])
               ):
                iso_pnts.append(pnt)
            # END if
        # END for
        return iso_pnts
    # END get_isolated_unassigned_points
    
    @logging_method
    def get_isolated_assigned_points(self):
        iso_pnts = []
        for pnt in list(self.saY):
            if pnt[0] == 0 or pnt[0] == len(self.Y)-1: continue
            if ( self.saY.is_grouped(pnt) and
                 (not self.saY.is_grouped(self.saY[pnt[0]-1])) and
                 (not self.saY.is_grouped(self.saY[pnt[0]+1]))
               ):
                iso_pnts.append(pnt)
            # END if
        # END for
        return iso_pnts
    # END get_isolated_unassigned_points
    
    @logging_method
    def open_data_file(self, *args):
        '''UI Prompt for opening files'''
        # look up all valid files
        files = set()
        for fn in os.listdir('.'):
            if re.search(r'^jv', fn):
                files.add(fn)
            elif fimps.mtrx_supported() and re.search(r'\.I\(V\)_mtrx$', fn):
                files.add(fn)
            # END if
        # END for
        for fn in os.listdir('.'):
            if re.search(r'\.sa\.csv~?$', fn):
                files.discard( re.sub(r'\.sa\.csv~?$', '', fn) )
                files.add(fn)
        # END for
        files = sorted(files)
        
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
            if re.search(r'\.sa\.csv~?', fn):
                # open auto-saved file instead
                Y, lbls = fimps.open_autosave(fn)
                self._set_data(Y)
                # set assignments
                for i in range(len(self.Y)):
                    if lbls[i] != '':
                        self.saY.assign(lbls[i], self.saY[i])
                # END for
            elif re.search(r'^jv', fn):
                self._set_data(fimps.open_jv(self.cwfile))
            elif re.search(r'\.I\(V\)_mtrx$', fn):
                self._set_data(fimps.open_mtrx(self.cwfile))
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
            self.open_data_file(*args)
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
        
        self._sm_func = lambda: self.Y
    # END set_data
    
    @logging_method
    def save_assignments(self, save_name=None):
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
        if save_name is None:
            print 'Current working file: {}'.format(self.cwfile)
            save_name = raw_input('File name >>> ')
            self.log.writeln(save_name, t='ui')
            if save_name == '':
                print 'abort saving'
                return
            # END if
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
        fig = self.plot_summary()
        fig.savefig(img_name, dpi=150)
        plt.close(fig)
        self.log.writeln('saved "{}"'.format(img_name))
    # END save_assignments
    
    # Analysis
    #---------
    @logging_method
    def calc_lifetimes(self):
        lifetimes = {lbl: [] for lbl in self.saY.keys()}
        labels = [(i,lbl) for i,lbl in enumerate(self.saY.list_memberships())]
        # purge all unassigned points
        # (this step is an import quirk of how the lifetime is expected to be
        #  calculated by the analyst)
        for i in range(len(labels)):
            i_lbl = labels.pop(0)
            if i_lbl[1] is not None: labels.append(i_lbl)
        # END for
        # count lengths of continuous segments
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
    
    def calc_assignment_smoothed_data(self):
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
                # END for
                m = np.mean(self.Y[i:j])
                for i in range(i, j): Y_chunky[i] = m
            # END if
            i += 1
        # END while
        
        return Y_chunky
    # END calc_assignment_smoothed_data
    
    def calc_result_data(self):
        # Create "chunked data"
        Y_out = np.array(self.Y)
        i = 0
        while i < len(self.Y):
            # start lookahead when arriving at a point that is grouped
            if self.saY.is_grouped(self.saY[i]):
                curr_lbl = self.saY.lkup_group(self.saY[i])
                buff = [self.Y[i],]
                for j in range(i+1, len(self.Y)+1):
                    if j == len(self.Y):
                        break
                    elif not self.saY.is_grouped(self.saY[j]):
                        continue
                    elif self.saY.lkup_group(self.saY[j]) != curr_lbl:
                        break
                    else:
                        buff.append(self.Y[j])
                    # END if
                # END for
                m = np.mean(buff)
                for i in range(i, j):
                    if self.saY.is_grouped(self.saY[i]): Y_out[i] = m
            # END if
            i += 1
        # END while
        
        return Y_out
    # END calc_assignment_smoothed_data
    
    def calc_derivative_chunked_data(self):
        try:
            use_sg = self._yn_input('Use SG filter for derivative? (Y/n) >>>')
            if use_sg:
                w = 0
                while w%2 != 1:
                    w = self._int_input(
                        'window size (odd number of points) >>> ', 3
                    )
                # END while
                p = self._int_input(
                    'polynomial order (0 < p < w-2) >>> ', 0, w-2
                )
                dY = SG_smooth(self.Y, w, p, 1)
            else:
                dY = np.concatenate(
                    ( np.diff(self.Y), [self.Y[-1]-self.Y[-2]] )
                )
        except RuntimeError:
            print 'abort'
            return
        # END try
        
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
            dy_cutoff = abs( self._float_input('Enter cuttoff value >>> ') )
        except RuntimeError:
            print 'abort'
            return
        # END try
        plt.close(fig)
        self.deriv_chunked_Y = np.array(self.Y)
        i = 0
        while i < len(dY):
            # start lookahead when arriving at a point that is "flat-enough"
            if abs(dY[i]) <= dy_cutoff:
                for j in range(i+1, len(dY)+1):
                    if j == len(dY): break
                    elif dy_cutoff < abs(dY[j]): break
                # END for
                m = np.mean(self.Y[i:j])
                for i in range(i, j): self.deriv_chunked_Y[i] = m
            # END if
            i += 1
        # END while
    # END calc_derivative_chunked_data
    
    def get_derivative_chunked_data(self): return self.deriv_chunked_Y
    
    # State assignment methods
    #-------------------------
    @logging_method
    def assign_selection(self, stlbl=None, *args):
        zX, _ = self.get_zdata()
        if len(zX) == 0: return
        
        if stlbl is None:
            stlbl = self.next_autoname()
        elif not re.search(r'^\w+$', stlbl):
            self._tmpout += 'No action; "{}" not a vaild group name'
            return
        # END if
        self.saY.commit()
        for i in zX:
            self.saY.assign(stlbl, self.saY[i])
        # END for
    # END assign_selection
    
    @logging_method
    def reorder_labels(self, *args):
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
    def toggle_lock(self, *args): self.saY.toggle_reassign()
    
    @logging_method
    def delete_state(self, *stlbls):
        for lbl in stlbls:
            if lbl not in self.saY.keys(): return
            self.saY.commit()
            self.saY.disband(lbl)
        # END for
    # END delete_state
    
    @logging_method
    def delete_all_states(self, *args):
        self.saY.commit()
        for k in self.saY.keys(): self.saY.disband(k)
    # END delete_state
    
    @logging_method
    def unassign_selection(self, *args):
        zX, _ = self.get_zdata()
        if len(zX) == 0: return
        
        self.saY.commit()
        self.saY.unassign( *[self.saY[i] for i in zX] )
        self._tmpout += 'unassigned {} points'.format(len(zX))
    # END unassign_selection
    
    @logging_method
    def unassign_short_lives(self, max_life, *stlbls):
        if not stlbls: return
        
        max_life = int(max_life)
        lives = []
        labels = [ (i,lbl) for i,lbl in
                   enumerate(self.saY.list_memberships())
                 ]
        
        # purge all unassigned points
        # (this step is an import quirk of how the lifetime is expected to be
        #  calculated by the analyst)
        for i in range(len(labels)):
            i_lbl = labels.pop(0)
            if i_lbl[1] is not None: labels.append(i_lbl)
        # END for
        
        # record starting points and lengths of continuous segments
        i0 = labels[0][0]
        for i in range(1, len(labels)):
            # check for a break in continuity
            if labels[i-1][1] != labels[i][1]:
                lives.append( (i0, labels[i-1][0]-i0 + 1, labels[i-1][1]) )
                i0 = labels[i][0]
            # END if
        # END for
        lives.append( (i0, labels[-1][0]-i0 + 1, labels[-1][1]) )
        
        # unassign all segments that are shorted than max_life
        # (i.e. unassign all lives that are too short)
        zchk = set(self.get_zdata()[0])
        for wg in stlbls:
            chopping_block = []
            for i0, t, g in lives:
                if g == wg and t < max_life:
                    for i in range(i0, i0+t+1):
                        if i in zchk: chopping_block.append(self.saY[i])
                    # END for
                # END if
            # END for
            if chopping_block:
                self.saY.commit()
                # .unassign will skip over points that are already unassigned,
                # therefore the output will have an accurate count of the number
                # of point actually unassigned
                chopping_block = self.saY.unassign(*chopping_block)
                self._tmpout += '{} points unassigned from "{}"'.format(
                    len(chopping_block), wg
                )
                
                # show the user the changes
                fig, ax = plt.subplots(1,1)
                XY = np.array(chopping_block)
                ax.scatter(
                    XY[:,0], XY[:,1], s=12, marker='x', c='red'
                )
                ax.plot(self.Y, '-k', linewidth=0.5, alpha=0.3)
                ax.set_title(
                    '{} points unassigned from "{}"'.format( len(chopping_block),
                                                             wg
                                                           )
                )
                fig.show()
                raw_input('press enter to continue')
                plt.close(fig)
            # END if
        # END for
    # END unassign_short_lives
    
    @logging_method
    def unassign_breaks(self, wg):
        # TODO: make me!
        pass
    # END unassign_breaks
    
    @logging_method
    def assign_singles(self):
        N_assigned = 0
        self.saY.commit()
        iso_pnts = self.get_isolated_unassigned_points()
        for pnt in iso_pnts:
            lpnt = self.saY[pnt[0]-1]
            rpnt = self.saY[pnt[0]+1]
            if abs(pnt[1] - lpnt[1]) < abs(pnt[1] - rpnt[1]):
                self.saY.assign(self.saY.lkup_group(lpnt), pnt)
                N_assigned += 1
            else:
                self.saY.assign(self.saY.lkup_group(rpnt), pnt)
                N_assigned += 1
            # END if
        # END for
        
        if N_assigned == 0: self.saY.rollback()
        self._tmpout += '{} points assigned'.format(N_assigned)
    # END assign_singles
    
    @logging_method
    def unassign_outliers( self, stlbl=None, w=None, zcutoff=None,
                           rounds=None, *args
                         ):
        '''Unassign points that are normal distribution outliers'''
        while True:
            if stlbl in self.saY.keys(): break
            stlbl = raw_input('State to work on >>>')
            self.log.writeln(stlbl, t='ui')
            if stlbl.lower() == 'x': return
        # END while
        
        stXY = np.array(
            sorted(self.saY.get_group(stlbl), key=lambda xy: xy[0])
        )
        try:
            w = abs(int(w))
            if not (3 <= w <= len(self.Y) and w%2 == 1):
                raise TypeError()
            zcutoff = abs(float(zcutoff))
            rounds = abs(int(rounds))
        except TypeError:
            w = 0
            while w%2 != 1:
                w = self._int_input(
                        'point window for avg. '+
                        '(2 < odd number < {}) >>> '.format(stXY.shape[0]),
                        3, stXY.shape[0]
                    )
            # END while
            # plot stdev bars for user
            sgY = SG_smooth(stXY[:,1], w, 1)
            stdY = np.sqrt( np.mean( (stXY[:,1]-sgY)**2 ) )
            self.ax_main.plot(stXY[:,0], sgY, '-k', linewidth=1)
            self.ax_main.plot(stXY[:,0], sgY+stdY, '--k', linewidth=1)
            self.ax_main.plot(stXY[:,0], sgY-stdY, '--k', linewidth=1)
            self.ax_main.plot(stXY[:,0], sgY+2*stdY, '--k', linewidth=0.5)
            self.ax_main.plot(stXY[:,0], sgY-2*stdY, '--k', linewidth=0.5)
            self.ax_main.plot(stXY[:,0], sgY+3*stdY, '--k', linewidth=0.5)
            self.ax_main.plot(stXY[:,0], sgY-3*stdY, '--k', linewidth=0.5)
            self.fig.canvas.draw()
            zcutoff = abs( self._float_input('z-score cutoff >>> ', 0) )
            rounds = abs( self._int_input('numer of rounds >>> ', 1) )
        # END try
        
        self.saY.commit()
        N_unassigned = 0
        selected_pnts = set( self.get_zdata()[0] )
        if len(selected_pnts) == 0: return
        for rnd in range(rounds):
            # Calculate state distribution estimates (i.e. mean & stdev)
            stXY = np.array(
                sorted(self.saY.get_group(stlbl), key=lambda xy: xy[0])
            )
            sgY = SG_smooth(stXY[:,1], w, 1)
            stdY = np.sqrt( np.mean( (stXY[:,1]-sgY)**2 ) )
            Z = abs(stXY[:,1] - sgY) / stdY
            # END for
            
            N_u = 0
            for j in range(stXY.shape[0]):
                if int(stXY[j,0]) in selected_pnts and zcutoff < Z[j]:
                    self.saY.unassign( self.saY[int(stXY[j,0])] )
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
    def undo_assignment(self, *args): self.saY.rollback()
    
    @logging_method
    def concat_states(self, st1=None, *other_stlbls):
        if st1 is None: return
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
    def multiassign_1D_KMeans(self, Nst=None, *args):
        '''Assign multiple states using only the displayed y-values'''
        zX, zY = self.get_zdata()
        if len(zX) == 0: return
        
        try:
            Nst = abs(int(Nst))
            if Nst < 2: raise ValueError()
        except Exception:
            Nst = None
         #END try
        if Nst is None:
            try:
                Nst = self._int_input('Number of states >>> ', 2)
            except RuntimeError:
                print 'abort'
                return
            # END try
        # END if
        
        # Fit a K-means model
        classifier = MiniBatchKMeans(n_clusters=Nst, n_init=12)
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
    def multiassign_2D_KMeans(self):
        '''Assign multiple states using only the displayed y-values'''
        zX, zY = self.get_zdata()
        zX = np.array(zX, dtype='int32')
        if len(zX) == 0: return
        
        # show preview of chunky data and fitting space
        fig, ax = plt.subplots(2,1)
        ax[0].scatter(
            zX, zY,
            s=12, marker='o', edgecolors='none', facecolor='black'
        )
        ax[0].plot(self.Y, '-k', linewidth=0.5, alpha=0.3)
        ax[1].scatter(
            self.Y, zY,
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
        classifier = MiniBatchKMeans(n_clusters=Nst, n_init=12)
        data = np.array(
            [(self.Y[zX[j]], zY[j]) for j in range(len(zX))]
        )
        classifier.fit(data)
        
        # show preview of results
        cm = plt.get_cmap('Dark2')
        fig, ax = plt.subplots(2,1)
        ax[0].scatter(
            zX, zY,
            s=12, marker='o', edgecolors='none',
            c=classifier.labels_, cmap=cm
        )
        ax[0].plot(self.Y, '-k', linewidth=0.5, alpha=0.3)
        ax[1].scatter(
            self.Y, zY,
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
    # END multiassign_2D_KMeans
    
    @logging_method
    def multiassign_compound(self, *args):
        try:
            use_sg = self._yn_input('Use SG filter for derivative? >>>')
            if use_sg:
                w = 0
                while w%2 != 1:
                    w = self._int_input(
                        'window size (odd number of points) >>> ', 3
                    )
                # END while
                p = self._int_input(
                    'polynomial order (0 < p < w-2) >>> ', 0, w-2
                )
                dY = SG_smooth(self.Y, w, p, 1)
            else:
                dY = np.concatenate(
                    ( np.diff(self.Y), [self.Y[-1]-self.Y[-2]] )
                )
        except RuntimeError:
            print 'abort'
            return
        # END try
        
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
    def expand_assignment(self, stlbl=None, *args):
        # TODO: add NN smoothed curve during z-cutoff input to make it clear
        # that the cutoff is based off the moving average
        while True:
            if stlbl in self.saY.keys(): break
            stlbl = raw_input('State to expand >>>')
            self.log.writeln(stlbl, t='ui')
            if stlbl.lower() == 'x': return
        # END while
        
        stXY = np.array(
            sorted(self.saY.get_group(stlbl), key=lambda xy: xy[0])
        )
        zchk = set(self.get_zdata()[0])
        
        print 'enter "X" at anytime to abort command'
        # Get input arguments from user
        N_buff = 0
        while N_buff%2 != 1:
            N_buff = self._int_input( 'Size of point buffer >>> ',
                                      1, stXY.shape[0]
                                    )
        # END while
        
        # plot stdev bars for user
        buff = stXY[-N_buff:,1]
        # END for
        #m = np.mean(buff)
        #std = np.std(buff)
        #X = [stXY[-N_buff,0], stXY[-1,0]]
        #X = [0, len(self.Y)]
        #M = m*np.ones(2)
        #self.ax_main.plot(X, M, '-k', linewidth=1)
        #self.ax_main.plot(X, M+std, '--k', linewidth=1)
        #self.ax_main.plot(X, M-std, '--k', linewidth=1)
        #self.ax_main.plot(X, M+2*std, '--k', linewidth=0.5)
        #self.ax_main.plot(X, M-2*std, '--k', linewidth=0.5)
        #self.ax_main.plot(X, M+3*std, '--k', linewidth=0.5)
        #self.ax_main.plot(X, M-3*std, '--k', linewidth=0.5)
        sgY = SG_smooth(stXY[:,1], N_buff, 0)
        stdY = np.sqrt( np.mean( (stXY[:,1]-sgY)**2 ) )
        self.ax_main.plot(stXY[:,0], sgY, '-k', linewidth=1)
        self.ax_main.plot(stXY[:,0], sgY+stdY, '--k', linewidth=1)
        self.ax_main.plot(stXY[:,0], sgY-stdY, '--k', linewidth=1)
        self.ax_main.plot(stXY[:,0], sgY+2*stdY, '--k', linewidth=0.5)
        self.ax_main.plot(stXY[:,0], sgY-2*stdY, '--k', linewidth=0.5)
        self.ax_main.plot(stXY[:,0], sgY+3*stdY, '--k', linewidth=0.5)
        self.ax_main.plot(stXY[:,0], sgY-3*stdY, '--k', linewidth=0.5)
        self.fig.canvas.draw()
        
        print ( 'error lines on plot do not denote the exact region of ' +
                 'acceptance, but are approximately correct'
               )
        
        z_cuttoff = abs( self._float_input('z-score cutoff >>> ', 0) )
        
        # drop save point in history before making assignments
        self.saY.commit()
        N_assigned = 0
        
        # capture moving from right to left (<--)
        buff = []
        for i in range(len(self.Y))[::-1]:
            if self.saY.lkup_group(self.saY[i]) == stlbl:
                buff.insert(0, self.Y[i])
                if len(buff) > N_buff:
                    buff.pop(-1)
            elif ( len(buff) == N_buff and
                   abs(self.Y[i]-np.mean(buff))/np.std(buff) < z_cuttoff and
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
            if self.saY.lkup_group(self.saY[i]) == stlbl:
                buff.insert(0, self.Y[i])
                if len(buff) > N_buff: buff.pop(-1)
            elif ( len(buff) == N_buff and 
                   abs(self.Y[i]-np.mean(buff))/np.std(buff) < z_cuttoff and
                   i in zchk
                 ):
                    self.saY.assign(stlbl, self.saY[i])
                    N_assigned += 1
                    buff.insert(0, self.Y[i])
                    buff.pop(-1)
                # END if
            # END if
        # END for
        
        if N_assigned == 0: self.saY.rollback()
        self._tmpout += '{} points assigned to {}'.format(N_assigned, stlbl)
    # END expand_assignment
    
    @logging_method
    def simple_pickup(self, stlbl=None, *other_stlbls):
        while True:
            if stlbl in self.saY.keys(): break
            stlbl = raw_input('State to expand >>>')
            self.log.writeln(stlbl, t='ui')
            if stlbl.lower() == 'x': return
        # END while
        
        # add all points that are sandwiched in x and y, and flag all points
        # that are sandwiched in x but not y
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
                # END if
                i = j
            # END if
            i += 1
        # END while
        
        self._tmpout += 'added {} points to state "{}"'.format( N_pickedup,
                                                                stlbl
                                                              )
        self.log.writeln('add {} points to "{}"'.format(N_pickedup, stlbl))
        
        if other_stlbls: self.simple_pickup(*other_stlbls)
    # END simple_pickup
# END App


