# -*- encoding: UTF-8 -*-
'''GroupedList type classes

    Documentation needed

    List of classes:
        GroupedList
        ColorCodedList
    List of functions:
        test
'''

from transactionable_objects import Transaction

#==============================================================================
class GroupedList(Transaction):
    '''List with disjoint groupings
    
    Maintains an ordered list of objects, and groups them into disjoint sets.
    Some elements of this list may be ungrouped.  All elements must be unique,
    if they are not the GroupedList will not function correctly.  Elements are
    considered unique if they have different hash values.
    
    Instantiation Args:
        iterobj: a Sequence with unique, hashable elements
    Instance Attributes:
        N_groups (int): Number groups, same as len(G.keys())
    Operations:
        S in G: tests if X (object) is in G's (GroupedList) array
        G[group_name]: returns group (set) keyed by group_name
        for X in G: iterate over all the objects in G (GroupedList)
        len(G): number of objects in G (GroupedList)
    Instance Methods:
        TODO make list here
    '''
    
    def __init__(self, iterobj=[], reassign=True):
        self._objlst = list(iterobj)
        unique_check = {}
        for x in self._objlst:
            if x in unique_check:
                raise ValueError( 'Sequence of input objects contains two ' +
                                  'different objects that have the same '+
                                  'hash value'
                                )
            unique_check[x] = True
        # END for
        del unique_check
        self._groups = {}
        self._ungrouped = set(iterobj)
        self._grp_rlkup = {}
        self._reassign = reassign
    # END __init__
    
    @classmethod
    def concatenate(cls, A, B):
        new_gl = cls()
        new_gl.extend(A)
        for agrp in A.keys():
            new_gl.assign(A.get_group(agrp), agrp)
        new_gl.extend(B)
        for bgrp in B.keys():
            new_gl.assign(B.get_group(bgrp), bgrp)
        return new_gl
    # END extend
    
    @staticmethod
    def pprint(gl):
        print '['
        for x in gl:
            print '    {} ({}),'.format(repr(x), repr(gl.lkup_group(x)))
        print ']'
    # END pprint
    
    # Magic Methods
    #--------------
    def __contains__(self, X): return X in self._objlst
    
    def __getitem__(self, i): return self._objlst[i]
    
    def __iter__(self):
        for obj in self._objlst:
            yield obj
    # END __iter__
    
    def __len__(self): return len(self._objlst)
    
    # Property methods
    #-----------------
    @property
    def N_groups(self): return len(self._groups.keys())
    
    @property
    def N_ungrouped(self): return len(self._ungrouped)
    
    # Instance methods
    #-----------------
    def assign(self, grpname, *someobjs):
        '''Assign some existing elements to a group
        
        Args:
            someobjs (list(obj)): list of objects that are in the GroupedList
                that will be assigned to grpname
            grpname (obj): Existing or new key to define the assignment
                destination
        Returns:
            None
        '''
        # Validate that all objects to be assigned are elements in _objlst
        for x in someobjs:
            if x not in self._ungrouped and x not in self._grp_rlkup:
                raise ValueError('Cannot assign non-element to a group')
            # END if
        # END for
        
        # Make sure that the members are released from any current memberships
        # before they are assigned to the requested group.
        someobjs = list(someobjs)
        for _ in range(len(someobjs)):
            x = someobjs[0]
            if x in self._grp_rlkup and self._reassign:
                # element is currently grouped, needs to be released
                former_grp = self._grp_rlkup[x]
                self._groups[former_grp].discard(x)
                if len(self._groups[former_grp]) == 0:
                    self._groups.pop(former_grp)
            elif x in self._grp_rlkup and not self._reassign:
                # element is currently grouped AND should not be reassigned
                someobjs.pop(0)
                continue
            else:
                # element not grouped
                self._ungrouped.discard(x)
            # END try
            someobjs.append(someobjs.pop(0))
        # END for
        
        if grpname not in self._groups:
            # new sub-group
            self._groups[grpname] = set(someobjs)
        else:
            # adding members to an existing group
            for x in someobjs:
                self._groups[grpname].add(x)
        # END if
        
        # update reverse look-up dict
        for x in someobjs:
            self._grp_rlkup[x] = grpname
    # END assign
    
    def overassign(self, grpname, *someobjs):
        '''Assign some existing elements to a group, IGNORING the assignment
           lock
        
        Args:
            someobjs (list(obj)): list of objects that are in the GroupedList
                that will be assigned to grpname
            grpname (obj): Existing or new key to define the assignment
                destination
        Returns:
            None
        '''
        reassign = self._reassign
        self._reassign = True
        self.assign(grpname, *someobjs)
        self._reassign = reassign
    # END overassign
    
    def underassign(self, grpname, *someobjs):
        '''Assign some existing elements to a group, IGNORING the assignment
           lock
        
        Args:
            someobjs (list(obj)): list of objects that are in the GroupedList
                that will be assigned to grpname
            grpname (obj): Existing or new key to define the assignment
                destination
        Returns:
            None
        '''
        reassign = self._reassign
        self._reassign = False
        self.assign(grpname, *someobjs)
        self._reassign = reassign
    # END overassign
    
    def append(self, obj):
        '''Append a new object to the end
        '''
        if obj in self._grp_rlkup or obj in self._ungrouped:
            raise ValueError('Cannot append a duplicate element')
        self._objlst.append(obj)
        self._ungrouped.add(obj)
        # END if
    # END append
    
    def disband(self, grpname):
        '''Delete a grouping and return all of it's former members
        '''
        # Remove all current group members from rlkup
        for x in self._groups[grpname]:
            self._grp_rlkup.pop(x)
        # Move all current group member from group to _ungrouped
        disbanded_members = self._groups.pop(grpname)
        self._ungrouped |= disbanded_members
        return disbanded_members
    # END disband
    
    def extend(self, someobjs):
        '''Extend a Sequence of new objects to the end
        '''
        for x in someobjs:
            if x in self._grp_rlkup or x in self._ungrouped:
                raise ValueError('Cannot append a duplicate element')
            # END if
        # END for
        self._objlst.extend(someobjs)
        for x in someobjs:
            self._ungrouped.add(x)
        # END for
    # END extend
    
    def get_group(self, grpname): return set(self._groups[grpname])
    
    def get_list(self): return list(self._objlst)
    
    def get_notgroup(self, grpname):
        out = set(self._ungrouped)
        for gn in self._groups:
            if grpname == gn:
                continue
            out |= self._grouops[gn]
        return out
    # END get_notgroup
    
    def get_reassign(self): return self._reassign
    
    def get_ungrouped(self): return set(self._ungrouped)
    
    def index(self, obj): return self._objlst.index(obj)
    
    def insert(self, i, obj):
        '''Insert a new object at index, i
        '''
        if obj in self._grp_rlkup or obj in self._ungrouped:
            raise ValueError('Cannot append a duplicate element')
        self._objlst.insert(i, obj)
        self._ungrouped.add(obj)
        # END if
    # END insert
    
    def is_grouped(self, obj): return obj not in self._ungrouped
    
    def list_groups(self, sort=False):
        if sort:
            return [set(self._groups[k]) for k in sorted(self._groups.keys())]
        else:
            return [set(self._groups[k]) for k in self._groups.keys()]
        # END if
    # END list_groups
    
    def list_memberships(self):
        memberships = []
        for x in self._objlst:
            memberships.append(self._grp_rlkup.get(x))
        return memberships
    # END list_memberships
    
    def lkup_group(self, x, default=None):
        return self._grp_rlkup.get(x, default)
    
    def keys(self): return self._groups.keys()
    
    def pop(self, i):
        '''Remove and return element at index, i, and it's former group key
        '''
        if self[i] in self._ungrouped:
            self._ungrouped.discard(self[i])
            return self._objlst.pop(i), None
        else:
            former_grp = self._grp_rlkup.pop(self[i])
            self._groups[former_grp].discard(self[i])
            if len(self._groups[former_grp]) == 0:
                self._groups.pop(former_grp)
            return self._objlst.pop(i), former_grp
        # END if
    # END pop
    
    def pop_ungrouped(self):
        out_objs = []
        i = 0
        N = len(self._objlst)
        while i < N:
            x = self._objlst.pop(0)
            if x not in self._grp_rlkup:
                # Toss out ungrouped element
                out_objs.append(x)
            else:
                # Keep grouped element
                self._objlst.append(x)
            # END if
            i += 1
        # END for
        return out_objs
    # END pop_ungrouped
    
    def remove(self, obj):
        '''TODO
        '''
        pass
    # END remove
    
    def set_reassign(self, reassign):
        if not isinstance(reassign, bool):
            raise TypeError('Argument must be bool')
        self._reassign = reassign
    # END set_reassign
    
    def sort(self, *args, **kwargs): self._objlst.sort(*args, **kwargs)
    
    def sort_by_group(self, *args, **kwargs):
        self._objlst.sort(*args, **kwargs)
        self._objlst.sort(key=lambda x: self._grp_rlkup.get(x))
    # END sort_by_group
    
    def toggle_reassign(self):
        self._reassign = not self._reassign
        return self._reassign
    # END toggle_reassign
    
    def unassign(self, *someobjs):
        disbanded_objs = []
        for x in someobjs:
            try:
                former_grp = self._grp_rlkup.pop(x)
                self._groups[former_grp].discard(x)
                disbanded_objs.append(x)
                self._ungrouped.add(x)
                if len(self._groups[former_grp]) == 0:
                    self._groups.pop(former_grp)
            except KeyError:
                # an ungrouped or non-member element was included,
                # just ignore it
                pass
            # END try
        # END for
        return disbanded_objs
    # END unassign
# END GroupedList

#==============================================================================
class ColoredList(GroupedList):
    '''List with color-coded, disjoint groupings
    '''
    #TODO: update _color_lkup after unassignment
    # color palatte created by user FreeSpirit Fashion
    # http://www.colourlovers.com/palette/2941817/FSBD_Palette_837
    colors = (
        (213.0/255, 45.0/255, 36.0/255), #Chinese Tea Cups (red)
        (6.0/255, 137.0/255, 204.0/255), #Popping Turquoise
        (163.0/255, 8.0/255, 254.0/255), #Purple Fusion
        (124.0/255, 174.0/255, 3.0/255), #Mossy Green
        (212.0/255, 30.0/255, 157.0/255), #pink
        (250.0/255, 107.0/255, 30.0/255), #Orange
        (4.0/255, 189.0/255, 149.0/255), #Teal
        (42.0/255, 163.0/255, 48.0/255), #dark Green
        #(254.0/255, 209.0/255, 9.0/255), #Solarized Yellow
    )
    
    def __init__(self, *args):
        super(ColoredList, self).__init__(*args)
        
        # colors will be pulled from the beginning (i=0) and moved to the end
        # (i=-1)
        self._colors = list(ColoredList.colors)
        self._color_lkup = {}
    # END __init__
    
    def assign(self, grpname, *someobjs):
        groups_before = set(self._groups.keys())
        super(ColoredList, self).assign(grpname, *someobjs)
        if groups_before != set(self._groups.keys()):
            nextcolor = self._colors.pop(0)
            self._colors.append(nextcolor)
            self._color_lkup[grpname] = nextcolor
        # END if
    # END assign
    
    def cycle_group_color(self, grpname):
        nextcolor = self._colors.pop(0)
        self._colors.append(nextcolor)
        self._color_lkup[grpname] = nextcolor
    # END cycle_group_color
    
    def list_colors(self):
        c = []
        for x in self._objlist:
            c.append(self._color_lkup[self._grp_rlkup[x]])
        return c
    # END list_colors
    
    def lkup_elem_color(self, x): return self._color_lkup[self._grp_rlkup[x]]
    
    def lkup_group_color(self, grpname):
        return self._color_lkup[grpname]
    # END lkup_group_color
    
    def reset_color_cycle(self):
        self._colors = list(ColoredList.colors)
    # END reset_color_cycle
# END ColoredList

#==============================================================================
def test():
    gl1 = GroupedList((1, 2, 3, 'one', 'two', 'three', (3.14, 1.618)))
    print 'made a GroupedList'
    print 'GroupedList.pprint(gl1) -->'
    GroupedList.pprint(gl1)
    print ''
    
    print 'Making Groups'
    gl1.assign((1,2,3), 'numbers')
    gl1.assign(('one', 'two', 'three'), 'strings')
    gl1.assign(((3.14, 1.618),), 'fun pairs of irrational numbers')
    GroupedList.pprint(gl1)
    print 'gl1.N_groups = {}'.format(gl1.N_groups)
    print ''
    
    print 'testing operations'
    print '1 in gl1 = {}'.format(1 in gl1)
    print 'gl1[4] = {}'.format(gl1[4])
    print 'len(gl1) = {}'.format(len(gl1))
    print ''
    
    print 'Object Methods'
    print 'gl1.pop(1) = {}'.format(gl1.pop(1))
    GroupedList.pprint(gl1)
    print 'gl1.append(-45)'
    gl1.append(-45)
    GroupedList.pprint(gl1)
    print 'gl1.disband("numbers")'
    gl1.disband("numbers")
    GroupedList.pprint(gl1)
    print 'gl1.extend([7.8, 9.0, "ten"])'
    gl1.extend([7.8, 9.0, "ten"])
    GroupedList.pprint(gl1)
    print 'gl1.get_group("strings") = {} {}'.format(
        repr(gl1.get_group("strings")),
        repr( type( gl1.get_group("strings") ) )
    )
    print 'gl1.index(1) = {}'.format(gl1.index(1))
    print 'updating assignments...'
    gl1.assign((1,3,-45,9.0,7.8), 'numbers')
    gl1.assign((gl1[-1],), 'strings')
    GroupedList.pprint(gl1)
    print 'gl1.insert(3, 8+2j)'
    gl1.insert(3, 8+2j)
    GroupedList.pprint(gl1)
    print 'gl1.pop_ungrouped() = {}'.format(gl1.pop_ungrouped())
    GroupedList.pprint(gl1)
    print 'gl1.list_groups() = {}'.format(gl1.list_groups())
    print 'gl1.list_groups(sort=True) = {}'.format(gl1.list_groups(sort=True))
    print 'gl1.unassign(((3.14, 1.618),))'
    gl1.unassign( ((3.14, 1.618),) )
    GroupedList.pprint(gl1)
    print 'gl1.keys() = {}'.format(gl1.keys())
    print 'gl1.sort_by_group()'
    gl1.sort_by_group()
    GroupedList.pprint(gl1)
    print 'gl1.sort()'
    gl1.sort()
    GroupedList.pprint(gl1)
    print ''
    
    print 'All tests complete'
# END test