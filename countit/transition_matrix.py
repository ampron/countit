# -*- encoding: UTF-8 -*-

""" TransMatrix class containing module

"""

# built-in modules
import csv

class TransMatrix(object):
    '''Transition Matrix representation of a directional graph of nodes
    
    Class methods:
        from_csv(file_name)
        copy(tranmtrx_obj)
    Instantiation Args:
        M (list(list)): List of lists of integers. All edge weights will be
            cast as integers and be unsigned by an absolute value
            transformation
        labels (list(str)): These strings will be the names of each node
    Instance methods:
        append_node(node_label)
        insert_node(i, node_label)
        remove_node(node_label_or_index)
        keys()
        save_csv(file_name)
    Supported operations:
        (TransMatrix objects will be denoted as capital letters, A, B, C, ...)
        C = A + B
        A += B
        len(A)
        str(A)
        label in A
        A[i,j]
        A[node_lbl_1, node_lbl_2]
        A[node_lbl, j]
        A[i,j] = n
        A[node_lbl_1, node_lbl_2] = n
        A[node_lbl, j] = n
    '''
    
    @classmethod
    def from_csv(cls, file_name):
        with open(file_name, 'rb') as f:
            M = [r for r in csv.reader(f)]
            labels = M.pop(0)
        return cls(M, labels)
    
    @classmethod
    def copy(cls, obj): return cls(obj._M, obj._labels)
    
    @classmethod
    def sum(cls, all_obj):
        Y = cls.copy(all_obj[0])
        for X in all_obj[1:]:
            Y += X
        return Y
    
    def __init__(self, M, labels):
        # M should be a sequence of sequences, with square dimensions
        N = len(M)
        self._M = [[abs(int(M[i][j])) for j in range(N)] for i in range(N)]
        # labels should be a sequence of strings that will label each node
        self._labels = [str(lbl) for lbl in labels]
        self._ilkup = {strlbl: i for i, strlbl in enumerate(self._labels)}
    
    def __getitem__(self, tail_head):
        # tail_head should be a length 2 tuple with either integer indices
        # of string labels for elements
        i, j = tail_head
        if i in self._ilkup:
            return self[self._ilkup[i], j]
        elif j in self._ilkup:
            return self[i, self._ilkup[j]]
        else:
            return self._M[i][j]
    
    def __setitem__(self, tail_head, x):
        # tail_head should be a length 2 tuple with either integer indices
        # of string labels for elements
        i, j = tail_head
        if i in self._ilkup:
            self[self._ilkup[i], j] = x
        elif j in self._ilkup:
            self[i, self._ilkup[j]] = x
        else:
            self._M[i][j] = abs(int(x))
    
    def __contains__(self, strlbl):
        if strlbl in self._ilkup:
            return True
        else:
            return False
    
    def __len__(self): return len(self._M)
    
    def __str__(self):
        max_label_len = max([len(lbl) for lbl in self._labels])
        strings = []
        for i, lbl in enumerate(self._labels):
            strings.append('{1:<{0:d}s} '.format(max_label_len, lbl))
            strings.append(repr(self._M[i]))
            strings.append('\n')
        return ''.join(strings[:-1])
    
    def append_node(self, label):
        label = str(label)
        if label in self._ilkup:
            raise ValueError('Cannot have duplicate label on new node')
        self._labels.append(label)
        self._ilkup[label] = len(self._labels)-1
        for row in self._M:
            row.append(0)
        self._M.append( [0 for _ in range(len(self._M[-1]))] )
    
    def insert_node(self, i, label):
        i = int(i)
        if i < 0: i += len(self._M)
        label = str(label)
        if label in self._ilkup:
            raise ValueError('Cannot have duplicate label on new node')
        self._labels.insert(i, label)
        self._ilkup[label] = i
        for row in self._M:
            row.insert(i, 0)
        self._M.insert( i, [0 for _ in range(len(self._M[-1]))] )
    
    def remove_node(self, x):
        if x in self._ilkup:
            label = x
            i = self._ilkup[label]
        else:
            i = int(x)
            if i < 0: i += len(self._M)
            label = self._labels[i]
        self._labels.pop(i)
        self._ilkup.pop(label)
        self._M.pop(i)
        for row in self._M:
            row.pop(i)
    
    def keys(self): return list(self._labels)
    
    def __iadd__(self, other):
        extra_nodes = set(other._ilkup.keys()) - set(self._ilkup.keys())
        for label in extra_nodes: self.append_node(label)
        for tail in self._labels:
            for head in self._labels:
                self[tail, head] += other[tail, head]
        return self
    
    def __add__(self, other):
        new = TransMatrix.copy(self)
        new += other
        return new
    
    def __radd__(self, other):
        new = TransMatrix.copy(other)
        new += self
        return new
    
    def save_csv(self, save_name):
        with open(save_name, 'w') as f:
            f.write(','.join(['"{}"'.format(s) for s in self._labels]))
            f.write('\n')
            for row in self._M:
                f.write( ','.join([str(n) for n in row]) )
                f.write('\n')
                # END for
            # END for
        # END with

