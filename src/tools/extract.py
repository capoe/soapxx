import numpy as np
import numpy.linalg

from momo import osio, endl, flush

class Xnklab(object):
    def __init__(self, atomic, types_global):
        self.S = len(types_global)
        self.N = atomic.basis.N
        self.L = atomic.basis.L
        S = self.S
        N = self.N
        L = self.L

        self.X = np.zeros((S*N*N,S*(L+1)))
        self.types_global = types_global

        types_atomic = atomic.getTypes()
        S_atomic = len(types_atomic)
        for i in range(S_atomic):
            a = types_atomic[i]
            sa = types_global.index(a)
            for j in range(S_atomic):
                b = types_atomic[j]
                sb = types_global.index(b)
                xnklab = atomic.getPower(a,b).array
                r1 = sa*N*N
                r2 = r1+N*N
                c1 = sb*(L+1)
                c2 = c1+(L+1)
                self.X[r1:r2,c1:c2] = xnklab.real
                #print a, b, sa, sb
                #print self.X
                #raw_input('...')
        #print "DONE"
        return
    def reduce(self):
        dim_linear_xnklab = (self.S*self.S+self.S)/2*(self.N*self.N*(self.L+1))
        dim_linear_xnkl = self.N*self.N*(self.L+1)
        self.X_linear = np.zeros((dim_linear_xnklab))
        for sa in range(self.S):
            for sb in range(sa,self.S):
                # Find slice in matrix X
                r1 = sa*self.N*self.N
                r2 = r1+self.N*self.N
                c1 = sb*(self.L+1)
                c2 = c1+(self.L+1)
                xnkl_linear = self.X[r1:r2,c1:c2].reshape((dim_linear_xnkl))
                # Map to vector
                sab = self.S*sa - (sa*sa-sa)/2 + (sb-sa) # Linear summation over upper triagonal section
                idx0 = sab*dim_linear_xnkl
                idx1 = (sab+1)*dim_linear_xnkl
                self.X_linear[idx0:idx1] = xnkl_linear
                #print sa, sb, sab, idx0, idx1
                #print self.X_linear
                #raw_input('...')
        return self.X_linear

def extract_xnklab(atomic, types, types_pairs):
    assert False
    types = atomic.getTypes()
    types = sorted(types)
    n_types = len(types)
    N = atomic.basis.N
    L = atomic.basis.L
    xnklab = []
    for i in range(n_types):
        for j in range(i, n_types):
            a = types[i]
            b = types[j]
            xnklab = xnklab + atomic.getPower(t1,t2).array
    return xnklab
