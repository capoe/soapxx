#! /usr/bin/env python
import sys
import numpy as np

IX = np.loadtxt('out.top.ix.txt')

K = np.loadtxt('out.top.kernelmatrix.txt')
D = (abs(1.-K))**0.5
#D = (abs(1.-K))
#D = (abs(2.-2*K))*0.5

Dsym = 0.5*(D+D.T)
Ksym = 0.5*(K+K.T)

np.savetxt('out.top.distmatrix.txt', Dsym)

from dimred import dimred_matrix

method = sys.argv[1]

prj_dimension = 2

if method == 'mds':
	dimred_matrix('mds', kmat=K, distmat=Dsym, outfile='tmp.txt', prj_dimension=prj_dimension)
elif method == 'kernelpca':
    dimred_matrix('kernelpca', kmat=K, distmat=Dsym, outfile='tmp.txt', prj_dimension=prj_dimension)
elif method == 'isomap':
    dimred_matrix('isomap', kmat=Ksym, distmat=Dsym, outfile='tmp.txt', ix=IX, symmetrize=False, prj_dimension=prj_dimension)
else:
	assert False

