import numpy as np

IX = np.loadtxt('out.kernelpot.buffer.ix.txt')
K = np.loadtxt('out.kernelpot.buffer.kernel.txt')
D = (abs(1.-K))**0.5

Dsym = 0.5*(D+D.T)
Ksym = 0.5*(K+K.T)

from dimred import dimred_matrix
dimred_matrix('mds', kmat=K, distmat=Dsym, outfile='tmp.txt')
#dimred_matrix('kernelpca', kmat=K, distmat=Dsym, outfile='tmp.txt')
#dimred_matrix('isomap', kmat=Ksym, distmat=Dsym, outfile='tmp.txt', ix=IX, symmetrize=False)
