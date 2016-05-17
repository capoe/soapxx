#! /usr/bin/env python
import numpy as np
import sklearn.manifold
import sklearn.decomposition

def dimred_matrix(method, kmat=None, distmat=None, outfile=None, ix=None, symmetrize=False, prj_dimension=2):
    if symmetrize:
        kmat = 0.5*(kmat+kmat.T)
        dmat = 0.5*(dmat+dmat.T)
    if method == 'mds':
        # MDS
        # http://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html#sklearn.manifold.MDS
        mds = sklearn.manifold.MDS(
            n_components=prj_dimension,
            metric=True,
            verbose=1,
            n_init=10,
            dissimilarity='precomputed')
        positions = mds.fit_transform(distmat)
    elif method == 'isomap':
        isomap = sklearn.manifold.Isomap(
            n_neighbors=5,
            n_components=prj_dimension,
            eigen_solver='auto',
            path_method='auto',
            neighbors_algorithm='auto')
        positions = isomap.fit_transform(ix)
    elif method == 'kernelpca':
        kernelpca = sklearn.decomposition.KernelPCA(
            n_components=None,
            kernel='precomputed',
            eigen_solver='auto',
            max_iter=None,
            remove_zero_eig=True)
        positions = kernelpca.fit_transform(kmat)
    else: raise NotImplementedError(method)
    if outfile: np.savetxt(outfile, positions)
    return positions

def dimred(kernelpot, method, outfile, symmetrize=False):
    ix = kernelpot.IX
    kmat = kernelpot.kernelfct.computeBlockDot(ix, return_distance=False)
    distmat = kernelpot.kernelfct.computeBlockDot(ix, return_distance=True)
    positions = dimred_matrix(method, kmat, distmat, outfile, ix, symmetrize=symmetrize)
    return positions















