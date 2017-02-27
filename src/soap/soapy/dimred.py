#! /usr/bin/env python
import numpy as np
import sklearn.manifold
import sklearn.decomposition

def dimred_matrix(method, kmat=None, distmat=None, outfile=None, ix=None, symmetrize=False, prj_dimension=2):
    dmat = distmat
    if symmetrize:
        if type(kmat) != type(None):
            kmat = 0.5*(kmat+kmat.T)
        if type(dmat) != type(None):
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
    elif method == 'diffmap':
        positions = diffusion_map(
            kmat=kmat,
            n_components=prj_dimension)
    else: raise NotImplementedError(method)
    if outfile: np.savetxt(outfile, positions)
    return positions

def dimred(kernelpot, method, outfile, symmetrize=False):
    ix = kernelpot.IX
    kmat = kernelpot.kernelfct.computeBlockDot(ix, return_distance=False)
    distmat = kernelpot.kernelfct.computeBlockDot(ix, return_distance=True)
    positions = dimred_matrix(method, kmat, distmat, outfile, ix, symmetrize=symmetrize)
    return positions

def diffusion_map(kmat, n_components, alpha=0.5):
    # Normalize kernel matrix 
    # (correct for non-unity diagonal components)
    kmat_diag = kmat.diagonal()
    kmat_norm = kmat/kmat_diag
    kmat_norm = kmat_norm.T/kmat_diag
    np.fill_diagonal(kmat_norm, 1.)
    #print kmat_norm

    # Rescale kernel
    D0 = np.sum(kmat_norm, axis=1)
    D0_diag = np.zeros(kmat_norm.shape)
    np.fill_diagonal(D0_diag, D0)
    kmat_norm = kmat_norm/D0**alpha
    kmat_norm = kmat_norm.T/D0**alpha
    #print kmat_norm

    # Form Markov matrix
    D1 = np.sum(kmat_norm, axis=1)
    kmat_norm = (kmat_norm/D1).T
    #print kmat_norm
    #print np.sum(kmat_norm, axis=1)

    # Decompose
    import scipy.sparse
    evals, evecs = scipy.sparse.linalg.eigs(kmat_norm, n_components+1, which="LM")
    order = evals.argsort()[::-1]
    evals = evals[order]
    evecs = evecs.T[order].T
    #print evals
    #print evecs

    # Project
    return kmat_norm.dot(evecs[:,1:])















