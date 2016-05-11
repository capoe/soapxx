#! /usr/bin/env python
import soap
import soap.tools
soap.silence()

import os
import sys
import numpy as np
import logging
import io

from momo import osio, endl, flush
from kernel import KernelPotential, KernelFunctionFactory, TrajectoryLogger
from kernel import perturb_positions, random_positions
from kernel import evaluate_energy, evaluate_energy_gradient


# KERNEL OPTIONS
options = soap.Options()
options.set('kernel.type', 'dot')
options.set('kernel.delta', 1.)
options.set('kernel.xi', 4.)
method = 'kernelpca'

# KERNEL FUNCTION
kernelfct = KernelFunctionFactory[options.get('kernel.type')](options)

# DISTANCE MATRIX
ix = np.loadtxt('in.kernelpot.ix.txt')
kmat = kernelfct.computeBlock(ix, return_distance=False)
distmat = kernelfct.computeBlock(ix, return_distance=True)


# PROJECTION
import sklearn.manifold
import sklearn.decomposition
if method == 'mds':
    # MDS
    # http://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html#sklearn.manifold.MDS
    mds = sklearn.manifold.MDS(
        n_components=3,
        metric=False,
        verbose=1,
        n_init=10,
        dissimilarity='precomputed')
    positions = mds.fit_transform(distmat)
elif method == 'isomap':
    isomap = sklearn.manifold.Isomap(
        n_neighbors=5,
        n_components=3,
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

# SAVE POSITIONS
np.savetxt('out.positions.txt', positions)




















