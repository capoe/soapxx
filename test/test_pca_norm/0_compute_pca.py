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
from kernel import KernelPotential, TrajectoryLogger
from kernel import perturb_positions, random_positions
from kernel import evaluate_energy, evaluate_energy_gradient
from pca import PCA, IPCA

# Load XX from text file
# Run PCA (normalize, diagonalize)
# Project structures onto PCA => plot
# TODO Reconstruct eigenstructures (unnormalize, load into kernel, vary centers, optimize)

logging.basicConfig(
    format='[%(asctime)s] %(message)s',
    datefmt='%I:%M:%S',
    level=logging.ERROR)

# >>>>>>>>>>>>>>>>
norm_mean = True
norm_std = False
adaptor_type = 'global-generic'
# <<<<<<<<<<<<<<<<

# OPTIONS
options = soap.Options()
options.excludeCenters(['H'])
options.excludeTargets(['H'])
options.excludeCenterIds([])
options.excludeTargetIds([])
# Spectrum
options.set('spectrum.gradients', True)
options.set('spectrum.2l1_norm', False)                     # <- pull here (True/False)
# Basis
options.set('radialbasis.type', 'gaussian')
options.set('radialbasis.mode', 'adaptive')
options.set('radialbasis.N', 9)
options.set('radialbasis.sigma', 0.5)
options.set('radialbasis.integration_steps', 15)
options.set('radialcutoff.Rc', 6.8)
options.set('radialcutoff.Rc_width', 0.5)
options.set('radialcutoff.type', 'heaviside')
options.set('radialcutoff.center_weight', 0.5)
options.set('angularbasis.type', 'spherical-harmonic')
options.set('angularbasis.L', 6)
# Kernel
options.set('kernel.adaptor', adaptor_type)                 # <- pull here (generic/global-generic)
options.set('kernel.type', 'dot')
options.set('kernel.delta', 1.)
options.set('kernel.xi', 4.)                                # <- pull here (1., ..., 4., ...)
# Cube files
options.set('densitygrid.N', 20)
options.set('densitygrid.dx', 0.15)

# LOAD STRUCTURES
structures = [ 
    soap.tools.setup_structure_ase(c.config_file, c.atoms)
    for c in soap.tools.ase_load_all('configs')
]

# SETUP KERNEL (=> BASIS, KERNELPOT WITH ADAPTOR)
basis = soap.Basis(options)
kernelpot = KernelPotential(options)

# FILL KERNEL
generate = True
if generate:
    for struct in structures:
        print struct.label    
        kernelpot.acquire(struct, 1., label=struct.label)
        print kernelpot.IX.shape
    np.savetxt('out.kernelpot.ix.txt', kernelpot.IX)
    ofs = open('out.kernelpot.labels.txt', 'w')
    for idx,label in enumerate(kernelpot.labels):
        ofs.write('%3d %s\n' % (idx,label))
    ofs.close()
else:
    IX = np.loadtxt('out.kernelpot.ix.txt')
    kernelpot.importAcquire(IX, 1.)

# KERNEL PCA
pca = PCA()
pca.compute(kernelpot.IX, normalize_mean=norm_mean, normalize_std=norm_std)
#pca = IPCA()
#pca.compute(IX, normalize_mean=False, normalize_std=False)


# =============================
# CHECK COMPONENT NORMALIZATION
# =============================

ones_vec = np.zeros(567)
ones_vec.fill(1.)
np.savetxt('out.pca.unnorm.txt', pca.unnormBlock(ones_vec))

# =================
# INDEX CUTOFF SCAN
# =================

K_full = None
idx_ref = 567
sel = 7
for idx_cutoff in [idx_ref, 500, 400, 300, 200, 100, 50, 40, 30, 20, 10, 5, 4, 3, 2, 1]:
    print idx_cutoff
    I_eigencoeffs = pca.expandBlock(kernelpot.IX, idx_cutoff)
    IX_recon = pca.reconstructBlock(I_eigencoeffs)
    K = IX_recon.dot(IX_recon.T)
    eigen_K = I_eigencoeffs.dot(I_eigencoeffs.T)    
    if idx_cutoff == idx_ref:
        K_full = K
    K_slice = K[sel,:]
    K_slice_full = K_full[sel,:]    
    def stretch_K(K):
        k_min = np.min(K)
        k_max = np.max(K)
        return (K-k_min)/(k_max-k_min)    
    K_slice = stretch_K(K_slice)
    K_slice_full = stretch_K(K_slice_full)    
    KK = np.array([K_slice,K_slice_full]).T
    np.savetxt('out.pca.kernel_recon_cutoff_%03d.txt' % (idx_cutoff), KK)

# ========================
# EXPAND KERNEL STRUCTURES
# ========================

I_eigencoeffs = pca.expandBlock(kernelpot.IX)
np.savetxt('out.pca.eigencoeffs.txt', I_eigencoeffs)

# =================
# SAVE EIGENVECTORS
# =================

X_eig_0 = pca.reconstructEigenvec(0)
X_eig_1 = pca.reconstructEigenvec(1)
print np.dot(X_eig_0, X_eig_0)
print np.dot(X_eig_0, X_eig_1)
print np.dot(X_eig_1, X_eig_1)

X_eig_vecs = pca.reconstructEigenvecs()
np.savetxt('out.pca.xeigenvecs.txt', X_eig_vecs)





