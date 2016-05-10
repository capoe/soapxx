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
options.set('kernel.adaptor', 'generic')                    # <- pull here (generic/global-generic)
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

for struct in structures:
    print struct.label    
    kernelpot.acquire(struct, 1.)
    print kernelpot.IX.shape
np.savetxt('kernelpot.ix.txt', kernelpot.IX)



# Load XX from text file
# Run PCA (normalize, diagonalize)
# Project structures onto PCA => plot
# Reconstruct eigenstructures (unnormalize, load into kernel, vary centers, optimize)

def PCAtrain(Xtrain):
    #### IMPLEMENT HERE
    #### Note that for the eigen decomposition you can use the numpy function 'np.linalg.eigh'
    mean = X_train.mean(0)
    std = X_train.std(0)

    print(mean.shape)
    print(X_train.shape)
    
    Xtrain = (Xtrain - mean) / std

    Sigma = np.dot(Xtrain.T,Xtrain)/Xtrain.shape[0]
    singular_values, pc_vectors = np.linalg.eigh(Sigma)
    
    idx = singular_values.argsort()[::-1]
    singular_values = singular_values[idx]
    pc_vectors = pc_vectors[:,idx]
    
    return pc_vectors, singular_values, mean, std


















