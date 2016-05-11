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

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
idx_eig_target = 199
N = 5vmd
select_struct_label = 'config_C_%02d.xyz' % N
exclude_pids_rnd = [1]
exclude_centers = [ i for i in range(2, N+1) ]
opt_pids = [ i for i in range(2, N+1) ]
x_array_file = 'out.pca.xeigenvecs.txt'
x_array_file = 'out.kernelpot.ix.txt'
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# OPTIONS
options = soap.Options()
options.excludeCenters(['H'])
options.excludeTargets(['H'])
options.excludeCenterIds(exclude_centers)
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
    for c in soap.tools.ase_load_all('configs_n_carbons')
]
struct_dict = {}
for struct in structures:
    print struct.label
    struct_dict[struct.label] = struct

# LOAD EIGENSTRUCTURES
print "Load fingerprints from '%s'" % x_array_file
IX_eig = np.loadtxt(x_array_file)
x1 = IX_eig[0]
x2 = IX_eig[4]
print np.dot(x1,x1), np.dot(x1,x2), np.dot(x2,x2)

# SELECT STRUCTURE, X_TARGET
print "Target eigenstructure with index", idx_eig_target
print "Exclude centers", exclude_centers
print "Select label", select_struct_label 
X_target = IX_eig[idx_eig_target]
struct = struct_dict[select_struct_label]
options.excludeCenterIds(exclude_centers)

# SETUP KERNEL (=> BASIS, KERNELPOT WITH ADAPTOR)
basis = soap.Basis(options)
kernelpot = KernelPotential(options)
kernelpot.importAcquire(np.array([X_target]), -1.) # <- attractive
sc = 2.
# RANDOMIZE INITIAL CONFIG
positions_0 = [ part.pos for part in struct ]
positions_0_rnd = random_positions(struct, exclude_pid=exclude_pids_rnd, b0=-1.*sc, b1=1.*sc)

print "Positions after initial randomization"
for part in struct:
    print part.id, part.pos

# EXTRACT POSITIONS (OPTIMIZATION TARGETS)
opt_pids = np.array(opt_pids)
opt_pidcs = opt_pids-1
positions = np.array(positions_0_rnd)
positions_short = positions[opt_pidcs]
positions_short = positions_short.flatten()
print "Optimization arguments (initial)"
for pos in positions_short.reshape((len(opt_pids),3)):
    print pos

raw_input('...')

# TRAJECTORY LOGGER
ofs = TrajectoryLogger('opt.xyz')
ofs.logFrame(struct)

# OPTIMIZER INTERFACE
import scipy.optimize
f = evaluate_energy
x0 = positions_short
fprime = evaluate_energy_gradient
args = (struct, kernelpot, opt_pids, False, ofs)
positions_opt = scipy.optimize.fmin_cg(f, x0, fprime=fprime, args=args, gtol=1e-6)

#positions_opt = scipy.optimize.minimize(f, x0, args=args, method='CG', jac=fprime, tol=1e-5)  
#positions_opt = scipy.optimize.fmin_cg(f, x0, args=args, gtol=1e-5)
#scipy.optimize.minimize(f, x0, args=args, method='CG', jac=fprime, tol=1e-99)   
#scipy.optimize.minimize(f, x0, args=args, method='CG', tol=1e-5)

ofs.close()

print "Optimized positions:", positions_opt


kernelpot.acquire(struct, -1.)
np.savetxt('opt_X.txt', kernelpot.IX[-1])











