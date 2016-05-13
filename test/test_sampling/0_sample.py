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
from dimred import dimred

logging.basicConfig(
    format='[%(asctime)s] %(message)s',
    datefmt='%I:%M:%S',
    level=logging.ERROR)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
select_struct_label = 'config.xyz'
exclude_centers = []
adaptor_type = 'global-generic'
kernel_type = 'dot-harmonic'
alpha = +1.
mu = 0.9
sc = 0.1 # 1.5 # 0.1
average_energy = True
lj_sigma = 0.02 # 0.00001 # 0.02
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# NOTE adaptor_type = 'generic' => exclude center if not constrained in optimization

def sample_random_x0(struct, exclude_pids_rnd, opt_pids, b0, b1):
    # Sample
    x0 = [ part.pos for part in struct ]
    x0 = random_positions(struct, exclude_pid=exclude_pids_rnd, b0=b0, b1=b1)
    x0 = np.array(x0)    
    # Short-list
    opt_pids = np.array(opt_pids)
    opt_pidcs = opt_pids-1
    x0 = x0[opt_pidcs]
    x0 = x0.flatten()    
    return x0

class LJRepulsive(object):
    def __init__(self, sigma=lj_sigma):
        self.sigma = sigma            
    def computeEnergy(self, struct):
        coords = np.zeros((struct.n_particles, 3))
        for idx, part in enumerate(struct):
            coords[idx,:] = part.pos            
        E = 0.
        for i in range(struct.n_particles):
            for j in range(i):
                r = np.sqrt(np.sum((coords[i, :] - coords[j, :]) ** 2))
                E += (self.sigma/r)**12
        return E
    def computeGradient(self, struct):
        coords = np.zeros((struct.n_particles, 3))
        for idx, part in enumerate(struct):
            coords[idx,:] = part.pos            
        grad = np.zeros(coords.shape)
        for i in range(struct.n_particles):
            for j in range(i):
                dr = coords[i, :] - coords[j, :]
                r = np.sqrt(np.sum(dr**2))
                g = 12./self.sigma*(self.sigma/r)**13
                grad[i, :] += -g * dr / r
                grad[j, :] += g * dr / r
        return grad

def evaluate_energy_local(positions, structure, kernelpot, opt_pids, verbose=False, ofs=None, average=False):
    if verbose: print "Energy"
    # Impose positions
    pid_pos = positions.reshape((opt_pids.shape[0],3))
    for pidx, pid in enumerate(opt_pids):
        pos = pid_pos[pidx,:]
        particle = structure.getParticle(pid)        
        particle.pos = pos        
    for part in structure:
        if verbose: print part.id, part.type, part.pos
    # Evaluate energy function
    energy = kernelpot.computeEnergy(structure)
    if average: energy /= kernelpot.nConfigs()
    # Evaluate additional potentials
    lj = LJRepulsive()
    energy_add = lj.computeEnergy(struct)
    energy += energy_add
    # Log
    if ofs: ofs.logFrame(structure)
    if verbose: print energy
    print energy
    return energy

def evaluate_energy_gradient_local(positions, structure, kernelpot, opt_pids, verbose=False, ofs=None, average=False):
    if verbose: print "Forces"
    # Adjust positions
    pid_pos = positions.reshape((opt_pids.shape[0],3))
    for pidx, pid in enumerate(opt_pids):
        pos = pid_pos[pidx,:]
        particle = structure.getParticle(pid)        
        particle.pos = pos        
    for part in structure:
        if verbose: print part.id, part.type, part.pos
    # Evaluate forces
    forces = kernelpot.computeForces(structure)
    gradients = -1.*np.array(forces)
    if average:
        gradients = gradients/kernelpot.nConfigs()
    # Evaluate additional potentials
    lj = LJRepulsive()
    gradients_add = lj.computeGradient(struct)
    gradients = gradients + gradients_add
    # Short-list
    opt_pidcs = opt_pids-1
    gradients_short = gradients[opt_pidcs]
    gradients_short = gradients_short.flatten()
    if verbose: print gradients_short
    #forces[2] = 0. # CONSTRAIN TO Z=0 PLANE
    return gradients_short

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
options.set('kernel.adaptor', adaptor_type)                 # <- pull here (generic/global-generic)
options.set('kernel.type', kernel_type)
options.set('kernel.delta', 1.)
options.set('kernel.xi', 2.)                                # <- pull here (1., ..., 4., ...)
options.set('kernel.mu', mu)
# Cube files
options.set('densitygrid.N', 20)
options.set('densitygrid.dx', 0.15)


# LOAD STRUCTURES
structures = [ 
    soap.tools.setup_structure_ase(c.config_file, c.atoms)
    for c in soap.tools.ase_load_all('in.configs')
]
struct_dict = {}
for struct in structures:
    print struct.label
    struct_dict[struct.label] = struct

# SELECT STRUCTURE
struct = struct_dict[select_struct_label]
options.excludeCenterIds(exclude_centers)

# SETUP KERNEL (=> BASIS, KERNELPOT WITH ADAPTOR)
kernelpot_buffer = KernelPotential(options)
kernelpot_buffer.acquire(struct, alpha)
kernelpot = KernelPotential(options)
kernelpot.acquire(struct, alpha) # <- repulsive
np.savetxt('out.x_ref.txt', kernelpot.IX)

# MANAGE EXCLUSIONS (RANDOMIZATION, OPTIMIZATION)
exclude_pids_rnd = [1]
opt_pids = [ i for i in range(2, struct.n_particles+1) ]
opt_pids = np.array(opt_pids)
#if adaptor_type == 'global-generic':
#    exclude_pids_rnd = []
#    opt_pids = [ i for i in range(1,struct.n_particles+1) ]
#elif adaptor_type == 'generic':
#    ...

# TRAJECTORY LOGGER
ofs_opt = TrajectoryLogger('out.opt.xyz')
ofs_opt.logFrame(struct)
ofs_acquired = TrajectoryLogger('out.acq.xyz')
ofs_acquired.logFrame(struct)
ofs_warned = TrajectoryLogger('out.warn.xyz')

# OPTIMIZER INTERFACE
import scipy.optimize
f = evaluate_energy_local
fprime = evaluate_energy_gradient_local
args = (struct, kernelpot, opt_pids, False, ofs_opt, average_energy)

print "Select label:    ", select_struct_label
print "Exclude centers: ", exclude_centers
print "Adaptor type:    ", adaptor_type

pos_opt_list = []
e_opt_list = []




for mu in [ 0.9, 0.7, 0.5, 0.3 ]:
    kernelpot.kernelfct.mu = mu   
    
    
    n_iterations = 10
    skip_warnings = False

    iter_count = 0
    while iter_count < n_iterations:
        # RANDOMIZE INITIAL CONFIG & SETUP X0
        x0 = sample_random_x0(struct, exclude_pids_rnd, opt_pids, -1.*sc, +1.*sc)    
        print "Positions after initial randomization"
        for part in struct:
            print part.id, part.pos
        print "Optimization arguments (initial)"
        for pos in x0.reshape((len(opt_pids),3)):
            print pos
        print type(x0)
        

        x0_opt, f_opt, n_calls, n_grad_calls, warnflag = scipy.optimize.fmin_cg(
            f=f, 
            x0=x0, 
            fprime=fprime, 
            args=args, 
            gtol=1e-6,
            full_output=True)
        if warnflag > 0:
            print "Warnflag =", warnflag, " => try again."
            ofs_warned.logFrame(struct)
            if skip_warnings: continue
        
        print "Acquire structure ..."
        ofs_acquired.logFrame(struct)
        #kernelpot.acquire(struct, alpha)
        kernelpot_buffer.acquire(struct, alpha)
        iter_count += 1
        print "Kernel size = %d" % kernelpot.nConfigs()
        
        e, prj = kernelpot.computeEnergy(struct, return_prj_mat=True)
        print prj
        
        pos = np.array([ part.pos for part in struct ])
        print "Positions"
        print pos
        pos_opt_list.append(pos)
        e_opt_list.append(f_opt)
        
        #raw_input('...')

# CLOSE LOGGER
ofs_opt.close()
ofs_acquired.close()
ofs_warned.close()

np.savetxt('out.e_opt_list.txt', np.array(e_opt_list))
np.savetxt('out.kernelpot.kernel.txt', kernelpot.computeDotKernelMatrix())
np.savetxt('out.kernelpot.buffer.kernel.txt', kernelpot_buffer.computeDotKernelMatrix())
np.savetxt('out.kernelpot.buffer.ix.txt', kernelpot_buffer.IX)
np.savetxt('out.kernelpot.ix.txt', kernelpot.IX)

if kernelpot.nConfigs() > 1: dimred(kernelpot, 'mds', 'out.dimred.txt', symmetrize=True)
if kernelpot_buffer.nConfigs() > 1: dimred(kernelpot_buffer, 'mds', 'out.dimred.buffer.txt', symmetrize=True)




















