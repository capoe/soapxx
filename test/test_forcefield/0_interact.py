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
from kernel import KernelAdaptorFactory, KernelFunctionFactory, TrajectoryLogger
from simspace import SimSpaceTopology, SimSpaceNode, SimSpacePotential
from simspace import LJRepulsive
from simspace import optimize_node
from dimred import dimred

logging.basicConfig(
    format='[%(asctime)s] %(message)s',
    datefmt='%I:%M:%S',
    level=logging.ERROR)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
select_struct_label = 'config.xyz'
exclude_centers = []
adaptor_type = 'global-generic'
kernel_type = 'dot' # 'dot-harmonic'
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

# DEFINE INTERACTION TEMPLATES
pot_options_dot = soap.Options()
pot_options_dot.set('kernel.adaptor', 'global-generic')
pot_options_dot.set('kernel.type', 'dot')
pot_options_dot.set('kernel.alpha', -1.)
pot_options_dot.set('kernel.delta', 1.)
pot_options_dot.set('kernel.xi', 2.)
pot_options_dot.set('kernel.mu', 0.5)
pot_options_harm = soap.Options()
pot_options_harm.set('kernel.adaptor', 'global-generic')
pot_options_harm.set('kernel.type', 'dot-harmonic')
pot_options_harm.set('kernel.alpha', +1.)
pot_options_harm.set('kernel.delta', 1.)
pot_options_harm.set('kernel.xi', 2.)
pot_options_harm.set('kernel.mu', 0.5)
pot_options_lj = soap.Options()
pot_options_lj.set('potentials.lj.sigma', 0.02)

# TOPOLOGY
top = SimSpaceTopology(options)

# DEFINE NODES
node_root = top.createNode(struct)

"""
node_leaf = top.createNode(struct)
top.summarize()

# META-INTERACTIONS
potentials = []
pot = SimSpacePotential(node_leaf, node_root, pot_options_harm)
potentials.append(pot)

# SELF-INTERACTIONS
potentials_self = []
pot = LJRepulsive(node_leaf, pot_options_lj)
potentials_self.append(pot)

# OPTIMIZE
x0 = node_leaf.randomizePositions(scale=sc, zero_pids=[0], compute=True)
optimize_node(node_leaf, potentials, potentials_self, np.array([1,2]), x0)
top.writeData()
"""

for n in range(6):
    node_leaf = top.createNode(struct)    
    node_root = top.nodes[0]
    
    # POTENTIALS
    potentials = []
    potentials_self = []
    
    # DOT-HARMONIC
    pot_options_harm.set('kernel.adaptor', 'global-generic')
    pot_options_harm.set('kernel.type', 'dot-harmonic')
    pot_options_harm.set('kernel.alpha', +10.)
    pot_options_harm.set('kernel.delta', 1.)
    pot_options_harm.set('kernel.xi', 2.)
    pot_options_harm.set('kernel.mu', 0.95)
    pot = SimSpacePotential(node_leaf, node_root, pot_options_harm)
    potentials.append(pot)
    print "<dot-harm, ij>", node_leaf.id, node_root.id
    
    # DOT
    for j in range(1, len(top.nodes)-1):
        pot_options_dot.set('kernel.adaptor', 'global-generic')
        pot_options_dot.set('kernel.type', 'dot-harmonic')
        pot_options_dot.set('kernel.alpha', +1.)
        pot_options_dot.set('kernel.delta', 1.)
        pot_options_dot.set('kernel.xi', 2.)
        pot_options_dot.set('kernel.mu', 0.95)
        pot = SimSpacePotential(node_leaf, top.nodes[j], pot_options_dot)
        potentials.append(pot)
        print "<dot, ij>", node_leaf.id, top.nodes[j].id
    
    # LJ
    pot_options_lj.set('potentials.lj.sigma', 0.02)
    pot = LJRepulsive(node_leaf, pot_options_lj)
    potentials_self.append(pot)
    
    if n == 0:
        x0 = node_leaf.randomizePositions(scale=sc, zero_pids=[0], compute=True)
    else:
        x0 = np.array([ part.pos for part in node_leaf.structure ])
    optimize_node(node_leaf, potentials, potentials_self, np.array([1,2]), x0)


top.writeData()
    








