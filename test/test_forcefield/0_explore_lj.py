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
from simspace import SimSpaceTopology, SimSpaceNode, SimSpacePotential, SimSpaceTree, SimSpaceJoint
from simspace import LJRepulsive
from simspace import optimize_node, multi_optimize_node
from dimred import dimred

logging.basicConfig(
    format='[%(asctime)s] %(message)s',
    datefmt='%I:%M:%S',
    level=logging.ERROR)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
select_struct_label = 'config.xyz'
exclude_centers = []
adaptor_type = 'global-generic'
kernel_type = 'dot'
sc = 0.1
lj_sigma = 0.02
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# OPTIONS
options = soap.Options()
options.excludeCenters(['H'])
options.excludeTargets(['H'])
options.excludeCenterIds(exclude_centers)
options.excludeTargetIds([])
options.set('spectrum.gradients', True)
options.set('spectrum.2l1_norm', False)                     # <- pull here (True/False)
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
options.set('kernel.adaptor', adaptor_type)                 # <- pull here (generic/global-generic)
options.set('kernel.type', kernel_type)
options.set('kernel.delta', 1.)
options.set('kernel.xi', 2.)                                # <- pull here (1., ..., 4., ...)
options.set('densitygrid.N', 20)
options.set('densitygrid.dx', 0.15)

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

pot_options_dotlj = soap.Options()
pot_options_dotlj.set('kernel.adaptor', 'global-generic')
pot_options_dotlj.set('kernel.type', 'dot-lj')
pot_options_dotlj.set('kernel.alpha', +1.)
pot_options_dotlj.set('kernel.delta', 1.)
pot_options_dotlj.set('kernel.xi', 2.)
pot_options_dotlj.set('kernel.lj_sigma', 0.125) # 0.1 # 0.2 too large => TODO Better force capping required
pot_options_dotlj.set('kernel.lj_eps_cap', 0.001)

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

# CREATE TREE
osio.cd('out.files')
tree = SimSpaceTree(options)
tree.seed(struct)

N_generations = 2
N_spawn = [3,3]

N_generations = 4
N_spawn = [2,2,2,2]

N_generations = 9
N_spawn = [ 2 for n in range(N_generations) ]
authorize_spawn_fct = lambda joint: joint.node.energy <= 0.0

for n in range(N_generations):
    gen_id = n+1
    gen_prefix = 'out.gen_%d' % gen_id
    print "Spawn generation %d" % gen_id    
    
    tree.top.computePotentialEnergy()
    new_joints = tree.spawn(N_spawn[n], authorize_spawn_fct)
    tree.summarize()
    tree.writeParentChildPairs('%s.tree.txt' % gen_prefix)
    
    # POTENTIALS
    tree.clearPotentials()
    tree.addPairPotential(pot_options_dotlj)
    
    # OPTIMIZE 1st GENERATION
    nodes_opt = [ joint.node for joint in tree.generations[-1] ]
    x0_opt = []
    for node in nodes_opt:
        x0 = node.perturbPositions(scale=0.1, zero_pids=[], compute=True)
        x0_opt.append(x0)
    x0_opt = np.array(x0_opt).flatten()
    multi_optimize_node(nodes_opt, x0_opt)
    
    # OUTPUT
    tree.top.writeData(prefix=gen_prefix)
    
    #raw_input('...')

osio.root()







