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
kernel_type = 'dot' # 'dot-harmonic'
alpha = +1.
mu = 0.9
sc = 0.1 # 1.5 # 0.1
average_energy = True
lj_sigma = 0.02 # 0.00001 # 0.02
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# NOTE adaptor_type = 'generic' => exclude center if not constrained in optimization

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

# CREATE ROOT NODE
node_R = top.createNode(struct)

# CREATE FIRST GENERATION
N_1st = 8
nodes_1st = []
for n in range(N_1st):
    node = top.createNode(struct)
    nodes_1st.append(node)

# ROOT o~o 1st
pot_options_harm.set('kernel.adaptor', 'global-generic')
pot_options_harm.set('kernel.type', 'dot-harmonic')
pot_options_harm.set('kernel.alpha', +1.)
pot_options_harm.set('kernel.delta', 1.)
pot_options_harm.set('kernel.xi', 2.)
pot_options_harm.set('kernel.mu', 0.95)
for node in nodes_1st:    
    node.createPotential(node_R, pot_options_harm)

# 1st <-> 1st
pot_options_dot.set('kernel.adaptor', 'global-generic')
pot_options_dot.set('kernel.type', 'dot')
pot_options_dot.set('kernel.alpha', +0.05)
pot_options_dot.set('kernel.delta', 1.)
pot_options_dot.set('kernel.xi', 2.)
pot_options_dot.set('kernel.mu', 0.80)
for i in range(len(nodes_1st)):
    for j in range(i+1, len(nodes_1st)):
        node_i = nodes_1st[i]
        node_j = nodes_1st[j]
        node_i.createPotential(node_j, pot_options_dot)
        node_j.createPotential(node_i, pot_options_dot)

# OPTIMIZE 1st GENERATION
nodes_opt = nodes_1st
x0_opt = []
for node in nodes_opt:
    x0 = node.perturbPositions(scale=0.1, zero_pids=[], compute=True)
    x0_opt.append(x0)
x0_opt = np.array(x0_opt).flatten()
multi_optimize_node(nodes_opt, x0_opt)




# CREATE 2nd GENERATION
N_2nd = 8
nodes_2nd = {}
for par_node in nodes_1st:
    nodes_2nd[par_node.id] = []
    for n in range(N_2nd):
        child_node = top.createNode(par_node.structure)
        nodes_2nd[par_node.id].append(child_node)

nodes_2nd_list = []
for par_id, child_nodes in nodes_2nd.iteritems():
    nodes_2nd_list = nodes_2nd_list + child_nodes

ofs = open('out.tree.txt', 'w')
for parent_id, child_nodes in nodes_2nd.iteritems():
    for child_node in child_nodes:
        #ofs.write("Parent-node-ID %d : Child-node-ID %d\n" % (parent_id, child_node.id))
        ofs.write("%d %d\n" % (parent_id, child_node.id))
ofs.close()

# PARENT-1st o~o CHILD-2nd
pot_options_harm.set('kernel.adaptor', 'global-generic')
pot_options_harm.set('kernel.type', 'dot-harmonic')
pot_options_harm.set('kernel.alpha', +1.)
pot_options_harm.set('kernel.delta', 1.)
pot_options_harm.set('kernel.xi', 2.)
pot_options_harm.set('kernel.mu', 0.95)
for parent_id, child_nodes in nodes_2nd.iteritems():
    par_node = top.nodes[parent_id-1]
    for child_node in child_nodes:
        child_node.createPotential(par_node, pot_options_harm)
        par_node.createPotential(child_node, pot_options_harm) # new

# ROOT <> 2nd
pot_options_dot.set('kernel.adaptor', 'global-generic')
pot_options_dot.set('kernel.type', 'dot')
pot_options_dot.set('kernel.alpha', +0.1) # <- Pull
pot_options_dot.set('kernel.delta', 1.)
pot_options_dot.set('kernel.xi', 2.)
pot_options_dot.set('kernel.mu', 0.80)
for parent_id, child_nodes in nodes_2nd.iteritems():
    for child_node in child_nodes:
        child_node.createPotential(node_R, pot_options_dot)

# CHILD-2nd <> SIBLING-2nd
pot_options_dot.set('kernel.adaptor', 'global-generic')
pot_options_dot.set('kernel.type', 'dot')
pot_options_dot.set('kernel.alpha', +0.05) # <- Pull
pot_options_dot.set('kernel.delta', 1.)
pot_options_dot.set('kernel.xi', 2.)
pot_options_dot.set('kernel.mu', 0.80)
for parent_id, child_nodes in nodes_2nd.iteritems():
    for i in range(len(child_nodes)):
        for j in range(i+1, len(child_nodes)):
            node_i = child_nodes[i]
            node_j = child_nodes[j]
            node_i.createPotential(node_j, pot_options_dot)
            node_j.createPotential(node_i, pot_options_dot)

# CHILD-2nd <> AUNT-1st
pot_options_dot.set('kernel.adaptor', 'global-generic')
pot_options_dot.set('kernel.type', 'dot')
pot_options_dot.set('kernel.alpha', +0.05) # <- Pull
pot_options_dot.set('kernel.delta', 1.)
pot_options_dot.set('kernel.xi', 2.)
pot_options_dot.set('kernel.mu', 0.80)
for parent_id, child_nodes in nodes_2nd.iteritems():
    for child_node in child_nodes:
        for aunt_node in nodes_1st:
            if aunt_node == parent_id: continue
            child_node.createPotential(aunt_node, pot_options_dot)
            aunt_node.createPotential(child_node, pot_options_dot)


# CHILD-2nd <> COUSIN-2nd
pot_options_dot.set('kernel.adaptor', 'global-generic')
pot_options_dot.set('kernel.type', 'dot')
pot_options_dot.set('kernel.alpha', +0.05) # <- Pull
pot_options_dot.set('kernel.delta', 1.)
pot_options_dot.set('kernel.xi', 2.)
pot_options_dot.set('kernel.mu', 0.80)
parent_ids = sorted(nodes_2nd.keys())
n_parents = len(parent_ids)
for i in range(n_parents):
    par_i = parent_ids[i]
    child_nodes_i = nodes_2nd[par_i]
    for j in range(i+1, n_parents):
        par_j = parent_ids[j]
        child_nodes_j = nodes_2nd[par_j]        
        for node_i in child_nodes_i:
            for node_j in child_nodes_j:
                node_i.createPotential(node_j, pot_options_dot)
                node_j.createPotential(node_i, pot_options_dot)


print "Root + %d nodes (1st) + %d nodes (2nd)" % (len(nodes_1st), len(nodes_2nd_list))

# OPTIMIZE 2nd GENERATION
nodes_opt = nodes_2nd_list
x0_opt = []
for node in nodes_opt:
    #x0 = np.array([ part.pos for part in node.structure ])
    x0 = node.perturbPositions(scale=0.1, zero_pids=[], compute=True)
    x0_opt.append(x0)
x0_opt = np.array(x0_opt).flatten()
multi_optimize_node(nodes_opt, x0_opt)

# OUTPUT
top.writeData()



"""
node_A = top.nodes[1]
node_B = top.nodes[2]
node_C = top.nodes[3]
node_D = top.nodes[4]

pot_options_harm.set('kernel.adaptor', 'global-generic')
pot_options_harm.set('kernel.type', 'dot-harmonic')
pot_options_harm.set('kernel.alpha', +1.)
pot_options_harm.set('kernel.delta', 1.)
pot_options_harm.set('kernel.xi', 2.)
pot_options_harm.set('kernel.mu', 0.95)
node_A.createPotential(node_R, pot_options_harm)
node_B.createPotential(node_R, pot_options_harm)
node_C.createPotential(node_R, pot_options_harm)
node_D.createPotential(node_R, pot_options_harm)

pot_options_dot.set('kernel.adaptor', 'global-generic')
pot_options_dot.set('kernel.type', 'dot')
pot_options_dot.set('kernel.alpha', +0.05)
pot_options_dot.set('kernel.delta', 1.)
pot_options_dot.set('kernel.xi', 2.)
pot_options_dot.set('kernel.mu', 0.80)
node_A.createPotential(node_B, pot_options_dot)
node_A.createPotential(node_C, pot_options_dot)
node_A.createPotential(node_D, pot_options_dot)
node_B.createPotential(node_A, pot_options_dot)
node_B.createPotential(node_C, pot_options_dot)
node_B.createPotential(node_D, pot_options_dot)
node_C.createPotential(node_A, pot_options_dot)
node_C.createPotential(node_B, pot_options_dot)
node_C.createPotential(node_D, pot_options_dot)
node_D.createPotential(node_A, pot_options_dot)
node_D.createPotential(node_B, pot_options_dot)
node_D.createPotential(node_C, pot_options_dot)

nodes_opt = [ node_A, node_B, node_C, node_D ]
x0_opt = []
for node in nodes_opt:
    #x0 = np.array([ part.pos for part in node.structure ])
    x0 = node.perturbPositions(scale=0.1, zero_pids=[], compute=True)
    x0_opt.append(x0)
x0_opt = np.array(x0_opt).flatten()
multi_optimize_node(nodes_opt, x0_opt)

top.writeData()
"""








