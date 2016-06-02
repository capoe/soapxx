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
sc = 0.05
lj_sigma = 0.02
N_network = 3
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
pot_options_harm.set('kernel.alpha', +1.) # TODO
pot_options_harm.set('kernel.delta', 1.)
pot_options_harm.set('kernel.xi', 2.)
pot_options_harm.set('kernel.mu', 0.95) # TODO

pot_options_3harm = soap.Options()
pot_options_3harm.set('kernel.adaptor', 'global-generic')
pot_options_3harm.set('kernel.type', 'dot-3-harmonic')
pot_options_3harm.set('kernel.alpha', +5.) # TODO
pot_options_3harm.set('kernel.delta', 1.)
pot_options_3harm.set('kernel.xi', 2.)


hex_c = 0.1
pot_options_harm_dist = soap.Options()
pot_options_harm_dist.set('kernel.adaptor', 'global-generic')
pot_options_harm_dist.set('kernel.type', 'dot-harmonic-dist')
pot_options_harm_dist.set('kernel.alpha', +1.) # TODO
pot_options_harm_dist.set('kernel.delta', 1.)
pot_options_harm_dist.set('kernel.xi', 2.)
pot_options_harm_dist.set('kernel.dotharmdist_d0', hex_c) # TODO
pot_options_harm_dist.set('kernel.dotharmdist_eps', 0.001)

pot_options_3harm_dist = soap.Options()
pot_options_3harm_dist.set('kernel.adaptor', 'global-generic')
pot_options_3harm_dist.set('kernel.type', 'dot-3-harmonic-dist')
pot_options_3harm_dist.set('kernel.alpha', +1.) # TODO
pot_options_3harm_dist.set('kernel.delta', 1.)
pot_options_3harm_dist.set('kernel.xi', 2.)
pot_options_3harm_dist.set('kernel.dot3harmdist_d0', 3.**0.5*hex_c)
pot_options_3harm_dist.set('kernel.dot3harmdist_eps', 0.001)

#pot_options_lj = soap.Options()
#pot_options_lj.set('potentials.lj.sigma', 0.02)

pot_options_dotlj = soap.Options()
pot_options_dotlj.set('kernel.adaptor', 'global-generic')
pot_options_dotlj.set('kernel.type', 'dot-lj')
pot_options_dotlj.set('kernel.alpha', +0.1) # TODO
pot_options_dotlj.set('kernel.delta', 1.)
pot_options_dotlj.set('kernel.xi', 2.)
pot_options_dotlj.set('kernel.lj_sigma', 0.1) # TODO
pot_options_dotlj.set('kernel.lj_eps_cap', 0.001)




class SimSpaceVertex(object):
    def __init__(self, id, pos, top):
        self.id = id
        self.pos = pos
        self.nbs = []
        
        self.top = top
        self.node = None        
    def initialize(self, structure):
        assert not self.isInitialized()
        self.node = self.top.createNode(structure)
        return self.node
    def isInitialized(self):
        return self.node != None
    def addNeighbour(self, nb_vertex):
        self.nbs.append(nb_vertex)
        return
    def printInfo(self):
        print "%d # nbs = %d" % (self.id, len(self.nbs))
        for nb in self.nbs:
            dp = nb.pos - self.pos
            print "   dr = %+1.7e %+1.7e %+1.7e" % (dp[0], dp[1], dp[2])
        return
    def printConnectivity(self, visited, level, indent=''):
        if level == 0:
            return visited
        print "%s ID=%d NBS=%d (level %d)" % (indent, self.id, len(self.nbs), level)
        for nb in self.nbs:
            if nb in visited:
                pass
            else:
                visited.append(nb)
                visited = nb.printConnectivity(visited, level=level-1, indent=indent+'  ')
        return visited

class SimSpaceNetwork(object):
    def __init__(self, options, pos_list=[]):
        self.vertices = []
        for pos in pos_list:
            self.createVertex(pos)
            
        self.top = SimSpaceTopology(options)
        self.root_vertex = None
        self.spawned = []  
        self.gen_spawned = []      
        return
    def seed(self, root_idx, structure):
        self.root_vertex = self.vertices[root_idx]
        self.root_vertex.initialize(structure)
        self.spawned.append(self.root_vertex)
        self.gen_spawned = [ [ self.root_vertex ] ]
    def spawn(self):
        spawned = []
        for vertex in self.spawned:
            struct = vertex.node.structure
            for nb in vertex.nbs:                
                if nb.isInitialized(): continue
                else:
                    print "Spawn vertex %d" % nb.id
                    nb.initialize(struct)
                    spawned.append(nb)
        self.spawned = self.spawned + spawned
        self.gen_spawned.append(spawned)
        return spawned
    def addNonbondedPairPotential(self, options, exclude_nbs=True):
        print "Add non-bonded pair potential ..."
        n_spawns = len(self.spawned)
        for i in range(n_spawns):
            for j in range(i+1, n_spawns):
                vert_i = self.spawned[i]
                vert_j = self.spawned[j]
                if exclude_nbs:
                    excluded = False
                    if vert_i in vert_j.nbs: excluded = True
                    if vert_j in vert_i.nbs: assert excluded
                    if excluded: continue
                    for nb in vert_i.nbs:
                        if vert_j in nb.nbs: excluded = True
                    for nb in vert_j.nbs:
                        if vert_i in nb.nbs: assert excluded
                    if excluded: continue
                node_i = self.spawned[i].node
                node_j = self.spawned[j].node
                print "%d:%d" % (node_i.id, node_j.id),
                node_i.createPotential(node_j, options)
                node_j.createPotential(node_i, options)
        print ""
        return
    def addPairPotential(self, options):
        print "Add bonded pair potential ..."
        for spawn in self.spawned:
            for nb in spawn.nbs:
                if nb.isInitialized():
                    print "%d:%d" % (spawn.node.id, nb.node.id),
                    spawn.node.createPotential(nb.node, options) # <- reverse potential added elsewhere in loop
            print ""
        return
    def addThreePotential(self, options):
        print "Add three-body potential ..."
        for spawn in self.spawned:
            if len(spawn.nbs) != 3: continue
            nb1 = spawn.nbs[0]
            nb2 = spawn.nbs[1]
            nb3 = spawn.nbs[2]
            if nb1.isInitialized() and nb2.isInitialized() and nb3.isInitialized():
                print "%d::%d:%d:%d" % (spawn.node.id, nb1.node.id, nb2.node.id, nb3.node.id)
                nb1.node.createThreePotential(nb2.node, nb3.node, options)
                nb2.node.createThreePotential(nb3.node, nb1.node, options)
                nb3.node.createThreePotential(nb1.node, nb2.node, options)
        return
    def clearPotentials(self):
        for spawn in self.spawned:
            spawn.node.clearPotentials()
        return
    def createVertex(self, pos):
        vert = SimSpaceVertex(len(self.vertices)+1, pos, self.top)
        self.vertices.append(vert)
        return
    def createNeighbours(self, cutoff):
        n_vertices = len(self.vertices)
        for i in range(n_vertices):
            verti = self.vertices[i]
            for j in range(i+1, n_vertices):
                vertj = self.vertices[j]
                dp = verti.pos-vertj.pos
                if np.dot(dp,dp) <= cutoff**2:
                    verti.addNeighbour(vertj)
                    vertj.addNeighbour(verti)
    def printInfo(self):
        print "Network with %d vertices" % (len(self.vertices))
        for vert in self.vertices:
            vert.printInfo()
        return
    def printInfoSpawned(self):
        for idx, gen in enumerate(self.gen_spawned):
            print "Gen # = %d : Spawn # = %d" % (idx+1, len(gen))
        return
    def createHexComb(self, c, N):
        # Primitive unit cell vectors
        vec_a = np.array([3**0.5*c, 0., 0.])
        vec_b = np.array([3**0.5*0.5*c, 1.5*c, 0.])
        vec_c = np.array([3**0.5*0.5*c, 0.5*c, 0.])
        # Rectangular unit cell
        vec_A = vec_a
        vec_B = np.array([0, 3*c, 0.])
        vec_C = vec_b
        # On-edge particles
        p_list = []
        for na in range(-2*N,2*N+1):
            for nb in range(-N,N+1):
                p1 = na*vec_A + nb*vec_B
                p2 = p1+vec_C
                p_list.append(p1)
                p_list.append(p2)
        # Further basis atoms
        sub_p = []
        for p in p_list:
            sub_p.append(p+vec_c)
        p_list = p_list + sub_p
        for p in p_list:
            self.createVertex(p)
        return
    def writeXyz(self, outfile='out.honeycomb.xyz'):
        ofs = open(outfile, 'w')
        ofs.write('%d\n\n' % len(self.vertices))
        for vert in self.vertices:
            p = vert.pos
            ofs.write('C %+1.7e %+1.7e %+1.7e\n' % (p[0], p[1], p[2]))
        ofs.close()
        return
    def writeNeighbourPairs(self, outfile):
        ofs = open(outfile, 'w')
        for vertex in self.spawned:
            for nb in vertex.nbs:
                if not nb.isInitialized(): continue
                id1 = vertex.node.id
                id2 = nb.node.id
                ofs.write('%d %d\n' % (id1, id2))
        ofs.close()
        return
    def printConnectivity(self, root_idx=0, levels=3, visited_xyz='out.visited.xyz'):
        root = self.vertices[root_idx]
        visited = [ root ]
        print "Root at", root.pos
        visited = visited + root.printConnectivity(visited, levels, '')
        ofs = open(visited_xyz, 'w')
        ofs.write('%d\n\n' % len(visited))
        for vis in visited:
            pos = vis.pos
            ofs.write('C %+1.7e %+1.7e %+1.7e\n' % (pos[0], pos[1], pos[2]))
        ofs.close()
        return



# LOAD STRUCTURES
structures = [ 
    soap.tools.setup_structure_ase(c.config_file, c.atoms)
    for c in soap.tools.ase_load_all('in.configs')
]
struct_dict = {}
for struct in structures:
    print struct.label
    struct_dict[struct.label] = struct

# CHANGE TO WORKING DIRECTORY
osio.cd('out.files.hexcomb')

# SELECT STRUCTURE
struct = struct_dict[select_struct_label]



# CREATE NETWORK
hexcomb = SimSpaceNetwork(options)
hexcomb.createHexComb(c=1.4, N=N_network)
hexcomb.createNeighbours(cutoff=1.41)
hexcomb.writeXyz()

root_idx = 2*((2*N_network+1)*2*N_network + N_network)
"""
hexcomb.printInfo()
hexcomb.printConnectivity(root_idx=root_idx, levels=3)
"""

# SEED
hexcomb.seed(root_idx, struct)

N_generations = 3
for i in range(N_generations):
    gen_id = i+1
    gen_prefix = "out.gen_%d" % gen_id
    print "Spawn generation %d" % gen_id   
    
    hexcomb.top.computePotentialEnergy()
    new_vertices = hexcomb.spawn()
    hexcomb.printInfoSpawned()
    hexcomb.writeNeighbourPairs('%s.tree.txt' % gen_prefix)
    
    # POTENTIALS
    hexcomb.clearPotentials()
    #hexcomb.addPairPotential(pot_options_harm)
    #hexcomb.addThreePotential(pot_options_3harm)
    hexcomb.addPairPotential(pot_options_harm_dist)
    hexcomb.addThreePotential(pot_options_3harm_dist)
    #hexcomb.addNonbondedPairPotential(pot_options_dotlj)
    
    # OPTIMIZE 1st GENERATION
    nodes_opt = [ vertex.node for vertex in hexcomb.gen_spawned[-1] ]
    print "Optimize %d nodes" % len(nodes_opt)
    x0_opt = []
    for node in nodes_opt:
        x0 = node.perturbPositions(scale=0.1, zero_pids=[], compute=True)
        x0_opt.append(x0)
    x0_opt = np.array(x0_opt).flatten()
    multi_optimize_node(nodes_opt, x0_opt)
    
    # OUTPUT
    hexcomb.top.writeData(prefix=gen_prefix)
    #raw_input('...')

# RETURN
osio.root()

"""
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

# TOPOLOGY
top = SimSpaceTopology(options)

node_0 = top.createNode(struct)

node_A = top.createNode(struct)
node_B = top.createNode(struct)
node_C = top.createNode(struct)

node_A.createPotential(node_0, pot_options_harm)
node_B.createPotential(node_0, pot_options_harm)
node_C.createPotential(node_0, pot_options_harm)

#node_A.createThreePotential(node_B, node_C, pot_options_3harm)
#node_B.createThreePotential(node_C, node_A, pot_options_3harm)
#node_C.createThreePotential(node_A, node_B, pot_options_3harm)

# CREATE TREE
osio.cd('out.files')
    
# OPTIMIZE 1st GENERATION
nodes_opt = [ node_A, node_B, node_C ]
x0_opt = []
for node in nodes_opt:
    x0 = node.perturbPositions(scale=0.1, zero_pids=[], compute=True)
    x0_opt.append(x0)
x0_opt = np.array(x0_opt).flatten()
multi_optimize_node(nodes_opt, x0_opt)
    
# OUTPUT
top.writeData(prefix='out.top')
    
osio.root()

"""





