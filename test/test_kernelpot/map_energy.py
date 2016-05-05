#! /usr/bin/env python
import soap
import soap.tools

import os
import numpy as np
import logging

from momo import osio, endl, flush
from kernel import KernelPotential, restore_positions, apply_force_step

logging.basicConfig(
    format='[%(asctime)s] %(message)s', 
    datefmt='%I:%M:%S', 
    level=logging.ERROR)
verbose = False

exclude_center_pids = []
options = soap.Options()
options.excludeCenters([])
options.excludeTargets([])
options.excludeCenterIds(exclude_center_pids)
options.excludeTargetIds([])
options.set('radialbasis.type', 'gaussian')
options.set('radialbasis.mode', 'adaptive')
options.set('radialbasis.N', 9) # 9
options.set('radialbasis.sigma', 0.5) # 0.9
options.set('radialbasis.integration_steps', 15)
options.set('radialcutoff.Rc', 6.8)
options.set('radialcutoff.Rc_width', 0.5)
options.set('radialcutoff.type', 'heaviside')
options.set('radialcutoff.center_weight', 0.5)
options.set('angularbasis.type', 'spherical-harmonic')
options.set('angularbasis.L', 6) # 6
options.set('densitygrid.N', 20)
options.set('densitygrid.dx', 0.15)
options.set('spectrum.gradients', True)
options.set('spectrum.2l1_norm', False) # <- pull here
options.set('kernel.adaptor', 'generic') # <- pull here
options.set('kernel.type', 'dot')
options.set('kernel.delta', 1.)
options.set('kernel.xi', 4.) # <- pull here

# STRUCTURE
xyzfile = 'config.xyz'
config = soap.tools.ase_load_single(xyzfile)
structure = soap.tools.setup_structure_ase(config.config_file, config.atoms)

for part in structure.particles:
    print part.id, part.type, part.pos
    
positions_0 = [ part.pos for part in structure ]

# KERNEL
kernelpot = KernelPotential(options)
kernelpot.acquire(structure, 1.)    
soap.silence()


positions = [ 
    np.array([0.,   0.,  0.]),
    np.array([1.75, 0.,  0.]),
    np.array([0.,  1.75, 0.]) ]
restore_positions(structure, positions)
#kernelpot.acquire(structure, 1.)
print "O'", kernelpot.computeEnergy(structure)

positions = [ 
    np.array([0.,  0., 0.]),
    np.array([1.5, 0., 0.]),
    np.array([1.5, 0., 0.]) ]
restore_positions(structure, positions)
print "A", kernelpot.computeEnergy(structure)

positions = [ 
    np.array([0.,  0.,  0.]),
    np.array([1.5, 0.,  0.]),
    np.array([0.,  1.5, 0.]) ]
restore_positions(structure, positions)
print "B", kernelpot.computeEnergy(structure)

positions = [ 
    np.array([0.,  0.,  0.]),
    np.array([0., 1.5,  0.]),
    np.array([0.,  0., 1.5]) ]
restore_positions(structure, positions)
print "C", kernelpot.computeEnergy(structure)

positions = [ 
    np.array([0.,  0.,  0.]),
    np.array([0.,  0.,  1.5]),
    np.array([1.5, 0.,  0.]) ]
restore_positions(structure, positions)
print "D", kernelpot.computeEnergy(structure)

positions = [ 
    np.array([0.,   0.,  0.]),
    np.array([1.5,  0.,  0.]),
    np.array([100.,100.,100.]) ]
restore_positions(structure, positions)
#kernelpot.acquire(structure, 1.)
print "E", kernelpot.computeEnergy(structure)




# MAP
Nx = 40
Ny = 40
dx = 0.25
dy = 0.25

xs = []
ys = []
zs = []

ofs = open('out', 'w')
for i in range(Nx+1):
    for j in range(Ny+1):
        x = i*dx - 0.5*Nx*dx
        y = j*dy - 0.5*Ny*dy
        print i, j
        positions = [ 
            np.array([0.,  0., 0.]),
            np.array([1.5, 0., 0.]),
            np.array([x,   y,  0.]) ]
        restore_positions(structure, positions)
        e_in, prj_mat = kernelpot.computeEnergy(structure, True)
        prj_mat = np.array(prj_mat)
        prj_mat = prj_mat.reshape(prj_mat.shape[0]*prj_mat.shape[1])
        prj_mat_str = ('%s' % prj_mat).replace('[','').replace(']','').replace('\n','')
        
        xs.append(x)
        ys.append(y)
        zs.append(prj_mat)
        
        print prj_mat_str
        ofs.write('%+1.7e %+1.7e %+1.7e\n' % (x, y, e_in))
        #f = kernelpot.computeForces(structure)[2]
        #ofs.write('%+1.7e %+1.7e %+1.7e %+1.7e %+1.7e %+1.7e\n' % (x, y, e_in, f[0], f[1], f[2]))
    ofs.write('\n')
ofs.close()

xs = np.array(xs)
ys = np.array(ys)
zs = np.array(zs)

np.savetxt('plot/np.x.txt', xs)
np.savetxt('plot/np.y.txt', ys)
np.savetxt('plot/np.z.txt', zs)

