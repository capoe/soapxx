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

options = soap.Options()
options.excludeCenters([])
options.excludeTargets([])
options.set('radialbasis.type', 'gaussian')
options.set('radialbasis.mode', 'adaptive')
options.set('radialbasis.N', 9) # 9
options.set('radialbasis.sigma', 0.5) # 0.9
options.set('radialbasis.integration_steps', 15)
options.set('radialcutoff.Rc', 6.8)
options.set('radialcutoff.Rc_width', 0.5)
options.set('radialcutoff.type', 'heaviside')
options.set('radialcutoff.center_weight', 1.)
options.set('angularbasis.type', 'spherical-harmonic')
options.set('angularbasis.L', 6) # 6
options.set('densitygrid.N', 20)
options.set('densitygrid.dx', 0.15)
options.set('spectrum.gradients', True)
options.set('kernel.adaptor', 'generic')
options.set('kernel.type', 'dot')
options.set('kernel.delta', 1.)
options.set('kernel.xi', 4.)

# STRUCTURE
xyzfile = 'config.xyz'
config = soap.tools.ase_load_single(xyzfile)
structure = soap.tools.setup_structure_ase(config.config_file, config.atoms)

for part in structure.particles:
    print part.id, part.type, part.pos
    
positions_0 = [ part.pos for part in structure ]

# KERNEL
soap.silence()
kernelpot = KernelPotential(options)
kernelpot.acquire(structure, 1.)

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




# MAP
Nx = 20
Ny = 20
dx = 0.25
dy = 0.25

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
        e_in = kernelpot.computeEnergy(structure)
        ofs.write('%+1.7e %+1.7e %+1.7e\n' % (x, y, e_in))
    ofs.write('\n')
ofs.close()


