#! /usr/bin/env python
import soap
import soap.tools

import os
import numpy as np
import logging

from momo import osio, endl, flush
from kernel import KernelPotential, restore_positions, apply_force_step, apply_force_norm_step, perturb_positions, random_positions

logging.basicConfig(
    format='[%(asctime)s] %(message)s', 
    datefmt='%I:%M:%S', 
    level=logging.ERROR)
verbose = False


exclude_center_pids = [2,3]
exclude_perturb_pids = [1]
constrain_pids = []
perturb_initial = False
random_initial = True



options = soap.Options()
options.excludeCenters([])
options.excludeTargets([])
options.excludeCenterIds(exclude_center_pids)
options.excludeTargetIds([])
options.set('radialbasis.type', 'gaussian')
options.set('radialbasis.mode', 'adaptive') # equispaced or adaptive
options.set('radialbasis.N', 9) # 9
options.set('radialbasis.sigma', 1.5) # 0.9
options.set('radialbasis.integration_steps', 15)
options.set('radialcutoff.Rc', 4.)
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
print "Positions at start"
for p in positions_0: print p

# SETUP KERNEL
soap.silence()
kernelpot = KernelPotential(options)
kernelpot.acquire(structure, -1.)

#positions = [ 
#    np.array([0.,   0.,  0.]),
#    np.array([1.75, 0.,  0.]),
#    np.array([0.,  1.75, 0.]) ]
#restore_positions(structure, positions)
#kernelpot.acquire(structure, -10.)
#restore_positions(structure, positions_0)

# PERTURB POSITIONS
if perturb_initial:
    positions_0_pert = perturb_positions(structure, exclude_pid=exclude_perturb_pids)    
    print "Positions after initial perturbation"
    for p in positions_0_pert: print p
    
if random_initial:
    positions_0_pert = random_positions(structure)
    print "Positions after initial randomization"
    for p in positions_0_pert: print p

raw_input('...')

acquire_restore_every_nth = -1

ofs = open('log.xyz', 'w')

# Write first frame
ofs.write('%d\n\n' % structure.n_particles)
for p in structure.particles:
    r = p.pos
    ofs.write('%s %+1.7f %+1.7f %+1.7f\n' % (p.type, r[0], r[1], r[2]))

# Energy (int)
e_in = kernelpot.computeEnergy(structure)
osio << osio.my << "Energy(in)" << e_in << osio.endl

for n in range(100):
    osio << osio.mg << "ITERATION n =" << n << osio.endl
    
    # Energy  
    e_in = kernelpot.computeEnergy(structure)  
    if verbose: osio << osio.my << "Energy(n=%d)" % n << e_in << osio.endl
    
    # Compute forces and step ...
    forces = kernelpot.computeForces(structure)
    if verbose:
        print "Forces"
        for f in forces: print f
        
    positions = apply_force_step(structure, forces, scale=0.01, constrain_particles=constrain_pids)
    if verbose:
        print "Positions after force step"
        for p in positions: print p

    # Write trajectory
    ofs.write('%d\n\n' % structure.n_particles)
    for p in structure.particles:
        r = p.pos
        ofs.write('%s %+1.7f %+1.7f %+1.7f\n' % (p.type, r[0], r[1], r[2]))

    if n > 0 and acquire_restore_every_nth > 0:
        if (n % acquire_restore_every_nth) == 0:
            osio << osio.mb << "Acquire and reset positions" << osio.endl
            kernelpot.acquire(structure, 1.)
            positions = restore_positions(structure, positions_0)
            print "Positions after restore"
            for p in positions: print p

    #raw_input('...')
# Energy (out)
e_out = kernelpot.computeEnergy(structure)
osio << osio.my << "Energy(out)" << e_out << osio.endl

ofs.close()






































