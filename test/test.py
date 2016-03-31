#! /usr/bin/env python
import soap
import soap.tools

import os
import numpy as np
from momo import osio, endl, flush

ase_config_list = soap.tools.ase_load_all('configs')
config = ase_config_list[0]
osio << config.atoms << endl

# INITIALIZE OPTIONS
options = soap.Options()
options.set('radialbasis.type', 'gaussian')
options.set('radialbasis.N', 9)
options.set('radialbasis.Rc', 4.0)
options.set('radialbasis.sigma', 0.5)
options.set('angularbasis.type', 'spherical-harmonic')
options.set('angularbasis.L', 6)
options.set('densitygrid.N', 20)
options.set('densitygrid.dx', 0.1)

# LOAD STRUCTURE
structure = soap.tools.setup_structure_ase(config.config_file, config.atoms)

osio << osio.mg << structure.label << endl
for atom in structure:
    osio << atom.name << atom.type << atom.weight << atom.sigma << atom.pos << endl
    atom.sigma = 0.0

# COMPUTE SPECTRUM
spectrum = soap.Spectrum(structure, options)
spectrum.compute()







osio.okquit()




































structure = soap.Structure(config.config_file)

# BOX TESTS
osio << osio.mg << "Box checks ..." << endl
r1 = np.array([0.2,0.3,0.4])
r2 = np.array([0.8,1.0,1.2])
# OPEN BOX
structure.box = np.array([[0,0,0],[0,0,0],[0,0,0]])
print "Connect", r1, r2, " => ", structure.connect(r1,r2)
# CUBIC BOX
a = np.array([0.9,0,0])
b = np.array([0,0.7,0])
c = np.array([0,0,0.5])
structure.box = np.array([a,b,c])
print "Connect", r1, r2, " => ", structure.connect(r1,r2)


segment = structure.addSegment()
particle = structure.addParticle(segment)
particle.pos = np.array([-1,0,1]) # vec(0,1,-1)
print type(particle.pos), particle.pos
particle.mass = 12.

