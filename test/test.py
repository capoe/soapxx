#! /usr/bin/env python
import soap
import soap.tools

import os
import numpy as np
from momo import osio, endl, flush

element_vdw_radius = { 
'C':1.70,
'N':1.55,
'O':1.52,
'H':1.20
}
element_mass = {
'C':12,
'N':14,
'O':16,
'H':1
}

ase_config_list = soap.tools.ase_load_all('configs')
config = ase_config_list[1]
osio << config.atoms << endl

# INITIALIZE OPTIONS
options = soap.Options()
options.set('radialbasis.type', 'gaussian')
options.set('radialbasis.mode', 'adaptive')
#options.set('radialbasis.mode', 'equispaced')
options.set('radialbasis.N', 9)
options.set('radialbasis.sigma', 0.5)
options.set('radialbasis.integration_steps', 15)
#options.set('radialbasis.N', 9)
options.set('radialcutoff.Rc', 4.)
options.set('radialcutoff.Rc_width', 0.5)
options.set('radialcutoff.type', 'shifted-cosine')
options.set('radialcutoff.center_weight', -7.)
options.set('radialcutoff.center_weight', 1.)

options.set('angularbasis.type', 'spherical-harmonic')
options.set('angularbasis.L', 6)
#options.set('angularbasis.L', 6)
options.set('densitygrid.N', 20)
options.set('densitygrid.dx', 0.15)

# LOAD STRUCTURE
structure = soap.tools.setup_structure_ase(config.config_file, config.atoms)

osio << osio.mg << structure.label << endl

for atom in structure:
    atom.sigma = 0.5 # 0.4*element_vdw_radius[atom.type]
    atom.weight = 1. #element_mass[atom.type]
    #atom.sigma = 0.5*element_vdw_radius[atom.type]
    #atom.weight = element_mass[atom.type]
    osio << atom.name << atom.type << atom.weight << atom.sigma << atom.pos << endl

# COMPUTE SPECTRUM
spectrum = soap.Spectrum(structure, options)
spectrum.compute()
#spectrum.writeDensityOnGrid(1, "C", "")
spectrum.save("test_serialization/%s.spectrum.arch" % structure.label)

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

