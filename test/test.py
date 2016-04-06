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
for config in ase_config_list:
    print config.config_file
config = ase_config_list[4]
osio << config.atoms << endl

sigma = 0.5

# INITIALIZE OPTIONS
options = soap.Options()
options.set('radialbasis.type', 'gaussian')
options.set('radialbasis.mode', 'adaptive')
#options.set('radialbasis.mode', 'equispaced')
options.set('radialbasis.N', 9)
options.set('radialbasis.sigma', sigma)
options.set('radialbasis.integration_steps', 15)
#options.set('radialbasis.N', 9)
options.set('radialcutoff.Rc', 6.8)
options.set('radialcutoff.Rc_width', 0.5)
options.set('radialcutoff.type', 'shifted-cosine')
options.set('radialcutoff.center_weight', 1.)
#options.set('radialcutoff.center_weight', -7.)

options.set('angularbasis.type', 'spherical-harmonic')
options.set('angularbasis.L', 6)
#options.set('angularbasis.L', 12)
#options.set('densitygrid.N', 20)
options.set('densitygrid.N', 20)
options.set('densitygrid.dx', 0.15)

# LOAD STRUCTURE
structure = soap.tools.setup_structure_ase(config.config_file, config.atoms)

osio << osio.mg << structure.label << endl

for atom in structure:
    atom.sigma = sigma # 0.5 # 0.4*element_vdw_radius[atom.type]
    atom.weight = 1. #element_mass[atom.type]
    #atom.sigma = 0.5*element_vdw_radius[atom.type]
    #atom.weight = element_mass[atom.type]
    osio << atom.name << atom.type << atom.weight << atom.sigma << atom.pos << endl
    #if atom.id > 60: atom.weight *= -1

# COMPUTE SPECTRUM
spectrum = soap.Spectrum(structure, options)
spectrum.compute()
spectrum.computePower()
spectrum.save("test_serialization/%s.spectrum.arch" % structure.label)
spectrum.save("test_invert/%s.spectrum.arch" % structure.label)

spectrum.writeDensity(1, "C", "")
spectrum.writePowerDensity(1, "C", "g", "c")
#spectrum.writeDensityOnGrid(1, "C", "")
#spectrum.writeDensityOnGridInverse(1, "C", "g", "c")

# INVERSION
basis = spectrum.basis
center = spectrum.getAtomic(1, "C").getCenter()
xnkl = spectrum.getAtomic(1, "C").getPower("g", "c").array

ofs = open('density.power.python.coeff', 'w')
for n in range(basis.N):
    for k in range(basis.N):
        for l in range(basis.L+1):
            x = xnkl[n*basis.N+k,l]
            ofs.write("%2d %2d %+2d %+1.7e %+1.7e\n" % (n, k, l, x.real, x.imag))
ofs.close()

pox = soap.PowerExpansion(basis)
pox.array = xnkl
ynkl = pox.array

# powex = spectrum.getAtomicSpectrum(1, "C").getPowerExpansion("g", "c")
#osio << osio.mg << "Atomic spectrum" << endl
#atomic = soap.AtomicSpectrum()

#spectrum.writeDensityOnGrid(2, "S", "")
#spectrum.writeDensityOnGrid(7, "C", "") # line.xyz
#spectrum.writeDensityOnGrid(3, "C", "") # linedot.xyz
#spectrum.writeDensityOnGrid(41, "C", "") # C60_pair.xyz

osio.okquit()



