#! /usr/bin/env python
import soap
import soap.tools

import os
import numpy as np
from momo import osio, endl, flush

options = soap.Options()
options.excludeCenters(['H'])
options.excludeTargets([])
options.set('radialbasis.type', 'gaussian')
options.set('radialbasis.mode', 'adaptive')
options.set('radialbasis.N', 2) # 9
options.set('radialbasis.sigma', 9) # 0.9
options.set('radialbasis.integration_steps', 15)
options.set('radialcutoff.Rc', 6.8)
options.set('radialcutoff.Rc_width', 0.5)
options.set('radialcutoff.type', 'shifted-cosine')
options.set('radialcutoff.center_weight', 1.)
options.set('angularbasis.type', 'spherical-harmonic')
options.set('angularbasis.L', 2) # 6
options.set('densitygrid.N', 20)
options.set('densitygrid.dx', 0.15)

# STRUCTURE
xyzfile = 'pair_1_185.xyz'
top = [('dcv2t', 2, 28)]
config = soap.tools.ase_load_single(xyzfile)
structure = soap.tools.setup_structure_ase(config.config_file, config.atoms, top)

# SPECTRUM
spectrum = soap.Spectrum(structure, options)
spectrum.compute()
spectrum.computePower()

types_global = ['H', 'C', 'N', 'S']
for atomic in spectrum:
    xnklab = soap.tools.Xnklab(atomic, types_global).reduce()
    print xnklab
    raw_input('...')

