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
options.set('radialbasis.N', 9)
options.set('radialbasis.sigma', 0.9)
options.set('radialbasis.integration_steps', 15)
options.set('radialcutoff.Rc', 6.8)
options.set('radialcutoff.Rc_width', 0.5)
options.set('radialcutoff.type', 'shifted-cosine')
options.set('radialcutoff.center_weight', 1.)
options.set('angularbasis.type', 'spherical-harmonic')
options.set('angularbasis.L', 6)
options.set('densitygrid.N', 20)
options.set('densitygrid.dx', 0.15)

# STRUCTURE
xyzfile = 'pair_1_185.xyz'
top = [('dcv2t', 2, 28)]
config = soap.tools.ase_load_single(xyzfile)
structure = soap.tools.setup_structure_ase(config.config_file, config.atoms, top)

seg1 = structure.getSegment(1)
seg2 = structure.getSegment(2)

print seg1.id, seg1.name
print seg2.id, seg2.name

for particle in seg1.particles:
    particle.weight = -1.

for particle in seg2.particles:
    particle.weight = +1.

for part in structure.particles:
    print part.id, part.weight

# SPECTRUM
spectrum = soap.Spectrum(structure, options)
spectrum.compute(seg1, seg2)
spectrum.compute(seg2, seg1)
spectrum.computePower()

spectrum.save('test.arch')
atomic = spectrum.getAtomic(1, "S")
atomic.getLinear("").writeDensityOnGrid('test.cube', options, structure, atomic.getCenter(), True)

