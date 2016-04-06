#! /usr/bin/env python
import soap
import soap.tools

import os
import numpy as np
from momo import osio, endl, flush

archfile = 'config_000057.xyz.spectrum.arch'

# SPECTRUM
spectrum = soap.Spectrum(archfile)
structure = spectrum.structure
basis = spectrum.basis
options = spectrum.options
osio << osio.mb << options << endl


density_type = "C"

# TARGET EXTRACTION
atomic = spectrum.getAtomic(1, "C")
atomic_center = atomic.getCenter()
#atomic_xnkl = atomic.getPower("g", "c").array
atomic_xnkl = atomic.getPower(density_type, density_type).array

ofs = open('density.power.python.coeff', 'w')
for n in range(basis.N):
    for k in range(basis.N):
        for l in range(basis.L+1):
            x = atomic_xnkl[n*basis.N+k,l]
            ofs.write("%2d %2d %+2d %+1.7e %+1.7e\n" % (n, k, l, x.real, x.imag))
ofs.close()

# INVERSION SPECTRUM
spectrum_inv = soap.Spectrum(structure, options, basis)

# Create empty atomic spectrum, invert given Xnkl
atomic_inv = soap.AtomicSpectrum(atomic_center, basis)
spectrum_inv.addAtomic(atomic_inv)

# Find basis expansion Qnlm from Xnkl
atomic_inv_qnlm = soap.tools.invert_xnkl_aa(atomic_xnkl, basis, l_smaller=2, l_larger=2)
basx = soap.BasisExpansion(basis)
basx.array = atomic_inv_qnlm

# Add basis expansion & compute power spectrum
atomic_inv.addLinear(density_type, basx)
atomic_inv.computePower()

# OUTPUT DENSITY
#spectrum_inv.writeDensity(0, "C", "")
#spectrum_inv.writePowerDensity(0, "C", "g", "c")
#spectrum_inv.writeDensityOnGrid(0, "C", "")

spectrum_inv.writeDensity(0, "C", density_type)
spectrum_inv.writePowerDensity(0, "C", density_type, density_type)
spectrum_inv.writeDensityOnGrid(0, "C", density_type)

#basx.array = atomic_xnkl
#powx = soap.PowerExpansion(basis)
#powx.array = atomic_xnkl

