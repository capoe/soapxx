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

center_id = 1
center_type = "C"
density_type = "O"

# TARGET EXTRACTION
atomic = spectrum.getAtomic(center_id, center_type)
center = atomic.getCenter()
atomic_xnkl = atomic.getPower(density_type, density_type).array

# INVERT XKNL => QNLM
atomic_inv_qnlm = soap.tools.invert_xnkl_aa(atomic_xnkl, basis, l_smaller=0, l_larger=1)

# CREATE INVERSION EXPANSION
basisexp = soap.BasisExpansion(basis)
basisexp.array = atomic_inv_qnlm

# CREATE INVERSION ATOMIC SPECTRUM
atomic_inv = soap.AtomicSpectrum(center, basis)
atomic_inv.addLinear(density_type, basisexp)
atomic_inv.computePower()
atomic_inv_xnkl = atomic_inv.getPower(density_type, density_type).array

dxnkl = atomic_inv_xnkl - atomic_xnkl
print atomic_xnkl
raw_input('...')
print atomic_inv_xnkl
raw_input('...')
print dxnkl
raw_input('...')

# OUTPUT DENSITY
cubefile = 'recon.id-%d_center-%s_type-%s.cube' % (center_id, center_type, density_type)
coefffile = cubefile[:-5]+'.coeff'
basisexp.writeDensityOnGrid(cubefile, options, structure, center, True)
basisexp.writeDensity(coefffile, options, structure, center)

qnlm_orig = atomic_inv_qnlm
for l in range(basis.L+1):
    qnlm = np.copy(qnlm_orig)
    for n in range(basis.N):
        for ll in range(basis.L+1):
            if ll == l: continue
            for m in range(-ll,ll+1):
                qnlm[n,ll*ll+ll+m] = np.complex(0.,0.)
    #print qnlm
    basisexp.array = qnlm
    cubefile = 'recon.id-%d_center-%s_type-%s_l-%d.cube' % (center_id, center_type, density_type, l)
    coefffile = cubefile[:-5]+'.coeff'
    #basisexp.writeDensityOnGrid(cubefile, options, structure, center, True)
    #basisexp.writeDensity(coefffile, options, structure, center)

