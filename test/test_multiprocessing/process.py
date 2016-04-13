#! /usr/bin/env python
import soap
import soap.tools
import numpy as np
import multiprocessing as mp
import functools as fct
import threading

from __pyosshell__ import get_dirs

from momo import os, osio, endl, flush
from momoxml import *

# LOGGER
soap.silence()
MP_LOCK = mp.Lock()

options_dict = {
'radialbasis.type' : 'gaussian',
'radialbasis.mode' : 'adaptive',
'radialbasis.N' : 9,
'radialbasis.sigma': 0.9,
'radialbasis.integration_steps': 15,
'radialcutoff.Rc': 6.8,
'radialcutoff.Rc_width': 0.5,
'radialcutoff.type': 'shifted-cosine',
'radialcutoff.center_weight': 1.,
'angularbasis.type': 'spherical-harmonic',
'angularbasis.L': 6,
'densitygrid.N': 20,
'densitygrid.dx': 0.15}

exclude_centers = ['H']
exclude_targets = []

element_vdw_radius = { 
'H':1.20,
'C':1.70,
'N':1.55,
'O':1.52,
'S':1.80
}
element_mass = {
'H':1,
'C':12,
'N':14,
'O':16,
'S':32
}
element_valence_order = {
'H':1,
'C':4,
'N':5,
'O':6,
'S':6
}

# READ COORDINATES
ase_config_list = soap.tools.ase_load_all('group_rnd_short', log=osio)

def compute_spectrum(config, options_dict, exclude_centers, exclude_targets):
    # ANNOUNCE
    global MP_LOCK
    MP_LOCK.acquire()
    osio << osio.item \
         << "Soap for:  " << config.config_file \
         << "PID=%d" % mp.current_process().pid << endl
    MP_LOCK.release()
    # SET OPTIONS
    options = soap.Options()
    for item in options_dict.iteritems():
        options.set(*item)
    options.excludeCenters(exclude_centers)
    options.excludeTargets(exclude_targets)
    # PROCESS STRUCTURE
    structure = soap.tools.setup_structure_ase(config.config_file, config.atoms)
    for particle in structure:
        particle.sigma = 0.5*element_vdw_radius[particle.type]
        particle.weight = element_valence_order[particle.type]
    # COMPUTE SPECTRUM
    spectrum = soap.Spectrum(structure, options)
    spectrum.compute()
    spectrum.computePower()
    spectrum.save('%s.spectrum' % config.config_file)
    return config.config_file

pool = mp.Pool(processes=4)
compute_spectrum_partial = fct.partial(
    compute_spectrum, 
    options_dict=options_dict, 
    exclude_centers=exclude_centers, 
    exclude_targets=exclude_targets)
arch_files = pool.map(compute_spectrum_partial, ase_config_list)    
pool.close()
pool.join()










