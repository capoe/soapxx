#! /usr/bin/env python
import soap
import soap.tools
import numpy as np

from __pyosshell__ import get_dirs

from momo import os, osio, endl, flush
from momoxml import *

# LOGGER
soap.silence()

# INITIALIZE OPTIONS
options = soap.Options()
options.set('radialbasis.type', 'gaussian')
options.set('radialbasis.mode', 'adaptive')
#options.set('radialbasis.mode', 'equispaced')
options.set('radialbasis.N', 9)
options.set('radialbasis.sigma', 0.9)
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
# DEFINE EXCLUSIONS
options.excludeCenters(['H'])
options.excludeTargets([])

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

# LOAD REFERENCE DATA
target_map = {}
tree = XmlTree('extract.pairs.xml')
pair_tags = tree.DiveForAll('pair')
pair_count = 0
for pair in pair_tags:
    id1 = pair['id1'](int)
    id2 = pair['id2'](int)
    jeff2_h = pair['channel'].GetUnique('jeff2_h')(float)
    if target_map.has_key(id1):
        pass
    else:
        target_map[id1] = {}
    target_map[id1][id2] = jeff2_h


# EXPAND STRUCTURES
osio.cd('frame0')
pair_folders = get_dirs('./', 'pair_')
n_pairs = len(pair_folders)
pair_folders = sorted(pair_folders, key = lambda f: int(f.split('_')[1])*n_pairs+int(f.split('_')[2]))

xnkl_list = []
target_list = []
label_list = []
pair_id = 0
for folder in pair_folders:
    pair_id += 1
    # READ XYZ => CONFIG
    osio.cd(folder)
    osio << "Pair-ID %5d ('%s')" % (pair_id, folder) << endl
    if 'jeff2_h.txt' in os.listdir('./'):
        assert 'xnkl.array.npy' in os.listdir('./')
        assert 'xnkl.array.txt' in os.listdir('./')
        #xnkl_pair_npy = np.load('xnkl.array.npy')
        #xnkl_pair_txt = np.loadtxt('xnkl.array.txt')
        #drms = np.sum((xnkl_pair_npy-xnkl_pair_txt)**2)/(xnkl_pair_npy.shape[0])
        #osio << "... Check zero =" << drms << endl
        osio << "... Continue." << endl
        osio.cd(-1)
        continue
    # PAIR DATA
    id1 = int(folder.split('_')[1])
    id2 = int(folder.split('_')[2])
    target = target_map[id1][id2]
    # READ COORDINATES
    ase_config_list = soap.tools.ase_load_all('dim')
    assert len(ase_config_list) == 1
    config = ase_config_list[0]
    # PROCESS STRUCTURE
    structure = soap.tools.setup_structure_ase(config.config_file, config.atoms)
    for particle in structure:
        particle.sigma = 0.5*element_vdw_radius[particle.type]
        particle.weight = element_valence_order[particle.type]
        if particle.id > 28:
            #print particle.id, particle.name
            particle.weight *= -1.
    # COMPUTE SPECTRUM
    spectrum = soap.Spectrum(structure, options)
    basis = spectrum.basis
    spectrum.compute()
    spectrum.computePower()
    # EXTRACT XNKL
    x_pair = soap.PowerExpansion(spectrum.basis)
    xnkl_pair = x_pair.array
    norm = 0.
    for atomic in spectrum:
        xnkl_pair = xnkl_pair + atomic.getPower("","").array
        norm += 1.
    xnkl_pair = xnkl_pair/norm
    xnkl_pair = xnkl_pair.real
    xnkl_pair = xnkl_pair.reshape(basis.N*basis.N*(basis.L+1))
    # SAVE TO HARDDRIVE
    np.save('xnkl.array.npy', xnkl_pair)
    np.savetxt('xnkl.array.txt', xnkl_pair)
    ofs = open('jeff2_h.txt', 'w')
    ofs.write('%+1.7e\n' % target)
    ofs.close()

    #xnkl_list.append(xnkl_pair)
    #label_list.append(folder)
    #target_list.append(target_map[id1][id2])


    #atomic = spectrum.getAtomic(1, "S")
    #center = atomic.getCenter()
    #atomic.getLinear("").writeDensityOnGrid('tmp.cube', options, structure, center, True)


    osio.cd(-1)
    #break
    #raw_input('...')
    #if pair_id > 10: break

osio.root()
osio.okquit()


"""
pool = mp.Pool(processes=options.n_procs)
lock = mp.Lock()
compute_soap_partial = fct.partial(compute_soap, options=options)        
soaps = pool.map(compute_soap_partial, metas)    
pool.close()
pool.join()
for soap, meta in zip(soaps, metas):
    meta.soap = soap
"""










