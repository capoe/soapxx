#! /usr/bin/env python
import soap
import json
import numpy as np
import pickle
import h5py
import time
import datetime
import os
import psutil
import momo
log = momo.osio

options_default = {
    "spectrum.gradients": False,
    "spectrum.global": False,
    "spectrum.2l1_norm": False, # NOTE "False" emphasizes coordination, "True" distances
    "radialbasis.type" : "gaussian",
    "radialbasis.mode" : "equispaced", # NOTE Alternatives: 'equispaced' or 'adaptive'
    "radialbasis.N" : 9,
    "radialbasis.sigma": 0.5,
    "radialbasis.integration_steps": 15,
    "radialcutoff.Rc": 3.5, # NOTE Only used for 'equispaced' basis set
    "radialcutoff.Rc_width": 0.5,
    "radialcutoff.type": "shifted-cosine",
    "radialcutoff.center_weight": 1.0,
    "angularbasis.type": "spherical-harmonic",
    "angularbasis.L": 6, 
    "kernel.adaptor": "specific-unique-dmap",
    "exclude_centers": [],
    "exclude_targets": [],
    "exclude_center_ids": [],
    "exclude_target_ids": [],
    "type_list": ["C", "H", "N", "O", "S"]
}

class StructureConverter(object):
    def __init__(self, sigma=0.5, typemap=None):
        self.typemap = typemap
        self.sigma = sigma
        return
    def convert(self, config):
        return soap.tools.io.convert(
            config=config,
            tag="?",
            sigma=self.sigma,
            typemap=self.typemap)

class PowerSpectrum(object):
    # TODO Need option for normalisation of gsdmap, sdmap
    # ----
    # TODO Certain header settings will throw errors:
    # TODO e.g., store_gcmap = False and store_gsd, store_gsdmap = True
    # ----
    # TODO Global descriptor computed twice if options["spectrum.global"] 
    # TODO and store_gcmap both true
    # ----
    # NOTE Output hdf5 becomes very large with options["spectrum.gradients"] true
    settings = {
        'cxx_compute_power' : True,
        'store_cxx_serial' : False,
        'store_cmap' : False,
        'store_gcmap' : False,
        'store_sd' : True,
        'store_gsd' : False,
        'store_sdmap' : False,
        'store_gsdmap' : False,
        'dtype': 'float32' # TODO implement
    }
    struct_converter = StructureConverter()
    verbose = False
    log = log
    def __init__(self, config=None, options=None, label="?", converter=None):
        if type(config) != type(None):
            if self.verbose: self.log << self.log.mg << "Initialising power spectrum '%s'" % label << self.log.endl
        self.label = label
        self.config = config
        self.options = options
        # Cxx structure and spectrum
        self.has_structure = False
        self.structure = None
        self.has_spectrum = False
        self.spectrum = None
        self.has_spectrum_global = False
        self.spectrum_global = None
        # Density expansion coefficients
        self.has_cnlm = False
        self.cmap = None
        self.gcmap = None
        # Power expansion coefficients
        self.has_pnkl = False
        self.sd = None
        self.gsd = None
        self.gsdmap = None
        self.sdmap = None
        if type(config) != type(None):
            if converter is None:
                converter = PowerSpectrum.struct_converter
            self.compute(config=config, options=options, converter=converter)
        return
    def compute(self, config, options, converter=None):
        if self.has_cnlm: pass
        else:
            # CONVERT ASE ATOMS TO SOAPXX TOPOLOGY
            #self.config, self.structure, top, frag_bond_mat, atom_bond_mat, frag_labels, atom_labels = \
            #    soap.tools.structure_from_ase(
            #        config, 
            #        do_partition=False, 
            #        add_fragment_com=False, 
            #        log=None)
            if converter is None: converter = PowerSpectrum.converter
            self.structure = converter.convert(self.config)
            self.has_structure = True
            # COMPUTE SPECTRUM
            options_cxx = self.getCxxOptions(options)
            self.spectrum = soap.Spectrum(self.structure, options_cxx)
            if self.verbose: self.log << "[cc] Computing density expansion" << self.log.endl
            self.spectrum.compute()
            self.has_spectrum = True
            # EXTRACT CNLM ATOMIC
            if PowerSpectrum.settings["store_cmap"]:
                if self.verbose: self.log << "[py] Storing coefficient map (-> cmap)" << self.log.endl
                self.cmap = []
                for atomic in self.spectrum:
                    cmap = {}
                    types = atomic.getTypes()
                    for t in types:
                        cnlm = atomic.getLinear(t).array
                        cmap[t] = cnlm
                    self.cmap.append(cmap)
            # EXTRACT CNLM GLOBAL
            if PowerSpectrum.settings["store_gcmap"]:
                if self.verbose: self.log << "[cc] Computing global coefficient map" << self.log.endl
                self.gcmap = []
                self.spectrum_global = self.spectrum.computeGlobal()
                if self.verbose: self.log << "[py] Storing global coefficient map (-> gcmap)" << self.log.endl
                types = atomic.getTypes()
                for t in types:
                    self.gcmap.append({ t: atomic.getLinear(t).array })
                self.has_spectrum_global = True
            self.has_cnlm = True
            # COMPUTE POWER SPECTRUM?
            if PowerSpectrum.settings["cxx_compute_power"]:
                self.computePower()
        return
    def computePower(self):
        if not self.has_cnlm:
            raise RuntimeError("Cannot compute power spectrum without computing spectrum first")
        if self.has_pnkl: pass
        else:
            if self.verbose: self.log << "[cc] Computing power spectrum" << self.log.endl
            self.spectrum.computePower()
            if self.options['spectrum.gradients']:
                if self.verbose: self.log << "[cc] ... gradients ..." << self.log.endl
                self.spectrum.computePowerGradients()
            if self.options['spectrum.global']:
                if self.verbose: self.log << "[cc] ... global avg ..." << self.log.endl
                self.spectrum.deleteGlobal()
                self.spectrum.computeGlobal()
            if PowerSpectrum.settings["store_sd"]:
                if self.verbose: self.log << "[py] Storing specific descriptor (-> sd)" << self.log.endl
                adaptor = soap.soapy.kernel.KernelAdaptorFactory['specific-unique'](
                    self.spectrum.options,
                    types_global=self.options['type_list'])
                IX, IR, types = adaptor.adapt(self.spectrum, return_pos_matrix=True)
                self.sd = IX
            if PowerSpectrum.settings["store_gsd"]:
                if self.verbose: self.log << "[py] Storing global specific descriptor (-> gsd)" << self.log.endl
                adaptor = soap.soapy.kernel.KernelAdaptorFactory['global-specific'](
                    self.spectrum.options,
                    types_global=self.options['type_list'])
                IX, IR, types = adaptor.adapt(self.spectrum, return_pos_matrix=True)
                self.gsd = IX
            if PowerSpectrum.settings["store_sdmap"]:
                if self.verbose: self.log << "[py] Storing specific descriptor map (-> sdmap)" << self.log.endl
                adaptor = soap.soapy.kernel.KernelAdaptorFactory['specific-unique-dmap'](None)
                IX, IR, types = adaptor.adapt(self.spectrum, return_pos_matrix=True)
                self.sdmap = IX
            if PowerSpectrum.settings["store_gsdmap"]:
                if self.verbose: self.log << "[py] Storing global specific descriptor map (-> gsdmap)" << self.log.endl
                if not self.has_spectrum_global:
                    raise RuntimeError("Cannot store global d-map without computing "+\
                        +"global descriptor first.")
                # Adapt spectrum
                adaptor = soap.soapy.kernel.KernelAdaptorFactory['global-specific-dmap'](None)
                IX, IR, types = adaptor.adapt(self.spectrum, return_pos_matrix=True)
                self.gsdmap = IX
            self.has_pnkl = True
        return
    def getCxxOptions(self, options):
        options_cxx = soap.Options()
        for key, val in options.items():
            if type(val) == list: continue
            options_cxx.set(key, val)
        # Exclusions
        excl_targ_list = options['exclude_targets']
        excl_cent_list = options['exclude_centers']
        options_cxx.excludeCenters(excl_cent_list)
        options_cxx.excludeTargets(excl_targ_list)
        excl_targ_id_list = options['exclude_target_ids']
        excl_cent_id_list = options['exclude_center_ids']
        options_cxx.excludeCenterIds(excl_cent_id_list)
        options_cxx.excludeTargetIds(excl_targ_id_list)
        return options_cxx
    def save(self, hdf5_handle):
        g = hdf5_handle
        # Class settings
        g.attrs.update(self.settings)
        # Class attributes
        h = g.create_group("class")
        h.attrs["label"] = self.label
        if self.settings["store_cxx_serial"]:
            if self.verbose: self.log << "[h5] Writing cxx serial" << self.log.endl
            # Prune pid data if not required to compute gradients
            prune_pid_data = False if self.options['spectrum.gradients'] else True
            cxx_serial = self.spectrum.saves(prune_pid_data)
            h = g.create_dataset("cxx_serial", data=np.void(cxx_serial))
        if self.settings["store_cmap"]:
            if self.verbose: self.log << "[h5] Writing coefficient map" << self.log.endl
            h = g.create_group("cmap")
            for idx, cmap in enumerate(self.cmap):
                hh = h.create_group('%d' % idx)
                for key in cmap:
                    hh.create_dataset(key, data=cmap[key], compression='gzip')
        if self.settings["store_gcmap"]:
            if self.verbose: self.log << "[h5] Writing global coefficient map" << self.log.endl
            h = g.create_group("gcmap")
            for idx, gcmap in enumerate(self.gcmap):
                hh = h.create_group('%d' % idx)
                for key in gcmap:
                    hh.create_dataset(key, data=gcmap[key], compression='gzip')
        if self.settings["store_sdmap"]:
            if self.verbose: self.log << "[h5] Writing descriptor map" << self.log.endl
            h = g.create_group('sdmap')
            for idx, sdmap in enumerate(self.sdmap):
                hh = h.create_group('%d' % idx)
                for key in sdmap:
                    hh.create_dataset(key, data=sdmap[key], compression='gzip')
        if self.settings["store_gsdmap"]:
            if self.verbose: self.log << "[h5] Writing global descriptor map" << self.log.endl
            h = g.create_group('gsdmap')
            for idx, gsdmap in enumerate(self.gsdmap):
                hh = h.create_group('%d' % idx)
                for key in gsdmap:
                    hh.create_dataset(key, data=gsdmap[key], compression='gzip')
        if self.settings["store_sd"]:
            if self.verbose: self.log << "[h5] Writing descriptor matrix" << self.log.endl
            g.create_dataset('sd', data=self.sd, compression='gzip')
        if self.settings["store_gsd"]:
            if self.verbose: self.log << "[h5] Writing global descriptor matrix" << self.log.endl
            g.create_dataset('gsd', data=self.gsd, compression='gzip')
        return self
    def load(self, hdf5_handle):
        g = hdf5_handle
        # SETTINGS
        self.settings = g.attrs
        self.label = g["class"].attrs["label"]
        if self.verbose: self.log << self.log.mb << "Loading power spectrum '%s'" % self.label << self.log.endl
        # HELPER FUNCTIONS
        def load_dict_array_data(h):
            D = []
            for idx in range(len(h)):
                hh = h["%d" % idx]
                d = { t: hh[t].value for t in hh }
                D.append(d)
            return D
        def load_descriptor_map_matrix(h):
            D = soap.soapy.kernel.DescriptorMapMatrix()
            for idx in range(len(h)):
                d = soap.soapy.kernel.DescriptorMap()
                hh = h['%d' % idx]
                for t in hh: d[t] = hh[t].value
                D.append(d)
            return D
        # LOAD FIELDS
        if self.settings["store_cxx_serial"]:
            if self.verbose: self.log << "[h5] Loading cxx serial and spectrum" << self.log.endl
            cxx_serial = g["cxx_serial"].value.tostring()
            self.spectrum = soap.Spectrum()
            self.spectrum.loads(cxx_serial)
            self.has_spectrum = True
        # DENSITY EXPANSION COEFFICIENTS
        if self.settings["store_cmap"]:
            if self.verbose: self.log << "[h5] Loading coefficient map" << self.log.endl
            self.cmap = load_dict_array_data(g["cmap"])
        if self.settings["store_gcmap"]:
            if self.verbose: self.log << "[h5] Loading global coefficient map" << self.log.endl
            self.cmap = load_dict_array_data(g["gcmap"])
        # POWER SPECTRUM COEFFICIENTS
        if self.settings["store_sdmap"]:
            if self.verbose: self.log << "[h5] Loading descriptor map" << self.log.endl
            self.sdmap = load_descriptor_map_matrix(g["sdmap"])
        if self.settings["store_gsdmap"]:
            if self.verbose: self.log << "[h5] Loading global descriptor map" << self.log.endl
            self.gsdmap = load_descriptor_map_matrix(g["gsdmap"])
        if self.settings["store_sd"]:
            if self.verbose: self.log << "[h5] Loading descriptor matrix" << self.log.endl
            self.sd = g["sd"].value
        return self

def test(log, do_verify):
    def verify(spec):
        if do_verify:
            k_glob = spec.gsdmap.dot(spec.gsdmap)
            K_mat = spec.sdmap.dot(spec.sdmap)
            print("kernel global")
            print(k_glob)
            print("kernel pairwise")
            print(K_mat)
        return
    # Configure storage settings
    PowerSpectrum.verbose = True
    PowerSpectrum.settings = {
        'cxx_compute_power' : True, # 'False' intended for users that compute their own fingerprint from the c_nlm-coeffs.
        'store_cxx_serial' : False, # binary serialisation string of C++ spectrum object
        'store_cmap' : True, # map of density expansion coeffs
        'store_gcmap' : True, # global cmap
        'store_sd' : False, # descriptor as a vector [ e.g. 0-500 C:C, 501-1000 C:H, ... ], length = S*N*(L+1) + S*N*(N-1)*(L+1)/2 + S*(S-1)*N*N*(L+1)/2
        'store_gsd' : False, # global descriptor
        'store_sdmap' : True, # descriptor map [ { "C:C": vec_cc, "C:H": vec_ch, ... }, { second environ. }, ... ]
        'store_gsdmap' : True, # global map [ { } ]
        'dtype': 'float32' # TODO implement
    }
    # Load structures
    options = options_default
    configs = soap.tools.io.read('data/structures.xyz', ':')
    # PowerSpectrum: compute and save
    for idx, config in enumerate(configs):
        spec = PowerSpectrum(config, options, "cfg-%07d" % idx)
        # How to access global descriptor map:
        print "C:C shape =", spec.gsdmap[0]["C:C"].shape
        # How to compute dot products:
        print "kernel =", spec.gsdmap.dot(spec.gsdmap)
        spec.save(h5py.File("out/out-cfg-%07d.hdf5" % idx, "w"))
    # PowerSpectrum: load and verify
    for idx, config in enumerate(configs):
        spec = PowerSpectrum().load(h5py.File("out/out-cfg-%07d.hdf5" % idx, "r"))
        verify(spec)

if __name__ == "__main__":
    np.set_printoptions(precision=2)
    soap.silence()
    test(log=log, do_verify=False)
    log.okquit('All done')

