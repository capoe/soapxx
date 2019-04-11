#! /usr/bin/env python
import soap
import numpy as np
import json
import os
import momo
import copy
from sklearn.kernel_ridge import KernelRidge
log = momo.osio

PATH = None
SETTINGS = None
TEMPDIR = 'temp'
PRECALC_SOAPS = {}

def configure_embedding(path):
    global PATH
    global SETTINGS
    global TEMPDIR
    global PRECALC_SOAPS
    PATH = path
    SETTINGS = soap.soapy.json_load_utf8(open(os.path.join(PATH, "settings.json")))
    TEMPDIR = 'temp'
    PRECALC_SOAPS = {}
    return

def initialize_embedding(rerun):
    for section_name, section in SETTINGS["channels"].iteritems():
        if section["embedding_options"]["type"] == "environment-specific":
            parametrize_environment_specific(
                settings=section, 
                rerun=rerun)

def check_is_configured():
    if SETTINGS is None:
        raise RuntimeError("Need to set data path via <soap.soapy.embedding.configure_embedding>.")
    return

# ==================
# SOAP CONFIGURATION
# ==================

def soap_configure_default(typemap={}, types=[]):
    # Logging
    soap.silence()
    soap.soapy.wrap.PowerSpectrum.verbose = True
    # Storage
    soap.soapy.wrap.PowerSpectrum.settings = {
        'cxx_compute_power' : True,
        'store_cxx_serial' : False,
        'store_cmap' : False,
        'store_gcmap' : False,
        'store_sd' : False,
        'store_gsd' : False,
        'store_sdmap' : False,
        'store_gsdmap' : False,
        'dtype': 'float64' # NOTE Not (yet) used
    }
    # Use (alchemical) type embedding
    if len(types) and len(typemap):
        raise ValueError("Both types and typemap non-zero, can only specify one.")
    elif len(types):
        soap.encoder.clear()
        for c in types: soap.encoder.add(c)
    elif len(typemap):
        log << "Using %d-dimensional type embedding" % len(typemap["channels"]) << log.endl
        soap.encoder.clear()
        for c in typemap["channels"]: soap.encoder.add(c)
        PowerSpectrum.struct_converter = soap.soapy.wrap.StructureConverter(typemap=typemap)
    log << "Using type encoder with %d types:" % (len(soap.encoder.types())) << ",".join(soap.encoder.types()) << log.endl
    return

def soap_evaluate(configs, options, output_file):
    dset = soap.DMapMatrixSet()
    for cidx, config in enumerate(configs):
        # Handle exclusions
        excl_idcs = []
        ts = np.array(config.symbols)
        for excl in options["exclude_centers"]:
            excl_idcs_add = np.where(ts == excl)[0] 
            excl_idcs = excl_idcs + list(excl_idcs_add)
        excl_idcs = sorted(excl_idcs)
        options["exclude_center_ids"] = [ (i+1) for i in excl_idcs ]
        # Compute
        spectrum = soap.soapy.PowerSpectrum(
            config=config, 
            options=options,
            label="config-%d" % cidx)
        dmap = spectrum.exportDMapMatrix()
        log << "Config %d: %d centers" % (cidx, dmap.rows) << log.endl
        dset.append(dmap)
    dset.save(output_file)
    return dset

def get_cxx_kernel(options):
    options_cxx = soap.Options()
    log << "Kernel:" << log.endl
    for k,v in options.iteritems():
        log << "- Setting %s=%s" % (k, str(v)) << log.endl
        options_cxx.set(k,v)
    kernel = soap.Kernel(options_cxx)
    return kernel

def kernel_evaluate(dset, options, output_file):
    kernel = get_cxx_kernel(options)
    if options["topkernel_type"] == "average":
        log << "Reducing DMapMatrices (::sum)" << log.endl
        for i in range(len(dset)): dset[i].sum()
    symmetric = True
    K = kernel.evaluate(dset, dset, symmetric, "float64")
    z = 1./np.sqrt(K.diagonal())
    K = K*np.outer(z,z)
    K = K+K.T
    np.fill_diagonal(K, 1.0)
    np.save(output_file, K)
    log << log.mg << "Saved kernel =" << log.endl
    log << K << log.endl
    return K

def kernel_attribute(dset1, dset2, kernel_options, kweights, xi):
    delta_Y = []
    if kernel_options["topkernel_type"] == "average":
        for i in range(len(dset2)):
            dset2[i].sum()
            dset2[i].normalize()
    kernel = get_cxx_kernel(kernel_options)
    soap.silence()
    for i in range(len(dset1)):
        log << log.back << "Attribute" << i << log.flush
        Ki = kernel.attributeLeft(dset1[i], dset2, "float64")
        dset_i = soap.DMapMatrixSet()
        dset_i.append(dset1[i])
        # TO CHECK
        # >>> K = kernel.evaluate(dset_i, dset2, False, "float64")
        # >>> print Ki
        # >>> print np.sum(Ki, axis=0), "==", K
        # >>> raw_input('...')
        Kii = kernel.evaluate(dset_i, dset_i, True, "float64")
        Ki = Ki/np.sum(Kii)**0.5
        # Account for top-level exponent xi
        Ki = Ki*np.sum(Ki, axis=0)**(xi-1)
        delta_Yi = Ki.dot(kweights)
        delta_Y.append(list(delta_Yi))
    log << log.endl
    return delta_Y

# =====
# APPLY
# =====

def apply(configs, channels, options):
    check_is_configured()
    for c in configs:
        config_reset_weights(c)
    for t in channels:
        type_settings = SETTINGS["channels"][t]
        type_options = options[t]
        if type_settings["embedding_options"]["type"] == "environment-specific":
            configs = apply_environment_specific(configs, type_settings, type_options)
        elif type_settings["embedding_options"]["type"] == "element-specific":
            configs = apply_element_specific(configs, type_settings, type_options)
        elif type_settings["embedding_options"]["type"] == "one-hot":
            configs = apply_onehot(configs, type_settings, type_options)
        else: raise ValueError(type_settings["embedding_options"]["type"])
    return configs

def apply_environment_specific(configs, settings, options):
    log << log.mg << "Apply" << settings["embedding_options"]["channel_name"] << "to %d configs" % (
        len(configs)) << log.endl
    # LOAD MODEL
    log << "Loading model ..." << log.endl
    soap_version = settings["soap_options_ref"]
    soap_options = SETTINGS["soap_options"][soap_version]
    soap_excls = set(soap_options["exclude_centers"])
    kernel_options = settings["kernel_options"]
    target_key = settings["regression_options"]["target_key"]
    regr_options = settings["regression_options"]
    paths = settings["paths"]
    dset2 = soap.DMapMatrixSet(os.path.join(PATH, paths["soap_file"]))
    targets = np.load(os.path.join(PATH, paths["targets_file"]))
    kweights = np.load(os.path.join(PATH, paths["weights_file"]))
    y_predict = np.load(os.path.join(PATH, paths["pred_file"]))
    yrange = json.load(open(os.path.join(PATH, paths["range_file"])))
    yrange = np.concatenate(yrange)
    # ATTRIBUTE
    dset1 = soap.DMapMatrixSet(PRECALC_SOAPS[soap_version])
    delta_Ys = kernel_attribute(dset1, dset2, kernel_options, kweights, regr_options["xi"])
    assert len(delta_Ys) == len(dset1)
    # ASSIGN BASIS WEIGHTS
    w_avg = np.average(yrange)
    w_std = np.std(yrange)
    z = options["weight"]
    log << "Channel normalization is" << z << log.endl
    channel_name = settings["embedding_options"]["channel_name"]
    basis = [ GaussianBasisFct(w_avg+centre*w_std, sigma*w_std) for centre, sigma in zip(
        options["centres"], options["sigmas"]) ]
    log << "- Using %d basis functions:" % (len(basis)) << log.endl
    for b in basis: log << "    Centre, width = %1.4f, %1.4f" % (b.centre, b.sigma) << log.endl
    channel_names = [ "%s%d" % (channel_name, i) for i in range(len(basis)) ]
    for cidx, config in enumerate(configs):
        log << log.back << "- Config %d" % cidx << log.flush
        types = config.symbols
        type_coords = []
        delta_Y = delta_Ys[cidx]
        attr_idx = 0
        sum_w = 0.0
        for t in types:
            if t in soap_excls:
                type_coords.append({})
            else:
                w = delta_Y[attr_idx]
                sum_w += w
                wb = np.array([ fct(w) for fct in basis ])
                if z is not None:
                    wb = norm_weights(wb, z)
                type_coords.append({ channel_names[i]: wb[i] for i in range(len(basis)) })
                attr_idx += 1
        # TO CHECK:
        # >>> log << sum_w << "==" << y_predict[cidx] << log.endl
        config_add_weights(config, type_coords)
    log << log.endl
    return configs
    
def apply_element_specific(configs, settings, options):
    log << log.mg << "Apply" << settings["embedding_options"]["channel_name"] << "to %d configs" % (
        len(configs)) << log.endl
    channel_name = settings["embedding_options"]["channel_name"]
    weights_file = os.path.join(PATH, settings["paths"]["weights_file"])
    log << "Reading weights from '%s'" % weights_file << log.endl
    element_weights = soap.soapy.json_load_utf8(open(weights_file))
    all_weights = [ v for k,v in element_weights.iteritems() ]
    all_weights = np.array(filter(lambda v: v is not None, all_weights))
    w_avg = np.average(all_weights)
    w_std = np.std(all_weights)
    z = options["weight"]
    log << "Channel normalization is" << z << log.endl
    basis = [ GaussianBasisFct(w_avg+centre*w_std, sigma*w_std) for centre, sigma in zip(
        options["centres"], options["sigmas"]) ]
    log << "- Using %d basis functions:" % (len(basis)) << log.endl
    for b in basis: log << "    Centre, width = %1.4f, %1.4f" % (b.centre, b.sigma) << log.endl
    channel_names = [ "%s%d" % (channel_name, i) for i in range(len(basis)) ]
    for cidx, config in enumerate(configs):
        log << log.back << "- Config %d" % cidx << log.flush
        types = config.symbols
        type_coords = []
        for t in types:
            if not t in element_weights:
                type_coords.append({})
            else:
                w = element_weights[t]
                wb = np.array([ fct(w) for fct in basis ])
                if z is not None:
                    wb = norm_weights(wb, z)
                type_coords.append({ channel_names[i]: wb[i] for i in range(len(basis)) })
        config_add_weights(config, type_coords)
    log << log.endl
    return configs

def apply_onehot(configs, settings, options):
    log << log.mg << "Apply" << settings["embedding_options"]["channel_name"] << "to %d configs" % (
        len(configs)) << log.endl
    z = options["weight"]
    log << "Channel normalization is" << z << log.endl
    for cidx, config in enumerate(configs):
        types = config.symbols
        type_coords = []
        for t in types:
            w = np.array([1.])
            w = norm_weights(w, z)
            type_coords.append({ t: w[0] })
        config_add_weights(config, type_coords)
    return configs

# ================
# WEIGHT EMBEDDING
# ================

class GaussianBasisFct(object):
    def __init__(self, centre, sigma):
        self.centre = centre
        self.sigma = sigma
    def __call__(self, x):
        return np.exp(-(x-self.centre)**2/(2*self.sigma**2))

def get_weight(w):
    return np.sum(np.outer(w,w))

def norm_weights(w, z):
    wz = w*z**0.5 / np.sum(np.outer(w,w))**0.5
    return wz

def config_norm_weights(config, z):
    for idx, c in enumerate(config.atoms):
        ws = np.array([ v for k,v in config.info["weights"][idx].iteritems() ])
        scale = z**0.5/get_weight(ws)**0.5
        for k in config.info["weights"][idx]:
            config.info["weights"][idx][k] *= scale

def configs_norm_weights(configs, z):
    for config in configs:
        config_norm_weights(config, z)

def config_reset_weights(config):
    config.info["weights"] = [ {} for i in range(len(config.atoms)) ]

def config_add_weights(config, type_coords):
    assert len(config) == len(type_coords)
    for idx, w in enumerate(config.info["weights"]):
        config.info["weights"][idx].update(type_coords[idx])

def configs_stringify_weights(configs):
    for config in configs:
        config.info["weights"] = json.dumps(
            config.info["weights"], separators=(",",":")).replace('"',"_")

def configs_load_weights(configs):
    for config in configs:
        config.info["weights"] = soap.soapy.json_loads_utf8(
            config.info["weights"].replace("_",'"'))
        
def embed(configs, channels, options):
    configs = apply(configs, channels, options)
    configs_norm_weights(configs, options["norm"])
    configs_stringify_weights(configs)
    return configs

# =====================
# MODEL PARAMETRIZATION
# =====================

def precalculate_soap(config_file, rerun):
    check_is_configured()
    log >> 'mkdir -p %s' % TEMPDIR
    configs = soap.tools.io.read(config_file)
    soap_types = SETTINGS["soap_types"]
    for version, soap_options in SETTINGS["soap_options"].iteritems():
        tmp_file = os.path.join(TEMPDIR, "%s_%s.soap" % (os.path.basename(config_file).replace('.xyz',''), version))
        PRECALC_SOAPS[version] = tmp_file
        log << "Path to precalculated SOAP '%s' = %s" % (version, tmp_file) << log.endl
        if rerun or not os.path.isfile(tmp_file):
            soap_configure_default(types=soap_types)
            dset = soap_evaluate(configs, soap_options, tmp_file)
    return

def parametrize_environment_specific(settings, rerun):
    channel_name = settings["embedding_options"]["channel_name"]
    log << log.mg << "Parametrizing" << channel_name << "model" << log.endl
    soap_types = SETTINGS["soap_types"]
    log << "Particle SOAP types are" << ", ".join(soap_types) << log.endl
    # PATHS - for example:
    # { "xyz_file": "data_esol/structures.xyz",
    #   "soap_file": "data_esol/structures.soap",
    #   "kmat_file": "data_esol/kernel.npy",
    #   "targets_file": "data_esol/targets.npy",
    #   "range_file": "data_esol/range.json",
    #   "weights_file": "data_esol/weights.npy" }
    paths = copy.deepcopy(settings["paths"])
    for p,v in paths.iteritems():
        paths[p] = os.path.join(PATH, v)
        log << "Path to %s = %s" % (p, paths[p]) << log.endl
    configs = soap.tools.io.read(paths["xyz_file"])
    # SOAP
    soap_options = SETTINGS["soap_options"][settings["soap_options_ref"]]
    if rerun or not os.path.isfile(paths["soap_file"]):
        log << "Make target: %s" % paths["soap_file"] << log.endl
        soap_configure_default(types=soap_types)
        dset = soap_evaluate(configs, soap_options, paths["soap_file"])
    else:
        log << "Load target: %s" % paths["soap_file"] << log.endl
        dset = soap.DMapMatrixSet(paths["soap_file"])
    # KERNEL
    kernel_options = settings["kernel_options"]
    if rerun or not os.path.isfile(paths["kmat_file"]):
        log << "Make target: %s" % paths["kmat_file"] << log.endl
        K = kernel_evaluate(dset, kernel_options, paths["kmat_file"])
    else:
        log << "Load target: %s" % paths["kmat_file"] << log.endl
        K = np.load(paths["kmat_file"])
    # TARGETS
    target_key = settings["regression_options"]["target_key"]
    if rerun or not os.path.isfile(paths["targets_file"]):
        log << "Make target: %s" % paths["targets_file"] << log.endl
        targets = np.array([float(c.info[target_key]) for c in configs])
        np.save(paths["targets_file"], targets)
    else:
        log << "Load target: %s" % paths["targets_file"] << log.endl
        targets = np.load(paths["targets_file"])
    # MODEL
    regr_options = settings["regression_options"]
    if rerun or not os.path.isfile(paths["weights_file"]):
        log << "Make target: %s" % paths["weights_file"] << log.endl
        y_avg = np.average(targets)
        krr = KernelRidge(
            alpha=regr_options["lreg"],
            kernel='precomputed')
        krr.fit(K**regr_options["xi"], targets)
        y_predict = krr.predict(K**regr_options["xi"])
        kweights = krr.dual_coef_
        np.save(paths["weights_file"], kweights)
        np.save(paths["pred_file"], y_predict)
    else:
        log << "Load target: %s" % paths["weights_file"] << log.endl
        kweights = np.load(paths["weights_file"])
        y_predict = np.load(paths["pred_file"])
    if rerun or not os.path.isfile(paths["range_file"]):
        dset_attr = soap.DMapMatrixSet(paths["soap_file"])
        delta_Ys = kernel_attribute(dset_attr, dset, kernel_options, kweights, regr_options["xi"])
        json.dump(delta_Ys, open(paths["range_file"], "w"))
    else:
        delta_Ys = json.load(open(paths["range_file"]))

