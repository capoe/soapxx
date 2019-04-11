import numpy as np
import soap
log = soap.log

def assert_zero(z, eps=1e-5):
    if z > eps: raise ValueError(z)
    else: log << "+" << log.flush

def configure_default(typemap={}, types=[]):
    # Logging
    soap.silence()
    soap.soapy.wrap.PowerSpectrum.verbose = False
    # Descriptor options
    options_soap = {
        "spectrum.2d": False,
        "spectrum.gradients": False,
        "spectrum.global": False,
        "spectrum.2l1_norm": False, # NOTE "False" emphasizes coordination, "True" distances
        "radialbasis.type" : "gaussian",
        "radialbasis.mode" : "adaptive", # NOTE Alternatives: 'equispaced' or 'adaptive'
        "radialbasis.N" : 9,
        "radialbasis.sigma": 0.5,
        "radialbasis.integration_steps": 15,
        "radialcutoff.Rc": 3.5, # NOTE Only used for 'equispaced' basis set
        "radialcutoff.Rc_width": 0.5,
        "radialcutoff.type": "heaviside",
        "radialcutoff.center_weight": 1.0,
        "angularbasis.type": "spherical-harmonic",
        "angularbasis.L": 6, 
        "kernel.adaptor": "specific-unique-dmap",
        "exclude_centers": ["H"],
        "exclude_targets": [],
        "exclude_center_ids": [],
        "exclude_target_ids": []
    }
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
    return options_soap

def test_dmap_convolve():
    log << log.mg << "<test_dmap_convolve>" << log.endl
    configs = soap.tools.io.read('structures.xyz')
    soap_options = configure_default()
    dset = soap.DMapMatrixSet()
    for idx, config in enumerate(configs):
        spec = soap.soapy.PowerSpectrum(config, soap_options, "S%d" % idx)
        soap.toggle_logger()
        dmap_x = spec.exportDMapMatrix()
        dmap_q = soap.DMapMatrix()
        dmap_q.appendCoherent(spec.spectrum)
        dmap_q.convolve(9, 6)
        dset.append(dmap_q)
        kxx = dmap_x.dot(dmap_x, "float64")
        kqq = dmap_q.dot(dmap_q, "float64")
        kxq = dmap_x.dot(dmap_q, "float64")
        kqx = dmap_q.dot(dmap_x, "float64")
        assert_zero(np.max(np.abs(kxx-kqq)))
        assert_zero(np.max(np.abs(kxx-kxq)))
        assert_zero(np.max(np.abs(kxx-kqx)))
        soap.toggle_logger()
        if idx == 10: break
    log << log.endl
    log << log.mg << "All passed" << log.endl

test_dmap_convolve()
