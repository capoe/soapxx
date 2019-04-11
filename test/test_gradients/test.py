import numpy as np
import soap
import scipy.stats
import sklearn.metrics
log = soap.log

def assert_equal(z, target, eps=1e-5):
    if np.abs(z-target) > eps: raise ValueError(z)
    else: log << "+" << log.flush

def configure_default(typemap={}, types=[]):
    # Logging
    soap.silence()
    soap.soapy.wrap.PowerSpectrum.verbose = False
    # Descriptor options
    options_soap = {
        "spectrum.2d": False,
        "spectrum.gradients": True,
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


def test_gradients_single(config, soap_options, atm_idx, spatial_dim, t1, t2, h):
    gxnkl_pid0 = None
    xnkl_pid0 = None
    xnkl_pid1 = None
    # Analytical gradients from spectrum in original configuration
    spec0 = soap.soapy.PowerSpectrum(config, soap_options)
    for a in spec0.spectrum:
        xnkl_pid0 = a.getPower(t1, t2)
        gxnkl_pid0 = a.getPowerGrad(atm_idx+1, t1, t2)
        break
    # Numerical gradients via step translation
    config.positions[atm_idx][spatial_dim] += h
    spec1 = soap.soapy.PowerSpectrum(config, soap_options)
    for a in spec1.spectrum:
        xnkl_pid1 = a.getPower(t1, t2)
        break
    # Compare
    if spatial_dim == 0:
        gxnkl_pid0_ana = gxnkl_pid0.getArrayGradX().flatten().real
    elif spatial_dim == 1:
        gxnkl_pid0_ana = gxnkl_pid0.getArrayGradY().flatten().real
    elif spatial_dim == 2:
        gxnkl_pid0_ana = gxnkl_pid0.getArrayGradZ().flatten().real
    else: raise ValueError()
    gxnkl_pid0_num = (xnkl_pid1.getArray().flatten().real-xnkl_pid0.getArray().flatten().real)/h
    r2 = sklearn.metrics.r2_score(gxnkl_pid0_ana, gxnkl_pid0_num)
    maxabsdev = np.max(np.abs(gxnkl_pid0_num-gxnkl_pid0_ana))
    assert_equal(r2, 1., 1e-5)
    return

def test_gradients():
    log << log.mg << "<test_gradients>" << log.endl
    configs = soap.tools.io.read('structures.xyz')
    soap_options = configure_default()
    for atm_idx, spatial_dim, t1, t2 in zip(
        [ 1, 1, 1, 2, 4 ],
        [ 0, 1, 2, 0, 1 ],
        [ "C", "C", "C", "C", "H" ],
        [ "C", "C", "H", "H", "Cl" ]):
        h = 0.00001
        test_gradients_single(configs[0], soap_options, atm_idx, spatial_dim, t1, t2, h)
    log << log.endl
    log << log.mg << "All passed" << log.endl
    return True

test_gradients()


