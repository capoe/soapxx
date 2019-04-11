import numpy as np
import soap
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

def test_dmap_gradients_single(config, soap_options, centre_idx, dim, gidx, h):
    # Analytical gradients from spectrum in original configuration
    spec = soap.soapy.PowerSpectrum(config, soap_options)
    X = spec.exportDMapMatrix(coherent=False)
    x = X[centre_idx]
    grads = [ g for g in x.gradients ]
    pids = [ g.pid for g in grads ]
    pid = grads[gidx].pid
    g = grads[gidx][dim]
    # Numerical gradients via step translation
    config.positions[pid-1][dim] += h
    spech = soap.soapy.PowerSpectrum(config, soap_options)
    Xh = spech.exportDMapMatrix(coherent=False)
    xh = Xh[centre_idx]
    # Compare
    xh.add(x, -1.)
    xh.multiply(1./h)
    shh = xh.dot(xh)
    sgg = g.dot(g)
    sgh = g.dot(xh)
    xh.add(g, -1.)
    dgh = xh.dot(xh)
    assert_equal(shh-sgg, 0., 1e-5)
    assert_equal(shh-sgh, 0., 1e-5)
    assert_equal(dgh, 0., 1e-7)
    log << "(overlaps= %+1.7e %+1.7e %+1.7e %+1.7e)" % (shh, sgg, sgh, dgh) << log.endl
    return

def test_dmap_gradients():
    log << log.mg << "<test_dmap_gradients>" << log.endl
    configs = soap.tools.io.read('structures.xyz')
    soap_options = configure_default()
    for centre_idx, dim, gidx in zip(
        [ 0, 0, 0, 2, 2 ],
        [ 0, 1, 2, 0, 2 ],
        [ 0, 0, 2, 0, 1 ]):
        test_dmap_gradients_single(configs[0], soap_options, centre_idx, dim, gidx, h=0.00001)
    log << log.mg << "All passed" << log.endl
    
test_dmap_gradients()

