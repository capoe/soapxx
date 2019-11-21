import soap

def convert_json_to_cxx(options):
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

def default_options_gylm(as_cxx=True):
    options = {
        "spectrum.2d": False,
        "spectrum.global": False,
        "spectrum.gradients": False,
        "spectrum.2l1_norm": False,
        "radialbasis.type": "gylm",
        "radialbasis.rmin": 0.0,
        "radialbasis.rmax": 4.5,
        "radialbasis.smin": 0.5,
        "radialbasis.smax": 1.0,
        "radialbasis.wcentre": 1.0,
        "radialbasis.wscale": 0.5,
        "radialbasis.ldamp": 4.,
        "radialbasis.sigma": 0.5,
        "radialcutoff.Rc": 7.5,
        "radialcutoff.Rc_width": 0.5,
        "radialcutoff.type": "heaviside",
        "radialcutoff.center_weight": 1.0,
        "angularbasis.type": "spherical-harmonic",
        "angularbasis.L": 6,
        "exclude_centers": ["H"],
        "exclude_targets": [],
        "exclude_target_ids": [],
        "exclude_center_ids": []
    }
    if as_cxx: options = convert_json_to_cxx(options)
    return options

def default_options():
    options = soap.Options()
    options.set("spectrum.global", False)
    options.set("spectrum.gradients", False)
    options.set("spectrum.2l1_norm", False)
    options.set("radialbasis.type", "gaussian")
    options.set("radialbasis.mode", "adaptive")
    options.set("radialbasis.N", 9)
    options.set("radialbasis.sigma", 0.5)
    options.set("radialbasis.integration_steps", 15)
    options.set("radialcutoff.Rc", 3.5)
    options.set("radialcutoff.Rc_width", 0.5)
    options.set("radialcutoff.type", "heaviside")
    options.set("radialcutoff.center_weight", 1.0)
    options.set("angularbasis.type", "spherical-harmonic")
    options.set("angularbasis.L", 6)
    options.set("basekernel.type", "dot")
    options.set("basekernel.dot.exponent", 3.0)
    options.set("basekernel.dot.coefficient", 1.0)
    options.set("topkernel.type", "rematch")
    options.set("topkernel.rematch.gamma", 0.01)
    options.excludeCenters(["H"])
    options.excludeTargets([])
    return options

