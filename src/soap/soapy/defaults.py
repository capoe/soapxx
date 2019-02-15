import soap

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

