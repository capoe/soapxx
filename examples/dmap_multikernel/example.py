#! /usr/bin/env python
import numpy as np
import soap
import os
import json
import glob
import utils
import h5py
import copy
log = soap.log

def evaluate_soap(configs, options):
    dset = soap.DMapMatrixSet()
    for cidx, config in enumerate(configs):
        spectrum = soap.soapy.PowerSpectrum(
            config=config, 
            options=options,
            label="config-%d" % cidx)
        dmap = spectrum.exportDMapMatrix()
        log << "Config %d: %d centers" % (cidx, dmap.rows) << log.endl
        dset.append(dmap)
    return dset

def evaluate_kernel(dset, options=None, kernel=None):
    assert not (options == kernel == None)
    if kernel is None: kernel = soap.Kernel(options)
    symmetric = True
    kernel.evaluateAll(dset, dset, symmetric)
    for slot in range(kernel.n_output):
        K = kernel.getOutput(slot, "float64")
        z = 1./np.sqrt(K.diagonal())
        K = K*np.outer(z,z)
        log << "Kernel: Output slot" << slot << log.endl
        log << K << log.endl

if __name__ == "__main__":
    configs = soap.tools.io.read('data/structures.xyz')

    # Compute SOAP descriptors
    options = soap.soapy.configure_default()
    dset = evaluate_soap(configs, options)

    # Multikernel: Variant A
    log << log.mg << "Variant A" << log.endl
    kernel_options = soap.Options()
    kernel_options.set("basekernel_type", "dot")
    kernel_options.set("base_exponent", 3.)
    kernel_options.set("base_filter", False)
    kernel_options.set("topkernel_type", "average;canonical;rematch")
    kernel_options.set("canonical_beta", 0.5)
    kernel_options.set("rematch_gamma", 0.05)
    kernel_options.set("rematch_eps", 1e-6)
    kernel_options.set("rematch_omega", 1.0)
    evaluate_kernel(dset, options=kernel_options)

    # Multikernel: Variant B
    log << log.mg << "Variant B" << log.endl
    kernel_options.set("topkernel_type", "")
    kernel = soap.Kernel(kernel_options)
    for beta in [ 0.01, 0.05, 0.1, 0.5, 1.0, 10.0, 100.0 ]:
        kernel_options.set("topkernel_type", "canonical")
        kernel_options.set("canonical_beta", beta)
        kernel.addTopkernel(kernel_options)
    evaluate_kernel(dset, kernel=kernel)

