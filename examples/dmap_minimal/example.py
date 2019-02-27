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

def evaluate_soap(configs, options, output_file):
    dset = soap.DMapMatrixSet()
    for cidx, config in enumerate(configs):
        spectrum = soap.soapy.PowerSpectrum(
            config=config, 
            options=options,
            label="config-%d" % cidx)
        dmap = spectrum.exportDMapMatrix()
        log << "Config %d: %d centers" % (cidx, dmap.rows) << log.endl
        dset.append(dmap)
    dset.save(output_file)
    return dset

def evaluate_kernel(dset, options, output_file):
    kernel = soap.Kernel(options)
    symmetric = True
    K = kernel.evaluate(dset, dset, symmetric, "float64")
    z = 1./np.sqrt(K.diagonal())
    K = K*np.outer(z,z)
    np.save(output_file, K)
    log << log.mg << "Saved kernel =" << log.endl
    log << K << log.endl
    return K

if __name__ == "__main__":
    configs = soap.tools.io.read('data/structures.xyz')

    # Compute SOAP descriptors
    options = soap.soapy.configure_default()
    dset = evaluate_soap(configs, options, 'data/structures.soap')

    # Compute molecular kernel
    dset = soap.DMapMatrixSet("data/structures.soap")
    kernel_options = soap.Options()
    kernel_options.set("basekernel_type", "dot")
    kernel_options.set("base_exponent", 3.)
    kernel_options.set("base_filter", False)
    kernel_options.set("topkernel_type", "average")
    K = evaluate_kernel(dset, kernel_options, 'data/kernel.npy')

