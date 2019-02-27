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
        # Exclude hydrogen centers:
        # NOTE that setting "exclude_centers"=["H"] is not sufficient
        # as discrete atomic types are lost in the embedding
        h_idcs = np.where(np.array(config.symbols) == "H")[0]
        options["exclude_center_ids"] = [ int(i+1) for i in h_idcs ]
        spectrum = soap.soapy.PowerSpectrum(
            config=config, 
            options=options,
            label="config-%d" % cidx)
        dmap = spectrum.exportDMapMatrix()
        log << "Config %d: %d centers" % (cidx, dmap.rows) << log.endl
        dset.append(dmap)
    return dset

def evaluate_kernel(dset, options):
    kernel = soap.Kernel(options)
    symmetric = True
    K = kernel.evaluate(dset, dset, symmetric, "float64")
    z = 1./np.sqrt(K.diagonal())
    K = K*np.outer(z,z)
    log << log.mg << "Kernel =" << log.endl
    log << K << log.endl
    return K

if __name__ == "__main__":
    configs = soap.tools.io.read('data/structures.xyz')
    typemap = soap.soapy.json_load_utf8(open('typemap.json'))

    # Compute SOAP descriptors
    options = soap.soapy.configure_default(typemap=typemap)
    dset = evaluate_soap(configs, options)

    # Compute molecular kernel
    kernel_options = soap.Options()
    kernel_options.set("basekernel_type", "dot")
    kernel_options.set("base_exponent", 3.)
    kernel_options.set("base_filter", False)
    kernel_options.set("topkernel_type", "average")
    K = evaluate_kernel(dset, kernel_options)

