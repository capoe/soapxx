#! /usr/bin/env python
import numpy as np
import soap
import os
import json
import glob
import h5py
import copy
log = soap.log

def assert_equal(z, target, eps=1e-5):
    if np.abs(z-target) > eps: raise ValueError(z)
    else: log << "+" << log.flush

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

def test_attribution_average(dset, options):
    log << log.mg << "<test_attribution_average>" << log.flush
    kernel_options = soap.Options()
    kernel_options.set("basekernel_type", "dot")
    kernel_options.set("base_exponent", 3.)
    kernel_options.set("base_filter", False)
    kernel_options.set("topkernel_type", "average")
    kernel = soap.Kernel(kernel_options)
    K = kernel.evaluate(dset, dset, False, "float64")
    for i in range(len(dset)):
        KK = kernel.attributeLeft(dset[i], dset, "float64")
        KK_sum = np.sum(KK, axis=0)
        for j in range(len(dset)):
            assert_equal(KK_sum[j]-K[i,j], 0.0, 1e-6)
    log << log.endl
    
def test_attribution_rematch(dset, options):
    log << log.mg << "<test_attribution_rematch>" << log.flush
    kernel_options = soap.Options()
    kernel_options.set("basekernel_type", "dot")
    kernel_options.set("base_exponent", 3.)
    kernel_options.set("base_filter", False)
    kernel_options.set("topkernel_type", "rematch")
    kernel_options.set("rematch_gamma", 0.01)
    kernel_options.set("rematch_eps", 1e-6)
    kernel_options.set("rematch_omega", 1.0)
    kernel = soap.Kernel(kernel_options)
    K = kernel.evaluate(dset, dset, False, "float64")
    for i in range(len(dset)):
        KK = kernel.attributeLeft(dset[i], dset, "float64")
        KK_sum = np.sum(KK, axis=0)
        for j in range(len(dset)):
            assert_equal(KK_sum[j]-K[i,j], 0.0, 1e-6)
    log << log.endl

if __name__ == "__main__":
    configs = soap.tools.io.read('structures.xyz')
    options = soap.soapy.configure_default()
    soap.silent(True)
    dset = evaluate_soap(configs, options)
    test_attribution_average(dset, options)
    test_attribution_rematch(dset, options)
    log << "All passed." << log.endl









