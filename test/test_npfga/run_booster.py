#! /usr/bin/env python
import numpy as np
import json
import sys
import os
import pickle
import copy
import librmt as rmt
import soap
log = rmt.log

log.Connect()
log.AddArg("statefile", typ=str, help="Path to binary file storing state object with fields: 'IX', 'T', 'tag', 'features_with_props'")
log.AddArg("output_folder", typ=str, default='out_booster', help='Output folder')
log.AddArg("n_iters", typ=int, default=1, help='Number of NPFGA iterations')
log.AddArg("bootstrap_type", typ=str, default="residuals", help="'samples' or 'residuals'")
# CV OPTIONS
log.AddArg("cv_mode", typ=str, default='none', help="CV method: 'loo', 'mc', 'none', 'user'")
log.AddArg("f_mccv", typ=float, default=0.9, help='CV-MC subsampling fraction')
log.AddArg("n_mccv", typ=int, default=10, help='CV-MC repetitions')
log.AddArg("test_on", typ=(list,int), default=[], help="List of sample idcs for test set in mode 'user'")
log.AddArg("seed", typ=int, default=830292, help='RNG seed')
options = log.Parse()

# PREPARATORY STEPS
state_base = rmt.State().unpickle(options.statefile)
cv_iterator = soap.soapy.npfga.cv_iterator[options.cv_mode](state_base, options)
np.random.seed(options.seed)
if not os.path.exists(options.output_folder):
    log >> 'mkdir -p %s' % options.output_folder

booster = soap.soapy.npfga.Booster(options)
while not cv_iterator.isDone():
    # LOAD STATE
    log << log.mg << "Clone state" << log.endl
    state = state_base.clone()

    # TRAIN-TEST SPLITTING
    cv_info_str, idcs_train, idcs_test = cv_iterator.next()
    log << "%s: Train on %d samples, test on %d" % (cv_info_str, len(idcs_train), len(idcs_test)) << log.endl
    log << "- Training idcs" << idcs_train[0:5] << "..." << log.endl
    log << "- Test idcs" << idcs_test[0:5] << "..." << log.endl

    # Descriptor matrix and target
    IX_train = state["IX"][idcs_train]
    IX_test = state["IX"][idcs_test]
    Y_train = state["T"][idcs_train]
    Y_test = state["T"][idcs_test]

    for iteration in range(options.n_iters):
        log << log.mb << "Iteration %d" % iteration << log.endl
        booster.dispatchY(iteration, Y_train, Y_test)
        Y_train, Y_test = booster.getResidues()
        booster.dispatchX(iteration, IX_train, IX_test)
        booster.train(bootstraps=2000, method=options.bootstrap_type)
        rmse_train, rho_train, rmse_test, rho_test = booster.evaluate()
        log << "Train: RMSE=%1.4e RHO=%1.4e" % (rmse_train, rho_train) << log.endl
        log << "Test:  RMSE=%1.4e RHO=%1.4e" % (rmse_test, rho_test) << log.endl

booster.write("%s/%s_%s_pred_i%%d" % (
    options.output_folder, os.path.basename(options.statefile), options.cv_mode))

