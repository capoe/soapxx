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
log.AddArg("mode", typ=str, default="pca")
options = log.Parse()

state = rmt.State().unpickle(options.statefile)

log << log.mg << options.statefile << options.mode << log.endl
log << "%d samples, %d descriptor components" % (state["IX"].shape[0], state["IX"].shape[1]) << log.endl

if options.mode == "pca":
    IX_norm_pca, IX_norm, X_mean, X_std, S, L, U = rmt.pca_compute(state["IX"], ddof=0)
    state["IX"] = IX_norm_pca
    sign_indicator = np.min(np.abs(state["IX"]), axis=0)
    fprops_z = []
    print " ".join([ f[0] for f in state["features_with_props"] ]) 
    for i in range(state["IX"].shape[1]):
        coeffs = U[:,i]
        print "L%02d lambda=%1.1e  " % (i,L[i,i]), " ".join(map(lambda c: "%+1.1e" % c, coeffs))
    for fidx, f in enumerate(state["features_with_props"]):
        f[0] = "L%02d" % fidx
        f[1] = "+-"
        f[2] = "-0" if sign_indicator[fidx] > 1e-5 else "+0"
        f[4] = ""
        fprops_z.append(f)
    state["features_with_props"] = fprops_z
elif options.mode == "mp":
    L_signal, V_signal, X_mean, X_std = rmt.labs.mp_transform(state["IX"], norm_avg=True, norm_std=True, log=log)
    log << "%d signal components" % V_signal.shape[1] << log.endl
    IX_tf = V_signal.T.dot(rmt.utils.div0(state["IX"]-X_mean, X_std).T).T
    state["IX"] = IX_tf
    print " ".join([ f[0] for f in state["features_with_props"] ]) 
    for i in range(state["IX"].shape[1]):
        coeffs = V_signal[:,i]
        print "lambda=%1.1e  " % L_signal[i], " ".join(map(lambda c: "%+1.1e" % c, coeffs))
    fprops_z = []
    for i in range(state["IX"].shape[1]):
        fprops_z.append(["L%02d" % i, "+-", "-0", 1.0, ""])
    state["features_with_props"] = fprops_z
else:
    raise ValueError()

state["tag"] = state["tag"] + "_" + options.mode
statefile_out = "%s/state_%s.jar" % (os.path.dirname(options.statefile), state["tag"])
log << "Saving to '%s'" % statefile_out << log.endl
state.pickle(statefile_out)

