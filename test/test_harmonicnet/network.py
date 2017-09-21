import numpy as np
import soap
import json

# Nodes
N = 100
tags = [ 'P%03d' for i in range(N) ]
kernel = 0.5*(1. + np.random.uniform(size=N*N).reshape((N,N)))
masses = 10*np.random.uniform(size=N) + 1.

# Network
network = soap.soapy.BondNetwork(tags, kernel, masses)
network.initialise(method='kernelpca')
network.integrate_md(
    n_steps=5000,
    rms_cut=1e-7,
    dt=0.01,
    dn_out=100,
    append_traj=False)

# Write final frame
network.write_confout('confout.txt')
    
