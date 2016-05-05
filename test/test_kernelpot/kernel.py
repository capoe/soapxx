#! /usr/bin/env python
import soap
import soap.tools

import os
import numpy as np
import logging
from momo import osio, endl, flush

class KernelAdaptorGeneric:
    def __init__(self, options):
        return
    def adapt(self, spectrum):
        # IX = [
        #    X1 ->,
        #    X2 ->,
        #    ...
        # ]
        IX = np.zeros((0,0), dtype='complex128')
        dimX = -1
        # TODO More adaptors:
        # TODO Particle ID mask, center-type mask
        # TODO For type-resolved adaptors, increase dimensionality accordingly
        for atomic_i in spectrum:
            #print atomic_i.getCenter().pos
            #if pid_list:
            #    pid = atomic_i.getCenter().id
            #    if not pid in pid_list:
            #        logging.debug("Skip, adapt atomic, pid = %d" % pid)
            #        continue
            
            Xi_unnorm, Xi_norm = self.adaptScalar(atomic_i)
            dimX = Xi_norm.shape[0]
            #Xi = np.real(atomic_i.getPower("","").array)
            #dimX = Xi.shape[0]*Xi.shape[1]
            #Xi = Xi.reshape((dimX))
            
            #print "dim(X) =", dimX
            if not IX.any():
                IX = np.copy(Xi_norm) # TODO Is this necessary?
                IX.resize((1, dimX))
            else:
                i = IX.shape[0]
                IX.resize((i+1, dimX))
                IX[-1,:] = Xi_norm
            #print IX
        return IX
    def adaptScalar(self, atomic):
        X = np.real(atomic.getPower("","").array)
        dimX = X.shape[0]*X.shape[1]
        X = np.real(X.reshape((dimX)))
        # Normalize
        X_norm = X/np.dot(X,X)**0.5
        return X, X_norm
    def adaptGradients(self, atomic, nb_pid, X):
        # NOTE X is not normalized (=> X = X')
        dxnkl_pid = atomic.getPowerGradGeneric(nb_pid)
        dX_dx = dxnkl_pid.getArrayGradX()
        dX_dy = dxnkl_pid.getArrayGradY()
        dX_dz = dxnkl_pid.getArrayGradZ()        
        dimX = dX_dx.shape[0]*dX_dx.shape[1]
        dX_dx = np.real(dX_dx.reshape((dimX)))
        dX_dy = np.real(dX_dy.reshape((dimX)))
        dX_dz = np.real(dX_dz.reshape((dimX)))
        # Normalize
        mag_X = np.dot(X,X)**0.5
        dX_dx = dX_dx/mag_X - np.dot(X, dX_dx)/mag_X**3 * X
        dX_dy = dX_dy/mag_X - np.dot(X, dX_dy)/mag_X**3 * X
        dX_dz = dX_dz/mag_X - np.dot(X, dX_dz)/mag_X**3 * X
        return dX_dx, dX_dy, dX_dz
        
class KernelFunctionDot(object):
    def __init__(self, options):
        self.delta = float(options.get('kernel.delta'))
        self.xi = float(options.get('kernel.xi'))
        return
    def computeDot(self, IX, X, xi, delta):
        return delta**2 * np.dot(IX,X)**xi
    def compute(self, IX, X):
        return self.computeDot(IX, X, self.xi, self.delta)    
    def computeDerivativeOuter(self, IX, X):
        c = self.computeDot(IX, X, self.xi-1, self.delta)
        return self.xi*np.diag(c).dot(IX)    
   
KernelAdaptorFactory = { 'generic': KernelAdaptorGeneric }     
KernelFunctionFactory = { 'dot':KernelFunctionDot }

class KernelPotential:
    def __init__(self, options):
        logging.info("Construct kernel potential ...")
        self.basis = soap.Basis(options)
        self.options = options        
        # CORE DATA
        self.IX = None
        self.alpha = None
        self.dimX = None # <- Also used as flag set by first ::acquire call
        # KERNEL
        logging.info("Choose kernel function ...")
        self.kernelfct = KernelFunctionFactory[options.get('kernel.type')](options)   
        # ADAPTOR
        logging.info("Choose adaptor ...")
        self.adaptor = KernelAdaptorFactory[options.get('kernel.adaptor')](options)
        # INCLUSIONS / EXCLUSIONS
        # -> Already be enforced when computing spectra
        return
    def acquire(self, structure, alpha):
        logging.info("Acquire ...")
        spectrum = soap.Spectrum(structure, self.options, self.basis)
        spectrum.compute()
        spectrum.computePower()
        spectrum.computePowerGradients()
        # New X's
        logging.info("Adapt spectrum ...")
        IX_acqu = self.adaptor.adapt(spectrum)
        n_acqu = IX_acqu.shape[0]
        dim_acqu = IX_acqu.shape[1]
        # New alpha's
        alpha_acqu = np.zeros((n_acqu))
        alpha_acqu.fill(alpha)
        if not self.dimX:
            # First time ...
            self.dimX = dim_acqu
            self.IX = IX_acqu
            self.alpha = alpha_acqu
        else:
            # Check and extend ...
            assert self.dimX == dim_acqu # Acquired descr. should match linear dim. of previous descr.'s
            I = self.IX.shape[0]
            self.IX.resize((I+n_acqu, self.dimX))
            self.IX[I:I+n_acqu,:] = IX_acqu
            self.alpha.resize((I+n_acqu))
            self.alpha[I:I+n_acqu] = alpha_acqu
        #print self.alpha
        #print self.IX
        logging.info("Acquired %d environments." % n_acqu)
        return
    def computeEnergy(self, structure):
        logging.info("Start energy ...")
        energy = 0.0
        spectrum = soap.Spectrum(structure, self.options, self.basis)
        spectrum.compute()
        spectrum.computePower()
        spectrum.computePowerGradients()
        IX_acqu = self.adaptor.adapt(spectrum)        
        n_acqu = IX_acqu.shape[0]
        dim_acqu = IX_acqu.shape[1]
        logging.info("Compute energy from %d atomic environments ..." % n_acqu)
        for n in range(n_acqu):
            X = IX_acqu[n]
            #print "X_MAG", np.dot(X,X)
            ic = self.kernelfct.compute(self.IX, X)
            energy += self.alpha.dot(ic)
            #print "Projection", ic            
        return energy            
    def computeForces(self, structure, verbose=False):
        logging.info("Start forces ...")
        if verbose:
            for p in structure: print p.pos
        forces = [ np.zeros((3)) for i in range(structure.n_particles) ]
        logging.info("Compute forces on %d particles ..." % structure.n_particles)
        # Compute X's, dX's
        spectrum = soap.Spectrum(structure, self.options, self.basis)
        spectrum.compute()
        spectrum.computePower()
        spectrum.computePowerGradients()
        # Extract & compute force
        for atomic in spectrum:
            pid = atomic.getCenter().id
            #if not pid in self.pid_list_force:
            #    logging.debug("Skip forces derived from environment with pid = %d" % pid)
            #    continue
            #if pid != 1: continue
            nb_pids = atomic.getNeighbourPids()
            logging.info("  Center %d" % (pid))
            # neighbour-pid-independent kernel "prevector" (outer derivative)
            X, X_norm = self.adaptor.adaptScalar(atomic)
            dIC = self.kernelfct.computeDerivativeOuter(self.IX, X)
            alpha_dIC = self.alpha.dot(dIC)
            for nb_pid in nb_pids:
                # Force on neighbour
                logging.info("    -> Nb %d" % (nb_pid))
                dX_dx, dX_dy, dX_dz = self.adaptor.adaptGradients(atomic, nb_pid, X)
                
                force_x = -alpha_dIC.dot(dX_dx)
                force_y = -alpha_dIC.dot(dX_dy)
                force_z = -alpha_dIC.dot(dX_dz)
                
                forces[nb_pid-1][0] += force_x
                forces[nb_pid-1][1] += force_y
                forces[nb_pid-1][2] += force_z
                #print forces
                #raw_input('...')
        return forces
        
def restore_positions(structure, positions):
    idx = 0
    for part in structure:
        part.pos = positions[idx]
        idx += 1
    return [ part.pos for part in structure ]

def perturb_positions(structure, exclude_pid=[]):
    for part in structure:
        if part.id in exclude_pid: continue
        dx = np.random.uniform(-1.,1.)
        dy = np.random.uniform(-1.,1.)
        dz = np.random.uniform(-1.,1.)
        part.pos = part.pos + 0.1*np.array([dx,dy,dz])
    return [ part.pos for part in structure ]
    
def random_positions(structure, exclude_pid=[]):
    for part in structure:
        if part.id in exclude_pid: continue
        dx = np.random.uniform(-1.,1.)
        dy = np.random.uniform(-1.,1.)
        dz = np.random.uniform(-1.,1.)
        part.pos = np.array([dx,dy,dz])
    return [ part.pos for part in structure ]

def apply_force_step(structure, forces, scale, constrain_particles=[]):
    max_step = 0.05
    max_f = 0.0
    for f in forces:
        df = scale*np.dot(f,f)**0.5
        if df > max_f: max_f = df
    if max_f > max_step: scale = scale*max_step/max_f
    else: pass
    #print "Scale =", scale
    idx = -1
    for part in structure:
        idx += 1
        if part.id in constrain_particles: 
            print "Skip force step, pid =", part.id
            continue
        #if np.random.uniform(0.,1.) > 0.5: continue
        part.pos = part.pos + scale*forces[idx]
    return [ part.pos for part in structure ]

def apply_force_norm_step(structure, forces, scale, constrain_particles=[]):
    min_step = 0.5
    max_f = 0.0
    for f in forces:
        df = np.dot(f,f)**0.5
        if df > max_f: max_f = df
    scale = min_step/max_f
    idx = -1
    for part in structure:
        idx += 1
        if part.id in constrain_particles: 
            print "Skip force step, pid =", part.id
            continue
        #if np.random.uniform(0.,1.) > 0.5: continue
        part.pos = part.pos + scale*forces[idx]
    return [ part.pos for part in structure ]

