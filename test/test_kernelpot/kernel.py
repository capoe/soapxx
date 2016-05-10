#! /usr/bin/env python
import soap
import soap.tools

import os
import numpy as np
import logging
from momo import osio, endl, flush

# THERE IS SOMETHING WRONG WITH THE GRADIENTS HERE (DIRECTION SEEMS FINE, BUT MAGNITUDE IS NOT?)! REMARK ~ok, error source found
# TODO Unit test for global spectrum                                    REMARK ~ok, automatize
# TODO Check outer kernel derivative                                    REMARK ~ok, automatize
# TODO Check normalization derivative (compute on C++ level already)    REMARK ~ok, -> C++

# TODO PCA + identification of eigenstructures
# TODO Sample Bethe tree

class TrajectoryLogger(object):
    def __init__(self, outfile):
        self.ofs = open(outfile, 'w')
    def logFrame(self, structure):
        # Write first frame
        self.ofs.write('%d\n\n' % structure.n_particles)
        for p in structure.particles:
            r = p.pos
            self.ofs.write('%s %+1.7f %+1.7f %+1.7f\n' % (p.type, r[0], r[1], r[2]))
        return
    def close(self):
        self.ofs.close()
        return

class KernelAdaptorGeneric(object):
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
        
class KernelAdaptorGlobalGeneric(object):
    def __init__(self, options):
        return
    def adapt(self, spectrum):
        # EXTRACT A SET OF CENTER-BASED POWER EXPANSIONS
        # Here: only global
        IX = np.zeros((0,0), dtype='complex128')
        dimX = -1
        atomic_global = spectrum.getGlobal()
        Xi_unnorm, Xi_norm = self.adaptScalar(atomic_global)
        dimX = Xi_norm.shape[0]
        IX = np.copy(Xi_norm)
        IX.resize((1,dimX))
        return IX
    def adaptScalar(self, atomic):
        # EXTRACT POWER EXPANSION FROM ATOMIC SPECTRUM
        # Here: type "":"" (= generic)
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
   
KernelAdaptorFactory = { 'generic': KernelAdaptorGeneric, 'global-generic': KernelAdaptorGlobalGeneric }     
KernelFunctionFactory = { 'dot':KernelFunctionDot }

class KernelPotential(object):
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
        self.use_global_spectrum = True if options.get('kernel.adaptor') == 'global-generic' else False
        # INCLUSIONS / EXCLUSIONS
        # -> Already be enforced when computing spectra
        return
    def importAcquire(self, IX_acqu, alpha):
        n_acqu = IX_acqu.shape[0]
        dim_acqu = IX_acqu.shape[1]
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
        logging.info("Imported %d environments." % n_acqu)
        return
    def acquire(self, structure, alpha):
        logging.info("Acquire ...")
        spectrum = soap.Spectrum(structure, self.options, self.basis)
        spectrum.compute()
        spectrum.computePower()
        spectrum.computePowerGradients()
        if self.use_global_spectrum:
            spectrum.computeGlobal()
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
    def computeEnergy(self, structure, return_prj_mat=False):
        logging.info("Start energy ...")
        energy = 0.0
        spectrum = soap.Spectrum(structure, self.options, self.basis)
        spectrum.compute()
        spectrum.computePower()
        spectrum.computePowerGradients()
        # TODO The kernel policy should take care of this:
        if self.use_global_spectrum:
            spectrum.computeGlobal()
        IX_acqu = self.adaptor.adapt(spectrum)        
        n_acqu = IX_acqu.shape[0]
        dim_acqu = IX_acqu.shape[1]
        logging.info("Compute energy from %d atomic environments ..." % n_acqu)
        projection_matrix = []
        for n in range(n_acqu):
            X = IX_acqu[n]
            #print "X_MAG", np.dot(X,X)
            ic = self.kernelfct.compute(self.IX, X)
            energy += self.alpha.dot(ic)
            #print "Projection", ic
            projection_matrix.append(ic)
        if return_prj_mat:
            return energy, projection_matrix
        else:
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
        if self.use_global_spectrum:
            atomic_global = spectrum.computeGlobal()
            spectrum_iter = [ atomic_global ]
        else:
            spectrum_iter = spectrum
        # Extract & compute force
        for atomic in spectrum_iter:
            pid = atomic.getCenter().id if not self.use_global_spectrum else -1
            #if not pid in self.pid_list_force:
            #    logging.debug("Skip forces derived from environment with pid = %d" % pid)
            #    continue
            #if pid != 1: continue
            nb_pids = atomic.getNeighbourPids()
            logging.info("  Center %d" % (pid))
            # neighbour-pid-independent kernel "prevector" (outer derivative)
            X_unnorm, X_norm = self.adaptor.adaptScalar(atomic)
            dIC = self.kernelfct.computeDerivativeOuter(self.IX, X_norm) # TODO This must be X_norm!
            alpha_dIC = self.alpha.dot(dIC)
            for nb_pid in nb_pids:
                # Force on neighbour
                logging.info("    -> Nb %d" % (nb_pid))
                dX_dx, dX_dy, dX_dz = self.adaptor.adaptGradients(atomic, nb_pid, X_unnorm)
                
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
    
def random_positions(structure, exclude_pid=[], b0=-1., b1=1., x=True, y=True, z=True):
    for part in structure:
        if part.id in exclude_pid: continue
        dx = np.random.uniform(b0,b1) if x else 0.
        dy = np.random.uniform(b0,b1) if y else 0.
        dz = np.random.uniform(b0,b1) if z else 0.
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

def evaluate_energy(positions, structure, kernelpot, opt_pids, verbose=False, ofs=None):
    if verbose: print "Energy"
    # Impose positions
    pid_pos = positions.reshape((opt_pids.shape[0],3))
    for pidx, pid in enumerate(opt_pids):
        pos = pid_pos[pidx,:]
        particle = structure.getParticle(pid)        
        particle.pos = pos        
    for part in structure:
        if verbose: print part.id, part.type, part.pos
    # Evaluate energy function
    energy = kernelpot.computeEnergy(structure)
    # Log
    if ofs: ofs.logFrame(structure)
    if verbose: print energy    
    return energy

def evaluate_energy_gradient(positions, structure, kernelpot, opt_pids, verbose=False, ofs=None):
    if verbose: print "Forces"
    # Adjust positions
    pid_pos = positions.reshape((opt_pids.shape[0],3))
    for pidx, pid in enumerate(opt_pids):
        pos = pid_pos[pidx,:]
        particle = structure.getParticle(pid)        
        particle.pos = pos        
    for part in structure:
        if verbose: print part.id, part.type, part.pos
    # Evaluate forces
    forces = kernelpot.computeForces(structure)
    gradients = -1.*np.array(forces)
    opt_pidcs = opt_pids-1
    gradients_short = gradients[opt_pidcs]
    gradients_short = gradients_short.flatten()
    if verbose: print gradients_short
    #forces[2] = 0. # CONSTRAIN TO Z=0 PLANE
    return gradients_short

