#! /usr/bin/env python
import soap
import soap.tools

import os
import numpy as np
import logging
from momo import osio, endl, flush

options = soap.Options()
options.excludeCenters([])
options.excludeTargets([])
options.set('radialbasis.type', 'gaussian')
options.set('radialbasis.mode', 'adaptive')
options.set('radialbasis.N', 9) # 9
options.set('radialbasis.sigma', 0.5) # 0.9
options.set('radialbasis.integration_steps', 15)
options.set('radialcutoff.Rc', 6.8)
options.set('radialcutoff.Rc_width', 0.5)
options.set('radialcutoff.type', 'heaviside')
options.set('radialcutoff.center_weight', 1.)
options.set('angularbasis.type', 'spherical-harmonic')
options.set('angularbasis.L', 6) # 6
options.set('densitygrid.N', 20)
options.set('densitygrid.dx', 0.15)
options.set('spectrum.gradients', True)
options.set('kernel.adaptor', 'generic')
options.set('kernel.type', 'dot')
options.set('kernel.delta', 0.1)
options.set('kernel.xi', 1.)

class KernelAdaptorGeneric:
    def __init__(self, options):
        return
    def adapt(self, spectrum, pid_list=None):
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
            if pid_list:
                pid = atomic_i.getCenter().id
                if not pid in pid_list:
                    logging.debug("Skip, adapt atomic, pid = %d" % pid)
                    continue
            Xi = np.real(atomic_i.getPower("","").array)
            dimX = Xi.shape[0]*Xi.shape[1]
            Xi = Xi.reshape((dimX))
            #print "dim(X) =", dimX
            if not IX.any():
                IX = np.copy(Xi)
                IX.resize((1, dimX))
            else:
                i = IX.shape[0]
                IX.resize((i+1, dimX))
                IX[-1,:] = Xi
            #print IX
        return IX
    def adaptScalar(self, atomic):
        X = np.real(atomic.getPower("","").array)
        dimX = X.shape[0]*X.shape[1]
        X = np.real(X.reshape((dimX)))
        return X
    def adaptGradients(self, atomic, nb_pid):
        dxnkl_pid = atomic.getPowerGradGeneric(nb_pid)
        dX_dx = dxnkl_pid.getArrayGradX()
        dX_dy = dxnkl_pid.getArrayGradY()
        dX_dz = dxnkl_pid.getArrayGradZ()        
        dimX = dX_dx.shape[0]*dX_dx.shape[1]
        dX_dx = np.real(dX_dx.reshape((dimX)))
        dX_dy = np.real(dX_dy.reshape((dimX)))
        dX_dz = np.real(dX_dz.reshape((dimX)))
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
        self.pid_list_acquire = [1]
        self.pid_list_force = [1]
        return
    def acquire(self, structure, alpha):
        logging.info("Acquire ...")
        spectrum = soap.Spectrum(structure, self.options, self.basis)
        spectrum.compute()
        spectrum.computePower()
        spectrum.computePowerGradients()
        # New X's
        logging.info("Adapt spectrum ...")
        IX_acqu = self.adaptor.adapt(spectrum, self.pid_list_acquire)        
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
        IX_acqu = self.adaptor.adapt(spectrum, self.pid_list_acquire)        
        n_acqu = IX_acqu.shape[0]
        dim_acqu = IX_acqu.shape[1]
        logging.info("Compute energy from %d atomic environments ..." % n_acqu)
        for n in range(n_acqu):
            X = IX_acqu[n]
            ic = self.kernelfct.compute(self.IX, X)
            energy += self.alpha.dot(ic)
            print "Projection", ic            
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
            if not pid in self.pid_list_force:
                logging.debug("Skip forces derived from environement with pid = %d" % pid)
                continue
            #if pid != 1: continue
            nb_pids = atomic.getNeighbourPids()
            logging.info("  Center %d" % (pid))
            # neighbour-pid-independent kernel "prevector" (outer derivative)
            X = self.adaptor.adaptScalar(atomic)            
            dIC = self.kernelfct.computeDerivativeOuter(self.IX, X)
            alpha_dIC = self.alpha.dot(dIC)
            for nb_pid in nb_pids:
                # Force on neighbour
                logging.info("    -> Nb %d" % (nb_pid))
                dX_dx, dX_dy, dX_dz = self.adaptor.adaptGradients(atomic, nb_pid)
                
                force_x = -alpha_dIC.dot(dX_dx)
                force_y = -alpha_dIC.dot(dX_dy)
                force_z = -alpha_dIC.dot(dX_dz)
                
                forces[nb_pid-1][0] += force_x
                forces[nb_pid-1][1] += force_y
                forces[nb_pid-1][2] += force_z
                #print forces
                #raw_input('...')
        return forces
        

    

"""
alpha = np.array([-1,0.5])
IX = np.array([ [1,2,3,4,5], [5,6,7,8,9] ])

X = np.array([0.1,0.2,0.3,0.4,0.5])
dX = np.array([0.0,-0.1,0.1,-0.2,0.2])


def kernel_C(IX, X, xi, delta):
    return delta*delta * np.dot(IX,X)**xi
def kernel_dC(IX, X, xi, delta):
    c = kernel_C(IX, X, xi-1, delta)
    print c
    print np.diag(c)
    print xi*np.diag(c)
    print xi*np.diag(c).dot(IX)
    return xi*np.diag(c).dot(IX)

C = kernel_C(IX,X,2.,1.)
dC = kernel_dC(IX,X,2.,1.)

print dC.dot(dX)
print alpha.dot(dC.dot(dX))
"""

# STRUCTURE
xyzfile = 'config.xyz'
config = soap.tools.ase_load_single(xyzfile)
structure = soap.tools.setup_structure_ase(config.config_file, config.atoms)

for part in structure.particles:
    print part.id, part.type, part.pos


def restore_positions(structure, positions):
    idx = 0
    for part in structure:
        part.pos = positions[idx]
        idx += 1
    return [ part.pos for part in structure ]

def apply_force_step(structure, forces):
    max_step = 0.05
    max_f = 0.0
    for f in forces:
        df = np.dot(f,f)**0.5
        if df > max_f: max_f = df
    if max_f > max_step: scale = max_step/max_f
    else: scale = 1.
    idx = 0
    for part in structure:
        #if np.random.uniform(0.,1.) > 0.5: continue
        part.pos = part.pos + scale*forces[idx]
        idx += 1
    return [ part.pos for part in structure ]

"""
positions_orig = [ part.pos for part in structure ]
print positions_orig

apply_force(structure, None)
positions = [ part.pos for part in structure ]
print positions

restore_positions(structure, positions_orig)
positions = [ part.pos for part in structure ]
print positions
"""

logging.basicConfig(
    format='[%(asctime)s] %(message)s', 
    datefmt='%I:%M:%S', 
    level=logging.ERROR)
verbose = False

positions_0 = [ part.pos for part in structure ]
print "Positions at start"
for p in positions_0: print p

soap.silence()
kernelpot = KernelPotential(options)
kernelpot.acquire(structure, 1.)

"""
for p in structure:
    dpos = np.array([0.,0.,0.])
    dpos[0] = np.random.uniform(-1.,1.)
    dpos[1] = np.random.uniform(-1.,1.)
    dpos[2] = np.random.uniform(-1.,1.)
    p.pos = p.pos + 0.2*dpos
"""
"""
positions = [ np.array([1.,0.,0.]), np.array([1.5,0.1,0.]), np.array([-0.7,0.1,0.]) ]
restore_positions(structure, positions)
"""

positions = [ 
    np.array([0.,0.,0.]),
    np.array([2.4,0.,0.]),
    np.array([0.,3.,0.]) ]
restore_positions(structure, positions)
kernelpot.acquire(structure, -1.)
restore_positions(structure, positions_0)

acquire = False
restore = False

acquire_restore_every_nth = -1

ofs = open('log.xyz', 'w')

# Energy (int)
e_in = kernelpot.computeEnergy(structure)
osio << osio.my << "Energy(in)" << e_in << osio.endl

for n in range(100):
    osio << osio.mg << "ITERATION n =" << n << osio.endl
    
    # Energy  
    e_in = kernelpot.computeEnergy(structure)  
    if verbose: osio << osio.my << "Energy(n=%d)" % n << e_in << osio.endl
    
    # Compute forces and step ...
    forces = kernelpot.computeForces(structure)
    if verbose:
        print "Forces"
        for f in forces: print f
        
    positions = apply_force_step(structure, forces)
    if verbose:
        print "Positions after force step"
        for p in positions: print p

    # Write trajectory
    ofs.write('%d\n\n' % structure.n_particles)
    for p in structure.particles:
        r = p.pos
        ofs.write('%s %+1.7f %+1.7f %+1.7f\n' % (p.type, r[0], r[1], r[2]))

    if n > 0 and acquire_restore_every_nth > 0:
        if (n % acquire_restore_every_nth) == 0:
            osio << osio.mb << "Acquire and reset positions" << osio.endl
            kernelpot.acquire(structure, 1.)
            positions = restore_positions(structure, positions_0)
            print "Positions after restore"
            for p in positions: print p

    if acquire:
        # Acquire structure
        kernelpot.acquire(structure, 1.)

    if restore:
        # Restore original structure
        positions = restore_positions(structure, positions_0)
        print "Positions after restore"
        for p in positions: print p
    
    #forces = kernelpot.computeForces(structure)
    #print "Forces after acquirement"
    #for f in forces: print f

# Energy (out)
e_out = kernelpot.computeEnergy(structure)
osio << osio.my << "Energy(out)" << e_out << osio.endl

ofs.close()



"""
# SPECTRUM
spectrum = soap.Spectrum(structure, options)
spectrum.compute()
spectrum.computePower()
spectrum.computePowerGradients()

atomic = spectrum.getAtomic(0, "C")
print "ID, pos =", atomic.getCenter().id, atomic.getCenter().pos
nb_pids = atomic.getNeighbourPids()

xnkl_generic = atomic.getPower("","")
X = xnkl_generic.getArray()

for nb_pid in nb_pids:
    dxnkl_pid = atomic.getPowerGradGeneric(nb_pid)
    dX_dx = dxnkl_pid.getArrayGradX()
    dX_dy = dxnkl_pid.getArrayGradY()
    dX_dz = dxnkl_pid.getArrayGradZ()

    fi_x = -np.sum(X*dX_dx)
    fi_y = -np.sum(X*dX_dy)
    fi_z = -np.sum(X*dX_dz)

    print "%d %+1.7e %+1.7e %+1.7e" % (nb_pid, fi_x.real, fi_y.real, fi_z.real)
"""

#spectrum.writeDensity(1, "C", "")
#spectrum.writeDensityOnGrid(1, "C", "")

