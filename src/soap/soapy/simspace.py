#! /usr/bin/env python
import soap
import soap.tools

import os
import sys
import numpy as np
import logging
import io

import scipy.optimize

from kernel import KernelAdaptorFactory, KernelFunctionFactory, TrajectoryLogger

class SimSpaceNode(object):
    def __init__(self, node_id, structure, basis, options, compute=True):
        self.id = node_id
        # Copy structure
        self.structure = soap.Structure("?")
        self.structure.model(structure)
        # Basis, options
        self.basis = basis
        self.options = options
        # Stored spectra
        self.reset()
        # Spectrum adaptor
        self.adaptor = KernelAdaptorFactory[options.get('kernel.adaptor')](options)
        if compute: self.acquire()
        # Potentials
        self.potentials = []
        self.potentials_self = []
        return
    def size(self):
        return self.IX.shape[0] # <- Number of environments stored
    def reset(self):
        self.IX = None
        self.dimX = None
        self.pid_X_unnorm = {}
        self.pid_X_norm = {}
        self.pid_nbpid_dX = {}
    def assignPositions(self, x0, compute=True):
        for idx, part in enumerate(self.structure):
            part.pos = x0[idx]
        if compute: self.acquire()
        return
    def randomizePositions(self, scale=0.1, zero_pids=[0], compute=True):
        x0 = np.random.uniform(-1.*scale, +1.*scale, (self.structure.n_particles, 3))
        for pid in zero_pids:
            x0[pid-1,:] = 0.
        self.assignPositions(x0, compute)
        return x0
    def perturbPositions(self, scale=0.1, zero_pids=[0], compute=True):
        x0 = np.random.uniform(-1.*scale, +1.*scale, (self.structure.n_particles, 3))
        for pid in zero_pids:
            x0[pid-1,:] = 0.
        for idx, part in enumerate(self.structure):
            pos = part.pos
            part.pos = part.pos + x0[idx]
            x0[idx] = part.pos
        if compute: self.acquire()
        return x0
    def acquire(self, reset=True):
        if reset: self.reset()
        # Compute spectrum
        self.spectrum = soap.Spectrum(self.structure, self.options, self.basis)
        self.spectrum.compute()
        self.spectrum.computePower()
        self.spectrum.computePowerGradients()
        self.spectrum.computeGlobal()
        # PID-resolved storage for X, dX
        for atomic in self.adaptor.getListAtomic(self.spectrum):
            pid = atomic.getCenterId()
            X_unnorm, X_norm = self.adaptor.adaptScalar(atomic) # TODO Done again below in ::adapt => simplify
            self.pid_X_unnorm[pid] = X_unnorm
            self.pid_X_norm[pid] = X_norm
            self.pid_nbpid_dX[pid] = {}
            # NB-PID-resolved derivatives
            nb_pids = atomic.getNeighbourPids()
            for nb_pid in nb_pids:
                dX_dx, dX_dy, dX_dz = self.adaptor.adaptGradients(atomic, nb_pid, X_unnorm)
                self.pid_nbpid_dX[pid][nb_pid] = (dX_dx, dX_dy, dX_dz)
        # New X's
        IX_acqu = self.adaptor.adapt(self.spectrum)
        n_acqu = IX_acqu.shape[0]
        dim_acqu = IX_acqu.shape[1]
        if not self.dimX:
            # First time ...
            self.dimX = dim_acqu
            self.IX = IX_acqu
        else:
            # Check and extend ...
            assert self.dimX == dim_acqu # Acquired descr. should match linear dim. of previous descr.'s
            I = self.IX.shape[0]
            self.IX.resize((I+n_acqu, self.dimX))
            self.IX[I:I+n_acqu,:] = IX_acqu
        return
    def createPotential(self, nb_node, options): # TODO Simplify. Only one potential needed per pair.
        self.potentials.append(SimSpacePotential(self, nb_node, options))
        return
    def createThreePotential(self, B, C, options):
        self.potentials.append(SimSpaceThreePotential(self, B, C, options))
        return
    def clearPotentials(self):
        del self.potentials
        del self.potentials_self
        self.potentials = []
        self.potentials_self = []
    def computePotentialEnergy(self):
        self.energy = 0.
        for pot in self.potentials:
            self.energy += pot.computeEnergy()
        for pot in self.potentials_self:
            self.energy += pot.computeEnergy()
        return self.energy
    def getListAtomic(self):
        return self.adaptor.getListAtomic(self.spectrum)
    def getPidX(self, pid):
        return self.pid_X_unnorm[pid], self.pid_X_norm[pid]
    def getPidGradX(self, pid, nb_pid):
        return self.pid_nbpid_dX[pid][nb_pid]
        
class SimSpaceTopology(object):
    def __init__(self, options):
        self.basis = soap.Basis(options)
        self.options = options
        self.nodes = []
        self.kernelfct = KernelFunctionFactory[options.get('kernel.type')](options)
        self.IX = np.zeros((0))
        return
    def compileIX(self):
        n_nodes = len(self.nodes)
        dim = self.nodes[0].IX.shape[1]
        self.IX = np.zeros((n_nodes,dim))
        for idx, node in enumerate(self.nodes):
            self.IX[idx] = node.IX
        return
    def computeKernelMatrix(self, return_distance=False):
        if not self.IX.any(): self.compileIX()
        K = self.kernelfct.computeBlock(self.IX, return_distance)
        return K
    def computePotentialEnergy(self):
        self.energy = 0.
        for node in self.nodes:
            self.energy += node.computePotentialEnergy()
        return self.energy
    def createNode(self, structure):
        node_id = len(self.nodes)+1
        node = SimSpaceNode(node_id, structure, self.basis, self.options)
        self.nodes.append(node)
        return node
    def summarize(self):
        for idx, node in enumerate(self.nodes):
            print "Node %d '%s'" % ((idx+1), node.structure.label)
            for part in node.structure:
                print part.id, part.type, part.pos
        return
    def writeData(self, prefix='out.top'):
        # Structures
        ofs = TrajectoryLogger('%s.xyz' % prefix)
        for node in self.nodes:
            ofs.logFrame(node.structure)
        ofs.close()
        # Descriptors
        self.compileIX()
        np.savetxt('%s.ix.txt' % prefix, self.IX)
        # Kernel
        K = self.computeKernelMatrix()
        np.savetxt('%s.kernelmatrix.txt' % prefix, K)
        # Energy
        E = [ node.computePotentialEnergy() for node in self.nodes ]
        np.savetxt('%s.energy.txt' % prefix, E)
        return

class SimSpaceThreePotential(object):
    def __init__(self, A, B, C, options):
        self.A = A # <- target (see force calculation)
        self.target = A
        self.B = B
        self.C = C
        self.alpha = np.array([float(options.get('kernel.alpha'))])
        # INTERACTION FUNCTION
        self.kernelfct = KernelFunctionFactory[options.get('kernel.type')](options)
        # Spectrum adaptor
        self.adaptor = KernelAdaptorFactory[options.get('kernel.adaptor')](options)   
        return
    def computeEnergy(self, return_prj_mat=False):
        energy = 0.0
        projection_matrix = []
        for n in range(self.A.size()):
            X = self.A.IX[n]
            ic = self.kernelfct.compute(self.B.IX, self.C.IX, X)
            energy += self.alpha.dot(ic)
            projection_matrix.append(ic)
        if return_prj_mat:
            return energy, projection_matrix
        else:
            return energy
    def computeForces(self, verbose=False):
        forces = [ np.zeros((3)) for i in range(self.A.structure.n_particles) ]        
        for atomic in self.A.getListAtomic():
            pid = atomic.getCenterId()
            nb_pids = atomic.getNeighbourPids()
            if verbose: print "  Center %d" % (pid)
            # neighbour-pid-independent kernel "prevector" (outer derivative)
            X_unnorm, X_norm = self.A.getPidX(pid)
            dIC = self.kernelfct.computeDerivativeOuter(self.B.IX, self.C.IX, X_norm)
            alpha_dIC = self.alpha.dot(dIC)
            for nb_pid in nb_pids:
                # Force on neighbour
                if verbose: print "    -> Nb %d" % (nb_pid)
                dX_dx, dX_dy, dX_dz = self.A.getPidGradX(pid, nb_pid)
                force_x = -alpha_dIC.dot(dX_dx)
                force_y = -alpha_dIC.dot(dX_dy)
                force_z = -alpha_dIC.dot(dX_dz)                
                forces[nb_pid-1][0] += force_x
                forces[nb_pid-1][1] += force_y
                forces[nb_pid-1][2] += force_z
        return np.array(forces)
    def computeGradients(self, verbose=False):
        return -1 * self.computeForces(verbose)

class SimSpacePotential(object):
    def __init__(self, target, source, options):
        self.target = target
        self.source = source
        self.alpha = np.array([float(options.get('kernel.alpha'))])
        # INTERACTION FUNCTION
        self.kernelfct = KernelFunctionFactory[options.get('kernel.type')](options)
        # Spectrum adaptor
        self.adaptor = KernelAdaptorFactory[options.get('kernel.adaptor')](options)   
        return
    def computeEnergy(self, return_prj_mat=False):
        energy = 0.0
        projection_matrix = []
        for n in range(self.target.size()):
            X = self.target.IX[n]
            ic = self.kernelfct.compute(self.source.IX, X)
            energy += self.alpha.dot(ic)
            projection_matrix.append(ic)
        if return_prj_mat:
            return energy, projection_matrix
        else:
            return energy
    def computeForces(self, verbose=False):
        forces = [ np.zeros((3)) for i in range(self.target.structure.n_particles) ]        
        for atomic in self.target.getListAtomic():
            pid = atomic.getCenterId()
            nb_pids = atomic.getNeighbourPids()
            if verbose: print "  Center %d" % (pid)
            # neighbour-pid-independent kernel "prevector" (outer derivative)
            X_unnorm, X_norm = self.target.getPidX(pid)
            dIC = self.kernelfct.computeDerivativeOuter(self.source.IX, X_norm)
            alpha_dIC = self.alpha.dot(dIC)
            for nb_pid in nb_pids:
                # Force on neighbour
                if verbose: print "    -> Nb %d" % (nb_pid)
                dX_dx, dX_dy, dX_dz = self.target.getPidGradX(pid, nb_pid)
                force_x = -alpha_dIC.dot(dX_dx)
                force_y = -alpha_dIC.dot(dX_dy)
                force_z = -alpha_dIC.dot(dX_dz)                
                forces[nb_pid-1][0] += force_x
                forces[nb_pid-1][1] += force_y
                forces[nb_pid-1][2] += force_z
        return np.array(forces)
    def computeGradients(self, verbose=False):
        return -1 * self.computeForces(verbose)

class SimSpaceJoint(object):
    def __init__(self, top, par_joint=None, structure=None):
        self.top = top
        self.par_joint = par_joint # TODO Extend to several parents => structural cross-over
        if self.par_joint:
            assert structure == None
            self.node = top.createNode(par_joint.node.structure)
        else:
            self.node = top.createNode(structure)
        self.child_joints = []
    def hasChildren(self):
        return len(self.child_joints) > 0
    def spawn(self, n, authorize_spawn_fct):
        new_joints = []
        if self.hasChildren():
            for child_joint in self.child_joints:
                new_joints = new_joints + child_joint.spawn(n, authorize_spawn_fct)
        else:
            if authorize_spawn_fct(self):
                for i in range(n):
                    child_joint = SimSpaceJoint(self.top, self)
                    self.child_joints.append(child_joint)
                    new_joints.append(child_joint)
        return new_joints
    def summarize(self, indent=''):
        print '%sID=%d: %d children' % (indent, self.node.id, len(self.child_joints))
        for child in self.child_joints:
            child.summarize(indent=indent+'  ')
        return
    def writeParentChildPairs(self, ofs):
        for child in self.child_joints:
            ofs.write('%d %d\n' % (self.node.id, child.node.id))
            child.writeParentChildPairs(ofs)
        return

class SimSpaceTree(object):
    def __init__(self, options):
        self.top = SimSpaceTopology(options)
        self.joints = []
        self.generations = []
        self.root_joint = None
    def seed(self, structure):
        self.root_joint = SimSpaceJoint(self.top, par_joint=None, structure=structure)
        self.joints.append(self.root_joint)
        self.generations.append([self.root_joint])
        return self.root_joint
    def spawn(self, n, authorize_spawn_fct=lambda node: True):
        new_joints = self.root_joint.spawn(n, authorize_spawn_fct)
        self.joints = self.joints + new_joints
        self.generations.append(new_joints)
        return new_joints
    def summarize(self):
        n_generations = len(self.generations)
        n_joints = len(self.joints)
        print "Generations # = %d   Joints # = %d" % (n_generations, n_joints)
        for idx, gen in enumerate(self.generations):
            print "  Gen %d : Joints %d" % (idx, len(gen))
        self.root_joint.summarize()
    def writeParentChildPairs(self, outfile='out.tree.txt'):
        ofs = open(outfile, 'w')
        self.root_joint.writeParentChildPairs(ofs)
        ofs.close()
        return
    def addPairPotential(self, options):
        n_joints = len(self.joints)
        for i in range(n_joints):
            node_i = self.joints[i].node
            for j in range(i+1, n_joints):
                node_j = self.joints[j].node
                print "%d:%d" % (node_i.id, node_j.id),
                node_i.createPotential(node_j, options)
                node_j.createPotential(node_i, options)
            print ""
        return
    def clearPotentials(self):
        for joint in self.joints:
            joint.node.clearPotentials()
        return

class LJRepulsive(object):
    def __init__(self, node, options):
        self.sigma = float(options.get('potentials.lj.sigma'))
        self.struct = node.structure
        return
    def computeEnergy(self):
        coords = np.zeros((self.struct.n_particles, 3))
        for idx, part in enumerate(self.struct):
            coords[idx,:] = part.pos
        E = 0.
        for i in range(self.struct.n_particles):
            for j in range(i):
                r = np.sqrt(np.sum((coords[i, :] - coords[j, :]) ** 2))
                E += (self.sigma/r)**12
        return E
    def computeGradient(self):
        coords = np.zeros((self.struct.n_particles, 3))
        for idx, part in enumerate(self.struct):
            coords[idx,:] = part.pos            
        grad = np.zeros(coords.shape)
        for i in range(self.struct.n_particles):
            for j in range(i):
                dr = coords[i, :] - coords[j, :]
                r = np.sqrt(np.sum(dr**2))
                g = 12./self.sigma*(self.sigma/r)**13
                grad[i, :] += -g * dr / r
                grad[j, :] += g * dr / r
        return grad

def multi_evaluate_potential_energy(x0_system, nodes, loggers):
    # Assign & update positions
    n_nodes = len(nodes)
    n_coords = x0_system.shape[0]
    x0_system = x0_system.reshape((n_nodes, n_coords/n_nodes))    
    for idx, node in enumerate(nodes):
        x0 = x0_system[idx]
        opt_pidcs = np.array([ i for i in range(node.structure.n_particles) ])
        update_node(x0, node, opt_pidcs)
    # Compute energy
    energy = 0.
    for idx, node in enumerate(nodes):
        x0 = x0_system[idx]
        opt_pidcs = [ i for i in range(node.structure.n_particles) ]
        trjlog = loggers[idx]
        update = False
        energy += evaluate_potential_energy(
            x0, node, node.potentials, node.potentials_self, opt_pidcs, 
            trjlog, update)
    print "==== %+1.7e ====" % energy
    return energy
    
def multi_evaluate_potential_gradient(x0_system, nodes, loggers):
    # Assign & update positions
    n_nodes = len(nodes)
    n_coords = x0_system.shape[0]
    x0_system = x0_system.reshape((n_nodes, n_coords/n_nodes))    
    for idx, node in enumerate(nodes):
        x0 = x0_system[idx]
        opt_pidcs = np.array([ i for i in range(node.structure.n_particles) ])
        update_node(x0, node, opt_pidcs)
    # Compute gradients
    gradients = []
    for idx, node in enumerate(nodes):
        x0 = x0_system[idx]
        opt_pidcs = [ i for i in range(node.structure.n_particles) ]
        trjlog = loggers[idx]
        update = False
        gradients_add = evaluate_potential_gradient(
            x0, node, node.potentials, node.potentials_self, opt_pidcs, 
            trjlog, update)
        gradients.append(gradients_add)
    gradients = np.array(gradients)
    gradients = gradients.flatten()
    return gradients        

def multi_optimize_node(nodes, x0):
    # Log trajectories
    loggers = []
    for node in nodes:
        logger = TrajectoryLogger('out.opt.%d.xyz' % node.id, mode='w')
        logger.logFrame(node.structure)
        loggers.append(logger)
    # Interface to optimizer
    f = multi_evaluate_potential_energy
    fprime = multi_evaluate_potential_gradient
    args = (nodes, loggers)
    # Run optimizer
    x0_opt, f_opt, n_calls, n_grad_calls, warnflag = scipy.optimize.fmin_cg(
        f=f, 
        x0=x0, 
        fprime=fprime, 
        args=args, 
        gtol=1e-6,
        full_output=True)
    if warnflag > 0:
        print "Warnflag =", warnflag, " => try again."
        return_code = False
    else:
        return_code = True
    # Close trajectory
    for logger, node in zip(loggers, nodes):
        logger.logFrame(node.structure)
        logger.close()
    return return_code
    
def update_node(x0, node, opt_pidcs):
    pid_pos = x0.reshape((opt_pidcs.shape[0],3))
    for rel_idx, abs_idx in enumerate(opt_pidcs):
        pos = pid_pos[rel_idx,:]
        particle = node.structure.getParticle(abs_idx+1)        
        particle.pos = pos
    node.acquire()
    return

def evaluate_potential_energy(x0, node, potentials, potentials_self, opt_pidcs, trjlog, update=True):
    if update: update_node(x0, node, opt_pidcs)
    # Compute energy
    energy = 0.
    for pot in potentials:
        energy += pot.computeEnergy()
    # Structure self-energy
    energy_self = 0.
    for pot in potentials_self:
        energy_self += pot.computeEnergy()
    energy_total = energy + energy_self    
    print "%+1.7e %+1.7e %+1.7e" % (energy_total, energy, energy_self)
    return energy_total
    
def evaluate_potential_gradient(x0, node, potentials, potentials_self, opt_pidcs, trjlog, update=True):
    if update: update_node(x0, node, opt_pidcs)
    # Compute gradients
    gradients = None
    target = None
    for pot in potentials:
        # Ascertain that target is the same for all potentials
        if target == None:
            target = pot.target
            gradients = pot.computeGradients()
        else:
            assert pot.target == target
            gradients = gradients + pot.computeGradients()
    trjlog.logFrame(target.structure)
    # Gradients from structure-internal potentials
    gradients_self = np.zeros(gradients.shape)
    for pot in potentials_self:
        gradients_self = gradients_self + pot.computeGradient()
    gradients = gradients + gradients_self
    # Short-list
    gradients_short = gradients[opt_pidcs]
    gradients_short = gradients_short.flatten()
    return gradients_short

def optimize_node(node, potentials, potentials_self, opt_pidcs, x0):
    # Log trajectory
    trjlog_opt = TrajectoryLogger('out.opt.xyz', mode='a')
    trjlog_opt.logFrame(node.structure)
    # Interface to optimizer
    f = evaluate_potential_energy
    fprime = evaluate_potential_gradient
    args = (node, potentials, potentials_self, opt_pidcs, trjlog_opt)
    x0 = np.array(x0[opt_pidcs])
    # Run optimizer
    x0_opt, f_opt, n_calls, n_grad_calls, warnflag = scipy.optimize.fmin_cg(
        f=f, 
        x0=x0, 
        fprime=fprime, 
        args=args, 
        gtol=1e-6,
        full_output=True)
    if warnflag > 0:
        print "Warnflag =", warnflag, " => try again."
        return_code = False
    else:
        return_code = True
    # Close trajectory
    trjlog_opt.logFrame(node.structure)
    trjlog_opt.close()
    return return_code

