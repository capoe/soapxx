#! /usr/bin/env python
import numpy as np
import sklearn.manifold
import sklearn.decomposition

def compute_dist_matrix_2d(X, Y):
    dX = np.subtract.outer(X, X)
    dY = np.subtract.outer(Y, Y)
    return dX, dY, (dX*dX + dY*dY)**0.5

def compute_dist_matrix_3d(X, Y, Z):
    dX = np.subtract.outer(X, X)
    dY = np.subtract.outer(Y, Y)
    dZ = np.subtract.outer(Z, Z)
    return dX, dY, (dX*dX + dY*dY + dZ*dZ)**0.5

class BondNetwork(object):
    def __init__(self, tags, K, M):
        """
        Args:
            tags (list of str, len=N): string labels for nodes
            K (np.ndarray, shape=(N,N)): kernel matrix, measuring similarity between nodes
            M (np.ndarray, shape=(N,)): list of node masses
        Example:
            >>> bond_network = BondNetwork(tags, K, M)
            >>> bond_network.initialise(method='kernelpca')
            >>> bond_network.integrate_md(
            >>>    n_steps=7000,
            >>>    rms_cut=1e-7,
            >>>    dt=0.01,
            >>>    dn_out=10,
            >>>    append_traj=False)
            >>> ofs = open('confout.txt', 'w')
            >>> for i in range(bond_network.N):
            >>>     ofs.write('%20s %3d %+1.7e %+1.7e\n' % (
            >>>         bond_network.tags[i], bond_network.M[i], 
            >>>         bond_network.X[i], bond_network.Y[i]))
            >>> ofs.close()
        """
        self.tags = tags
        self.N = len(tags)
        self.K = K
        self.D = (1. - K**2 + 1e-7)**0.5
        self.M = M # Masses
        # Exclusions and constraints
        self.have_excl = False
        self.have_constr = False
        assert K.shape[0] == self.N
        assert K.shape[1] == self.N
    def initialise(self, method):
        # Positions
        if method == 'random':
            X = np.random.uniform(size=self.N)
            Y = np.random.uniform(size=self.N)
        elif method in ['mds', 'kernelpca']:
            pos_mat = dimred_matrix(
                method=method, #'kernelpca', #'mds', 
                kmat=self.K,
                distmat=self.D, 
                symmetrize=True, 
                outfile=None)
            self.X = pos_mat[:,0]
            self.Y = pos_mat[:,1]
        else: raise NotImplementedMethod(method)
        # Velocities
        self.Vx = np.zeros((self.N,), dtype='float64')
        self.Vy = np.zeros((self.N,), dtype='float64')
        # Equilibrium distances
        self.R0 = self.D
        # Harmonic force constants
        alpha = 10. # 10.
        beta = 1.
        k_t = 0.85 # 0.85 # 0.85
        self.H = beta*0.5*(
            1.+np.tanh(alpha*(self.K-k_t))
        )
        # Universal force constant?
        self.H.fill(0.28) # NOTE HACK
        np.fill_diagonal(self.H, 0.0)
        return
    def steep(self, n_steps, dt):
        """
        TODO Turn this into a proper gradient descent (step size opt etc.)
        """
        ofs = open('traj.xyz', 'w')
        for i in range(n_steps):
            # Compute distance matrix
            dX, dY, dR = compute_dist_matrix_2d(self.X, self.Y)
            # Compute forces
            Fx_harm = self.H * (dR - self.R0) * dX  # Xij = xi-xj. Thus dR > R0 => Fij ~ -(xi-xj) is force experienced by j
            Fy_harm = self.H * (dR - self.R0) * dY
            Fx_harm = np.sum(Fx_harm, axis=1)
            Fy_harm = np.sum(Fy_harm, axis=1)
            # Integrate
            rms = (1./self.N * dt**2 * (np.dot(Fx_harm, Fx_harm) + np.dot(Fy_harm, Fy_harm)))**0.5
            self.X = self.X - Fx_harm*dt
            self.Y = self.Y - Fy_harm*dt
            # Energy
            E = np.sum(0.5*self.H*(dR - self.R0)**2)
            print "Step %d   %+1.7e   %+1.7e" % (i, rms, E)
            # Trajectory
            scale = 100.
            ofs.write('%d\n\n' % (self.N))
            for i in range(self.N):
                if 'OR' in self.tags[i]:
                    type_ = 'O'
                else:
                    type_ = 'C'
                ofs.write('%s %+1.7e %+1.7e 0.000\n' % (type_, self.X[i]*scale, self.Y[i]*scale))
        ofs.close()
        return self.X, self.Y 
    def exclude_below_mass(self, m_th):
        self.have_excl = True
        self.idcs_excl = np.where(self.M < m_th)
        self.idcs_noexcl = np.where(self.M >= m_th)
    def constrain_above_mass(self,m_th):
        self.have_constr = True
        self.idcs_constr = np.where(self.M > m_th)
        self.idcs_free = np.where(self.M <= m_th)
    def constrain_below_mass(self,m_th):
        self.have_constr = True
        self.idcs_constr = np.where(self.M < m_th)
        self.idcs_free = np.where(self.M >= m_th)
    def integrate_md(self, n_steps, dt, append_traj=False, dn_out=1, rms_cut=1.e-5):
        ofs = open('traj.xyz', 'a' if append_traj else 'w')
        gamma = 1.
        if self.have_excl:
            print "Noexcl", self.idcs_noexcl
            print "Constrain", self.idcs_constr
        for i in range(n_steps):
            # Compute distance matrix
            dX, dY, dR = compute_dist_matrix_2d(self.X, self.Y)
            # Update H
            self.H.fill(0.28)
            np.fill_diagonal(self.H, 0.0)
            bool_D = np.zeros(self.R0.shape, dtype='i4')
            bool_R = np.zeros(dR.shape, dtype='i4')
            bool_D[np.where(self.R0 > 0.8)] = 1
            bool_R[np.where(dR > 0.8)] = 1
            self.H[np.where((bool_D+bool_R) > 1)] = 0.0
            # Compute forces
            Fx_harm = self.H * (dR - self.R0) * dX  # Xij = xi-xj. Thus dR > R0 => Fij ~ -(xi-xj) is force experienced by j
            Fy_harm = self.H * (dR - self.R0) * dY
            # ... Exclusions
            if self.have_excl:
                Fx_harm[:,self.idcs_excl] = 0.0
                Fy_harm[:,self.idcs_excl] = 0.0
            # ... Sum pair forces
            Fx_harm = np.sum(Fx_harm, axis=1)
            Fy_harm = np.sum(Fy_harm, axis=1)
            # Integrate
            rms = (1./self.N * dt**2 * (np.dot(Fx_harm, Fx_harm) + np.dot(Fy_harm, Fy_harm)))**0.5
            self.Vx = (1.-gamma*dt)*self.Vx - Fx_harm*dt #/self.M
            self.Vy = (1.-gamma*dt)*self.Vy - Fy_harm*dt #/self.M
            # ... Constraints
            if self.have_constr:
                self.Vx[self.idcs_constr] = 0.0
                self.Vy[self.idcs_constr] = 0.0
            delta_X = self.Vx*dt
            delta_Y = self.Vy*dt
            self.X = self.X + delta_X
            self.Y = self.Y + delta_Y
            rms = (np.sum(delta_X*delta_X+delta_Y*delta_Y)/self.N)**0.5
            rms_max = np.max((delta_X*delta_X+delta_Y*delta_Y)**0.5)
            # Energy
            if self.have_excl:
                E = np.sum(0.5*self.H[self.idcs_noexcl][:,self.idcs_noexcl]*(
                    dR[self.idcs_noexcl][:,self.idcs_noexcl]
                  - self.R0[self.idcs_noexcl][:,self.idcs_noexcl])**2)
            else:
                E = np.sum(0.5*self.H*(dR - self.R0)**2)
            # Trajectory
            if i % dn_out == 0:
                print "Step %5d   rms/max %+1.7e/%+1.7e  energy %+1.7e" % (i, rms, rms_max, E)
                scale = 100.
                ofs.write('%d\n\n' % (self.N))
                for i in range(self.N):
                    if 'OR' in self.tags[i]:
                        type_ = 'O'
                    else:
                        type_ = 'C'
                    ofs.write('%s %+1.7e %+1.7e 0.000\n' % (type_, self.X[i]*scale, self.Y[i]*scale))
            if rms < rms_cut and rms_max < rms_cut:
                print "*** converged ***"
                break
        print "Step %5d   rms/max %+1.7e/%+1.7e  energy %+1.7e" % (i, rms, rms_max, E)
        ofs.close()
        return self.X, self.Y 
    def write_confout(self, filename):
        ofs = open(filename, 'w')
        for i in range(self.N):
            ofs.write('%10s %1.2e %+1.4e %+1.4e\n' % (
                self.tags[i], self.M[i], self.X[i], self.Y[i]))
        ofs.close()
        return
        
            
def dimred_matrix(method, kmat=None, distmat=None, outfile=None, ix=None, symmetrize=False, prj_dimension=2):
    if symmetrize:
        if type(kmat) != type(None):
            kmat = 0.5*(kmat+kmat.T)
        if type(distmat) != type(None):
            distmat = 0.5*(distmat+distmat.T)
    if method == 'mds':
        # MDS
        # http://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html#sklearn.manifold.MDS
        mds = sklearn.manifold.MDS(
            n_components=prj_dimension,
            metric=True,
            verbose=1,
            n_init=10,
            dissimilarity='precomputed')
        positions = mds.fit_transform(distmat)
    elif method == 'isomap':
        isomap = sklearn.manifold.Isomap(
            n_neighbors=5,
            n_components=prj_dimension,
            eigen_solver='auto',
            path_method='auto',
            neighbors_algorithm='auto')
        positions = isomap.fit_transform(ix)
    elif method == 'kernelpca':
        kernelpca = sklearn.decomposition.KernelPCA(
            n_components=None,
            kernel='precomputed',
            eigen_solver='auto',
            max_iter=None,
            remove_zero_eig=True)
        positions = kernelpca.fit_transform(kmat)
    elif method == 'diffmap':
        positions = diffusion_map(
            kmat=kmat,
            n_components=prj_dimension)
    else: raise NotImplementedError(method)
    if outfile: np.savetxt(outfile, positions)
    return positions

def dimred(kernelpot, method, outfile, symmetrize=False):
    ix = kernelpot.IX
    kmat = kernelpot.kernelfct.computeBlockDot(ix, return_distance=False)
    distmat = kernelpot.kernelfct.computeBlockDot(ix, return_distance=True)
    positions = dimred_matrix(method, kmat, distmat, outfile, ix, symmetrize=symmetrize)
    return positions

def diffusion_map(kmat, n_components, alpha=0.5):
    # Normalize kernel matrix 
    # (correct for non-unity diagonal components)
    kmat_diag = kmat.diagonal()
    kmat_norm = kmat/kmat_diag
    kmat_norm = kmat_norm.T/kmat_diag
    np.fill_diagonal(kmat_norm, 1.)
    # Rescale kernel
    D0 = np.sum(kmat_norm, axis=1)
    D0_diag = np.zeros(kmat_norm.shape)
    np.fill_diagonal(D0_diag, D0)
    kmat_norm = kmat_norm/D0**alpha
    kmat_norm = kmat_norm.T/D0**alpha
    # Form Markov matrix
    D1 = np.sum(kmat_norm, axis=1)
    kmat_norm = (kmat_norm/D1).T
    # Decompose
    import scipy.sparse
    evals, evecs = scipy.sparse.linalg.eigs(kmat_norm, n_components+1, which="LM")
    order = evals.argsort()[::-1]
    evals = evals[order]
    evecs = evecs.T[order].T
    # Project
    return kmat_norm.dot(evecs[:,1:])















