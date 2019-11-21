import soap
import scipy.special
import numpy as np
import numpy.linalg as linalg
import tnn
from momo import log

class SphericalHarmonicSmoother(object):
    def __init__(self, lmax=7, subdivide=3, regular=0.000001, smooth=0.2, train=True):
        self.grid = soap.soapy.geodesic.subdivide_sphere(
            args_repeats=subdivide, args_radius=1.0)
        self.regular = regular
        self.smooth = smooth
        self.lmax = lmax
        self.Y = None
        self.K = None
        self.K_inv = None
        self.W = None
        self.isotropic = np.zeros((lmax+1, 2*lmax+1), dtype='complex128')
        self.isotropic[0,0] = scipy.special.sph_harm(0, 0, 0., 0.)
        if train: self.train()
        log << "SphHarm: Using grid of size"  << self.grid.shape[0] << log.endl
    def train(self):
        self.Y = self.computeAnalytical(self.grid)
        self.Y = np.array(self.Y)
        self.K = self.kernel(self.grid)
        K_reg = self.K + self.regular*np.identity(self.K.shape[0])
        self.K_inv = np.linalg.inv(K_reg)
        self.W = np.einsum('gh,hlm->glm', self.K_inv, self.Y, optimize='greedy')
    def kernel(self, g):
        return np.exp(1./self.smooth*(0.5*(1.+g.dot(self.grid.T))))
    def computeAnalytical(self, d, r=None, eps=1e-10):
        phi = np.arctan2(d[:,1], d[:,0])
        phi = np.heaviside(-phi, 0.0)*2*np.pi + phi
        theta = np.arccos(d[:,2])
        Ylm = np.zeros((len(d), self.lmax+1, 2*self.lmax+1), dtype='complex128')
        for l in range(self.lmax+1):
            for m in range(-l, l+1):
                Ylm[:,l,l+m] = scipy.special.sph_harm(m, l, phi, theta)
        if r is not None: Ylm[np.where(r < eps)] = self.isotropic
        return Ylm
    def computePredict(self, d, r=None, eps=1e-10):
        K = self.kernel(d)
        Y_pred = np.einsum('gh,hlm->glm', K, self.W, optimize='greedy')
        if r is not None: Y_pred[np.where(r < eps)] = self.isotropic
        return Y_pred
    def compute(self, d, r=None, eps=1e-10, predict=True):
        if predict: return self.computePredict(d=d, r=r, eps=eps)
        else: return self.computeAnalytical(d=d, r=r, eps=eps)

class GylmBasis(object):
    def __init__(self, lmax=7, ldamp=4, apply_lnorm=False, r_excl=None, r_cut=None, r_cut_width=None, 
            sigma=None, type_basis=None, 
            weight_decay_length=1., weight_decay_constant=False, weight_centre=1.0, 
            mode='equispaced', sigma_range=None, r_range=None, log=log):
        self.type_basis = type_basis
        # Angular
        self.lmax = lmax
        self.ldamp = ldamp
        self.lnorm = np.array([ (1./np.sqrt(2*l+1) if apply_lnorm else 1.) for l in range(self.lmax+1) ])
        self.ylm = None
        # Radial
        self.r_excl = r_excl
        self.r_cut = r_cut
        self.r_cut_width = r_cut_width
        self.weight_decay_length = weight_decay_length
        self.weight_decay_constant = weight_decay_constant
        self.weight_centre = weight_centre
        self.sigma = sigma
        self.mode = mode
        self.range = None
        self.sigmas = None
        self.centres = None
        # Adaptive
        self.sigma_range = sigma_range
        self.r_range = r_range
        # Dimension
        self.pair_dim = None
        self.triplet_dim = None
        self.pair_ch_dim = None
        self.triplet_ch_dim = None
        self.setup(log=log)
    def setup(self, log=log):
        self.range = [ self.r_excl, self.r_cut ]
        if self.mode == 'equispaced':
            rng = self.range[1] - self.range[0]
            n_fcts = int(rng/self.sigma+1.)
            dr = rng/(n_fcts-1)
            self.centres = np.array([ self.range[0]+j*dr for j in range(n_fcts) ])
            self.sigmas = np.array([ self.sigma for _ in range(len(self.centres)) ])
        elif self.mode == 'adaptive':
            assert self.sigma_range is not None
            sng = self.sigma_range[1] - self.sigma_range[0]
            sigma_avg = 0.5*(self.sigma_range[1] + self.sigma_range[0])
            rng = self.r_range[1] - self.r_range[0]
            n_fcts = int(rng/(sigma_avg/2.**0.5)+1.)
            dr = rng/(n_fcts-1)
            ds = sng/(n_fcts-1)
            self.centres = np.array([ self.range[0]+j*dr for j in range(n_fcts) ])
            self.sigmas = np.array([ self.sigma_range[0]+j*ds for j in range(n_fcts) ])
            print self.centres
            print self.sigmas
        else: raise ValueError(self.mode)
        self.triplet_dim = len(self.centres)**2*(self.lmax+1)
        self.triplet_ch_dim = self.type_basis.NN
        log << "GylmBasis:  Triplet chs: %3d  Triplet dim: %3d  Total dim: %5d" % (
            self.triplet_ch_dim, self.triplet_dim, 
            self.triplet_ch_dim*self.triplet_dim) << log.endl
        self.ylm = SphericalHarmonicSmoother(lmax=self.lmax)
        log << "Gn: %2d fcts" % (
            len(self.centres)) << log.flush
        log << " s0=%1.4f, s1=%1.4f, ... sn=%1.4f" % (
            self.sigmas[0], self.sigmas[1], self.sigmas[-1]) << log.endl
        log << " r0=%1.4f, r1=%1.4f, ... rn=%1.4f" % (
            self.centres[0], self.centres[1], self.centres[-1]) << log.endl

class GylmModule(tnn.torch.nn.Module):
    def __init__(self, basis, eps=1e-10, predict_ylm=True):
        super(GylmModule, self).__init__()
        self.basis = basis
        self.eps = eps
        self.predict_ylm = predict_ylm
    def forward(self, *args, **kwargs):
        T = args[0]
        R = args[1]
        W = args[2]
        R_shaped = R.reshape((-1,3))
        D_shaped = np.sqrt(np.sum(R_shaped**2, axis=1))
        # Types
        t = T.reshape((-1, R.shape[1]//3, self.basis.type_basis.N))
        # Jacobian weight
        J = evaluate_weight(
            dR=D_shaped, 
            scale=self.basis.weight_decay_length, 
            weight_centre=self.basis.weight_centre,
            constant=self.basis.weight_decay_constant)
        J = J.reshape((R.shape[0], R.shape[1]//3))
        # Ylm expansion
        E_shaped = (R_shaped.T/(D_shaped+self.eps)).T
        ylm = self.basis.ylm.compute(d=E_shaped, r=D_shaped, predict=self.predict_ylm)
        ylm = ylm.reshape((R.shape[0], R.shape[1]//3, ylm.shape[1], ylm.shape[2]))
        # Gn expansion
        d = np.subtract.outer(D_shaped, self.basis.centres)
        alpha = 1./(2.*self.basis.sigmas**2)
        gn = np.exp(-d**2*alpha)
        # Gnl (frequency damping)
        inv_d = self.basis.ldamp*self.basis.sigma/(np.sqrt(4.*np.pi)*D_shaped+self.eps)
        hl = np.exp(-np.einsum('i,l->il', inv_d, np.sqrt(2*np.arange(self.basis.lmax+1)), optimize='greedy'))
        gnl = np.einsum('in,il->inl', gn, hl, optimize='greedy')
        gnl = gnl.reshape((R.shape[0], R.shape[1]//3, gn.shape[1], self.basis.lmax+1))
        #gnl = np.einsum('canl,ca->canl', gnl, J, optimize='greedy')
        # >>> idx = 0
        # >>> for c in range(gnl.shape[0]):
        # >>>     for i in range(gnl[c].shape[0]):
        # >>>         for n in range(gnl[c][i].shape[0]):
        # >>>             for l in range(gnl[c][i][n].shape[0]):
        # >>>                 if W[c,i] > 0.5:
        # >>>                     print D_shaped[idx], n, l, gnl[c][i][n][l]
        # >>>         idx += 1
        # >>> log.okquit()
        # Reduce and contract
        # >>> qanlm = np.einsum('can,calm,ca,ca->canlm', gn, ylm, W, J, optimize='greedy')
        # >>> qtnlm = np.einsum('canlm,cat->ctnlm', qanlm, t, optimize='greedy')
        # >>> xvnkl = np.einsum('ctnlm,cuklm,tuv->cvnkl', 
        # >>>     qtnlm, np.conj(qtnlm), self.basis.type_basis.K_abc, optimize='greedy')
        qtnlm = np.einsum('cat,canl,calm,ca,ca,l->ctnlm', 
            t, gnl, ylm, W, J, self.basis.lnorm, optimize='greedy')
        #qtnlm = np.einsum('cat,canl,calm,ca,l->ctnlm', 
        #    t, gnl, ylm, W, self.basis.lnorm, optimize='greedy')
        xvnkl = (\
            np.einsum('ctnlm,cuklm,tuv->cvnkl', 
                qtnlm.real, qtnlm.real, self.basis.type_basis.K_abc, optimize='greedy') + \
            np.einsum('ctnlm,cuklm,tuv->cvnkl', 
                qtnlm.imag, qtnlm.imag, self.basis.type_basis.K_abc, optimize='greedy'))
        # Normalize
        xvnkl = xvnkl.reshape((T.shape[0],-1))
        xvnkl = (xvnkl.T/np.sqrt(np.sum(xvnkl**2, axis=1))).T
        return xvnkl

def evaluate_weight(dR, scale, weight_centre, eps=1e-4, constant=False):
    if constant: return np.ones_like(dR)
    t = np.exp(-dR**2/scale**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        w = (1.-t)*(scale/dR)**2 + t*(weight_centre-1.)
        near_zero = np.where(dR < eps)
        w[near_zero] = weight_centre
    return w

def build_gylm_graph(basis):
    g = tnn.ModuleGraph(tag="GYLM")
    t = g.create("T")
    r = g.create("R")
    w = g.create("W")
    x = g.create("X", parents=[t,r,w], module=GylmModule(basis))
    return g

def extract_environments(centres, targets, target_types, basis, 
        pad=True,
        with_nb_idcs=False, 
        max_envs_per_config=-1):
    if type(target_types) is list: target_types = np.array(target_types)
    R_out = []
    W_out = []
    T_out = []
    I_out = []
    D = soap.tools.partition.calculate_distance_mat(centres, targets)
    L = np.heaviside(basis.r_cut - D, 0.0)
    max_nbs = int(np.max(np.sum(L, axis=1))+0.5)
    for c in range(centres.shape[0]):
        nbs = np.where(L[c] > 0.5)[0]
        Wc = np.ones((len(nbs),))
        Rc = targets[nbs]
        Rc = Rc - centres[c]
        Rc = Rc.flatten()
        Tc = target_types[nbs]
        tids = ( np.arange(len(Tc)), [ basis.type_basis.encodeType(t) for t in Tc ] )
        Tc = np.zeros((len(Tc), basis.type_basis.N))
        Tc[tids] = 1.
        Tc = Tc.flatten()
        if pad and len(nbs) < max_nbs:
            n_pad = max_nbs - len(nbs)
            Wc = np.pad(Wc, [0, n_pad], 'constant', constant_values=0)
            Rc = np.pad(Rc, [0, 3*n_pad], 'constant', constant_values=0)
            Tc = np.pad(Tc, [0, basis.type_basis.N*n_pad], 'constant', constant_values=0)
            nbs = np.pad(nbs, [0, n_pad], 'constant', constant_values=-1)
        R_out.append(Rc)
        T_out.append(Tc)
        W_out.append(Wc)
        if with_nb_idcs: I_out.append(nbs)
    T_out = np.array(T_out)
    R_out = np.array(R_out)
    W_out = np.array(W_out)
    if with_nb_idcs: I_out = np.array(I_out)
    if max_envs_per_config > 0 and len(centres) > max_envs_per_config:
        sel = np.arange(len(centres))
        np.random.shuffle(sel)
        sel = sel[0:max_envs_per_config]
        T_out = T_out[sel]
        R_out = R_out[sel]
        W_out = W_out[sel]
        if with_nb_idcs: I_out = I_out[sel]
    return T_out, R_out, W_out, I_out


