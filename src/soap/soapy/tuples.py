import numpy as np
import pickle
import soap
import momo
log = momo.osio

def get_types_pairs_default(reduced=True):
    types = 'C N O S H F Cl Br I P B'.split()
    r = 0 if reduced else 1
    pairs = [
        ['C:C'  , 1],  ['C:N'  , 1],  ['C:O'  , 1],  ['C:S'  , 1],  ['C:H'  , 1],  ['C:F'  , 1],  ['C:Cl' , 1],  ['C:Br' , 1],  ['C:I'  , 1],  ['C:P'  , 1],  ['C:B'  , 1],
        ['N:N'  , 1],  ['N:O'  , 1],  ['N:S'  , 1],  ['N:H'  , 1],  ['N:F'  , 1],  ['N:Cl' , 1],  ['N:Br' , 1],  ['N:I'  , 1],  ['N:P'  , 1],  ['N:B'  , 1],
        ['O:O'  , 1],  ['O:S'  , 1],  ['O:H'  , 1],  ['O:F'  , 1],  ['O:Cl' , 1],  ['O:Br' , 1],  ['O:I'  , 1],  ['O:P'  , 1],  ['O:B'  , 1],
        ['S:S'  , 1],  ['S:H'  , 1],  ['S:F'  , 1],  ['S:Cl' , 1],  ['S:Br' , 1],  ['S:I'  , 1],  ['S:P'  , 1],  ['S:B'  , 1],
        ['H:H'  , 1],  ['H:F'  , 1],  ['H:Cl' , 1],  ['H:Br' , 1],  ['H:I'  , 1],  ['H:P'  , 1],  ['H:B'  , 1],
        ['F:F'  , 1],  ['F:Cl' , 1],  ['F:Br' , r],  ['F:I'  , r],  ['F:P'  , r],  ['F:B'  , r],
        ['Cl:Cl', 1],  ['Cl:Br', r],  ['Cl:I' , r],  ['Cl:P' , r],  ['Cl:B' , r],
        ['Br:Br', 1],  ['Br:I' , r],  ['Br:P' , r],  ['Br:B' , r],
        ['I:I'  , r],  ['I:P'  , r],  ['I:B'  , r],
        ['P:P'  , r],  ['P:B'  , r],
        ['B:B'  , r]
    ]
    pairs = filter(lambda p: p[1] > 0, pairs)
    pairs = [ p[0] for p in pairs ]
    return types, pairs

def get_types_pairs_cnosh():
    types = 'C N O S H'.split()
    pairs = [
        ['C:C'  , 1],  ['C:N'  , 1],  ['C:O'  , 1],  ['C:S'  , 1],  ['C:H'  , 1],  
        ['N:N'  , 1],  ['N:O'  , 1],  ['N:S'  , 1],  ['N:H'  , 1],
        ['O:O'  , 1],  ['O:S'  , 1],  ['O:H'  , 1],
        ['S:S'  , 1],  ['S:H'  , 1],
        ['H:H'  , 1],
    ]
    pairs = filter(lambda p: p[1] > 0, pairs)
    pairs = [ p[0] for p in pairs ]
    return types, pairs

def get_types_pairs(elements):
    types = elements[:]
    pairs = []
    for i in range(len(types)):
        for j in range(i, len(types)):
            pairs.append("%s:%s" % (types[i], types[j]))
    return types, pairs

class TypeBasis(object):
    def __init__(self, types, pairs):
        self.types = types
        self.pairs = pairs
        self.N = len(self.types)
        self.NN = len(self.pairs)
        self.encoder_types = { t: tidx for tidx, t in enumerate(self.types) }
        self.encoder_pairs = { p: pidx for pidx, p in enumerate(self.pairs) }
        self.decoder_types = { tidx: t for t, tidx in self.encoder_types.iteritems() }
        self.K_abc = None
    def setupPairContractionTensor(self, mode):
        if mode == 'constant':
            self.K_abc = np.ones((self.N,self.N,1))
        elif mode == 'linear':
            self.K_abc = 0.5*(
                np.einsum('ac,b->abc', np.identity(self.N), np.ones((self.N,))) \
              + np.einsum('bc,a->abc', np.identity(self.N), np.ones((self.N,))) \
            )
        elif mode == 'bilinear':
            self.K_abc = np.zeros((self.N, self.N, self.NN))
            for i in range(self.N):
                ti = self.types[i]
                for j in range(self.N):
                    tj = self.types[j]
                    tij = '%s:%s' % (ti,tj)
                    if not tij in self.encoder_pairs:
                        tij = '%s:%s' % (tj,ti)
                    self.K_abc[i,j,self.encoder_pairs[tij]] = 1.
        else: raise ValueError("Mode: '%s'" % mode)
    def encodeType(self, t):
        return self.encoder_types[t]
    def decodeType(self, tidx):
        return self.decoder_types[tidx]
    def encodePair(self, t1, t2):
        t1, t2 = tuple(sorted([t1, t2], key=lambda t: self.encoder_types[t]))
        p_str = '%s:%s' % (t1, t2)
        if not p_str in self.encoder_pairs: return None
        return self.encoder_pairs[p_str]

class BasisRThetaPhi(object):
    def __init__(self, r_excl=None, r_cut=None, r_cut_width=None, sigma=None, sigma_ang=None):
        self.r_excl = r_excl
        self.r_cut = r_cut
        self.r_cut_width = r_cut_width
        self.sigma = sigma
        self.sigma_ang = sigma_ang
        self.r_theta_phi = None
        self.dr_dtheta_dphi = None
        self.r_centres = []
        if r_excl != None: self.setup()
    def setup(self, eps=1e-10):
        r_span = self.r_cut - self.r_excl
        r_n_fcts = int(r_span/self.sigma+1)
        r_spacing = r_span/(r_n_fcts-1)
        self.r_centres = [ self.r_excl + j*r_spacing for j in range(r_n_fcts) ]
        self.r_theta_phi = []
        self.dr_dtheta_dphi = []
        dr = self.sigma
        dr_ang = self.sigma_ang
        for r_c in self.r_centres:
            dtheta = dr_ang/r_c
            theta_span = np.pi
            theta_n_fcts = int(theta_span/dtheta+1)
            log << "  r = %1.4f => %d theta fcts" % (r_c, theta_n_fcts) << log.endl
            theta_spacing = theta_span/(theta_n_fcts-1)
            theta_centres = [ j*theta_spacing for j in range(theta_n_fcts) ]
            for theta_c in theta_centres:
                if theta_c < eps or theta_c > np.pi-eps:
                    phi_centres = [ 0. ]
                    dphi = dtheta
                else:
                    dphi = dtheta/np.sin(theta_c)
                    phi_span = 2*np.pi
                    phi_n_fcts = int(phi_span/dphi)
                    phi_spacing = phi_span/phi_n_fcts
                    phi_centres = [ j*phi_spacing for j in range(phi_n_fcts) ]
                for phi_c in phi_centres:
                    self.r_theta_phi.append([r_c, theta_c, phi_c])
                    self.dr_dtheta_dphi.append([dr, dtheta, dphi])
        self.r_theta_phi = np.array(self.r_theta_phi)
        self.dr_dtheta_dphi = np.array(self.dr_dtheta_dphi)
        log << "Basis: %d fcts" % len(self.r_theta_phi) << log.endl
        log << "  R centres @" << self.r_centres << log.endl
    def asXyz(self, outfile='basis.xyz', weights=None, R=[], translate=None):
        if translate is None: translate = np.zeros((3,))
        ofs = open(outfile, 'w')
        ofs.write('%d\n\n' % (len(self.r_theta_phi)+len(R)))
        for i in range(len(self.r_theta_phi)):
            r = self.r_theta_phi[i]
            x = r[0]*np.sin(r[1])*np.cos(r[2]) + translate[0]
            y = r[0]*np.sin(r[1])*np.sin(r[2]) + translate[1]
            z = r[0]*np.cos(r[1]) + translate[2]
            ofs.write('X %+1.4f %+1.4f %+1.4f %+1.4e\n' % (
                x, y, z, 0.0 if weights is None else weights[i]))
        R = R + translate
        for j in range(len(R)):
            ofs.write('Y %+1.4f %+1.4f %+1.4f %+1.4e\n' % (R[j,0], R[j,1], R[j,2], 10.))
        ofs.close()
    def getXyz(self):
        return np.concatenate([
            self.r_theta_phi[:,0]*np.sin(self.r_theta_phi[:,1])*np.cos(self.r_theta_phi[:,2]),
            self.r_theta_phi[:,0]*np.sin(self.r_theta_phi[:,1])*np.sin(self.r_theta_phi[:,2]),
            self.r_theta_phi[:,0]*np.cos(self.r_theta_phi[:,1])]).reshape((3,-1)).T
    def calcWeights(self, dR):
        w = np.heaviside(-dR+self.r_cut, 0.0)
        transition_idcs = np.where(w*dR > self.r_cut-self.r_cut_width)[0]
        w[transition_idcs] = np.cos(0.5*(dR[transition_idcs]-self.r_cut+self.r_cut_width)/(
            self.r_cut_width)*np.pi)
        return w
    def expandVectors(self, R):
        G = np.zeros((self.r_theta_phi.shape[0],))
        if len(R) > 0:
            r = np.sum(R*R, axis=1)**0.5
            t = np.arccos(R[:,2]/r)
            p = np.arctan2(R[:,1], R[:,0])
            rtp = np.concatenate([ r, t, p ]).reshape((3,-1)).T
            weights = self.calcWeights(r)
            # >>> X_re = rtp[:,0]*np.sin(rtp[:,1])*np.cos(rtp[:,2])
            # >>> Y_re = rtp[:,0]*np.sin(rtp[:,1])*np.sin(rtp[:,2])
            # >>> Z_re = rtp[:,0]*np.cos(rtp[:,1])
            # >>> R[:,0] = R[:,0] - X_re
            # >>> R[:,1] = R[:,1] - Y_re
            # >>> R[:,2] = R[:,2] - Z_re
            # >>> assert np.max(np.abs(R)) < 1e-10
            for i in range(R.shape[0]):
                Gi = (self.r_theta_phi-rtp[i])
                Gi[:,2] = Gi[:,2] - np.round(Gi[:,2]/(2*np.pi))*2*np.pi # wrap phi
                Gi = np.exp(-Gi**2/(2*self.dr_dtheta_dphi**2))
                Gi = Gi[:,0]*Gi[:,1]*Gi[:,2]
                Gi = Gi/np.sum(Gi)
                G = G + weights[i]*Gi
            # >>> self.asXyz(weights=G, R=R)
        return G
    def save(self, jarfile):
        open(jarfile, 'w').write(pickle.dumps(self))
    def load(self, jarfile):
        self = pickle.load(open(jarfile, 'rb'))
        return self
         
class Basis(object):
    def __init__(self, r_excl=None, r_cut=None, r_cut_width=None, 
            sigma=None, type_basis=None, r_err=None, 
            weight_decay_length=None, weight_constant=False, log=log):
        self.type_basis = type_basis
        self.r_excl = r_excl
        self.r_cut = r_cut
        self.r_cut_width = r_cut_width
        self.r_err = r_err
        self.weight_decay_length = weight_decay_length
        self.weight_constant = weight_constant
        self.sigma = sigma
        self.ranges = None
        self.sigmas = None
        self.centres = None
        self.pair_dim = None
        self.triplet_dim = None
        self.pair_ch_dim = None
        self.triplet_ch_dim = None
        if self.r_cut is not None: self.setup(log=log)
    def setup(self, log=log, sigmas=None):
        self.ranges = [
            [ 0, self.r_cut ],
            [ 2*self.r_excl, 2*self.r_cut ],
            [ 0, self.r_cut-self.r_excl ],
            [ self.r_excl, 2*self.r_cut ],
        ]
        self.sigmas = sigmas
        if self.sigmas is None:
            self.sigmas = np.array([
                self.sigma,
                2**0.5*self.sigma,
                2**0.5*self.sigma,
                2**0.5*self.sigma
            ])
        self.centres = [
        ]
        for i in range(len(self.ranges)):
            rng = self.ranges[i][1] - self.ranges[i][0]
            sig = self.sigmas[i]
            n_fcts = int(rng/sig+1.)
            dr = rng/(n_fcts-1)
            self.centres.append(np.array(
                [ self.ranges[i][0]+j*dr for j in range(n_fcts) ]
            ))
        self.pair_dim = len(self.centres[0])
        self.triplet_dim = len(self.centres[1])*len(self.centres[2])*len(self.centres[3])
        self.pair_ch_dim = self.type_basis.N
        self.triplet_ch_dim = self.type_basis.NN
        log << "Basis:    Pair chs: %3d    Triplet chs: %3d" % (
            self.pair_ch_dim, self.triplet_ch_dim) << log.endl
        log << "Basis:    Pair dim: %3d    Triplet dim: %3d" % (
            self.pair_dim, self.triplet_dim) << log.endl
        for i in range(len(self.centres)):
            log << "  G%d: %2d fcts  sigma=%1.2f" % (
                i, len(self.centres[i]), self.sigmas[i]) << log.flush
            log << " r0=%1.4f, r1=%1.4f, ... rn=%1.4f" % (
                self.centres[i][0], self.centres[i][1], self.centres[i][-1]) << log.endl
    def calcWeights(self, dR):
        w = np.heaviside(-dR+self.r_cut, 0.0)
        transition_idcs = np.where(w*dR > self.r_cut-self.r_cut_width)[0]
        w[transition_idcs] = 0.5*(1.+np.cos((dR[transition_idcs]-self.r_cut+self.r_cut_width)/(
            self.r_cut_width)*np.pi))
        dw2_w2 = np.zeros_like(w)
        dw2_w2[transition_idcs] = (1. - (2*w[transition_idcs] - 1.)**2)/w[transition_idcs]**2 * (
            np.pi*self.r_err/(self.r_cut_width+1e-10))**2
        return w, dw2_w2
    def expandPairs(self, pair_types, pairs):
        X = np.zeros((self.type_basis.N*self.pair_dim,))
        dX = np.zeros((self.type_basis.N*self.pair_dim,))
        if len(pairs) > 0:
            assert pairs.shape[1] == 3 # [ ri, hi, dhi2_hi2]
            G = np.exp(
                -(pairs[:,0].reshape((-1,1)) - self.centres[0])**2/(2.*self.sigmas[0]**2)
            )
            V = ((pairs[:,0].reshape((-1,1)) - self.centres[0])/self.sigmas[0])**2 * (
                self.r_err/self.sigmas[0])**2
            W = pairs[:,1]
            dW2_W2 = pairs[:,2]
            G = (G.T*W).T
            V = (V.T+dW2_W2).T
            block_size = self.pair_dim
            assert G[0].shape[0] == self.pair_dim
            for i in range(len(pair_types)):
                c = self.type_basis.encodeType(pair_types[i])
                X[c*block_size:(c+1)*block_size] = X[c*block_size:(c+1)*block_size] + G[i]
                dX[c*block_size:(c+1)*block_size] = dX[c*block_size:(c+1)*block_size] + G[i]**2*V[i]
        return X, dX**0.5
    def expandTriplets(self, trip_types, trips):
        X = np.zeros((self.type_basis.NN*self.triplet_dim,))
        dX = np.zeros((self.type_basis.NN*self.triplet_dim,)) # <<<
        if len(trips) > 0:
            assert trips.shape[1] == 5 # [ ri+rj, |ri-rj|, rij, hij, dhij2_hij2 ]
            G = []
            V = []
            for i in range(3):
                Ci = self.centres[i+1]
                Si = self.sigmas[i+1]
                Gi = np.exp(
                    -(trips[:,i].reshape((-1,1)) - Ci)**2/(2.*Si**2)
                )
                G.append(Gi)
                Vi = ((trips[:,i].reshape((-1,1)) - Ci)/Si)**2 * (self.r_err/Si)**2
                V.append(Vi)
            GG = []
            VV = []
            for t in range(len(trips)):
                GGt = np.outer(
                    np.outer(G[0][t], G[1][t]).flatten(), G[2][t]
                ).flatten()
                GG.append(GGt)
                VVt = np.add.outer(
                    np.add.outer(V[0][t], V[1][t]).flatten(), V[2][t]
                ).flatten()
                VV.append(VVt)
            GG = np.array(GG)
            VV = np.array(VV)
            W = trips[:,3]
            dW2_W2 = trips[:,4]
            GG = (GG.T*W).T
            VV = (VV.T+dW2_W2).T
            assert GG[0].shape[0] == self.triplet_dim
            block_size = self.triplet_dim
            for i in range(len(trip_types)):
                c = self.type_basis.encodePair(*tuple(trip_types[i]))
                if c is None:
                    log << "Skip" << trip_types[i] << log.endl
                    continue
                X[c*block_size:(c+1)*block_size] = X[c*block_size:(c+1)*block_size] + GG[i]
                dX[c*block_size:(c+1)*block_size] = dX[c*block_size:(c+1)*block_size] + GG[i]**2*VV[i]
        return X, dX**0.5
    def save(self, jarfile):
        open(jarfile, 'w').write(pickle.dumps(self))
    def load(self, jarfile):
        self = pickle.load(open(jarfile, 'rb'))
        return self

def get_pairs_triplets(rc, T, R, D, basis, eps=1e-10, with_triplets=True):
    R = R-rc
    dR = np.sum(R**2, axis=1)**0.5
    pair_types = []
    pairs = []
    triplet_types = []
    triplets = []
    # Pairs: [ ri, hi, dhi2_hi2 ]
    for i in range(len(T)):
        pair_types.append(T[i])
        h = basis.sigma**2/dR[i]**2 if dR[i] > eps else 1.
        dh2_h2 = (2*basis.r_err/dR[i])**2 if dR[i] > eps else 0.
        pairs.append([ dR[i], h, dh2_h2 ])
    pairs = np.array(pairs)
    w_pairs, dw2_w2_pairs = basis.calcWeights(pairs[:,0])
    pairs[:,1] = pairs[:,1]*w_pairs
    pairs[:,2] = pairs[:,2] + dw2_w2_pairs
    if not with_triplets:
        return pair_types, pairs, triplet_types, trips
    # Triplets [ ri+rj, |ri-rj|, rij, hij, dhij2_hij2 ]
    for i in range(len(T)):
        ri = dR[i]
        if ri < eps: continue
        for j in range(i+1, len(T)):
            rj = dR[j]
            if rj < eps: continue
            rij = D[i,j]
            hij = basis.sigma**4/(ri*rj)**2
            dh2_h2 = (2*basis.r_err/ri)**2 + (2*basis.r_err/rj)**2
            triplet_types.append(sorted([T[i],T[j]]))
            triplets.append([ri+rj, np.abs(ri-rj), rij, hij, dh2_h2, ri, rj])
    triplets = np.array(triplets)
    if len(triplets) > 0:
        w_1, dw2_w2_1 = basis.calcWeights(triplets[:,5])
        w_2, dw2_w2_2 = basis.calcWeights(triplets[:,6])
        triplets[:,3] = triplets[:,3]*w_1*w_2
        triplets[:,4] = triplets[:,4] + dw2_w2_1 + dw2_w2_2
        triplets = triplets[:,0:5]
    return pair_types, pairs, triplet_types, triplets

def expand_structure(config, basis, centres=None, log=None, norm=True):
    T = np.array(config.symbols)
    R = config.positions
    D = soap.tools.partition.calculate_distance_mat(R, R)
    IX_pairs = []
    IX_trips = []
    dIX_pairs = []
    dIX_trips = []
    if centres is None: centres = range(len(T))
    for i in centres:
        nbs = np.where(D[i] <= basis.r_cut)[0]
        rc = R[i]
        pair_types, pairs, triplet_types, triplets = \
            get_pairs_triplets(rc, T[nbs], R[nbs], D[nbs][:,nbs], basis=basis)
        X_pairs, dX_pairs = basis.expandPairs(pair_types, pairs)
        if norm and len(pair_types) > 0:
            norm = np.dot(X_pairs, X_pairs)**0.5
            X_pairs = X_pairs/norm
            dX_pairs = dX_pairs/norm
        X_trips, dX_trips = basis.expandTriplets(triplet_types, triplets)
        if norm and len(triplet_types) > 0:
            norm = np.dot(X_trips, X_trips)**0.5
            X_trips = X_trips/norm
            dX_trips = dX_trips/norm
        IX_pairs.append(X_pairs)
        dIX_pairs.append(dX_pairs)
        IX_trips.append(X_trips)
        dIX_trips.append(dX_trips)
    return np.array(IX_pairs), np.array(dIX_pairs), np.array(IX_trips), np.array(dIX_trips)

def expand_structure_frame(config, basis, centres=None, log=None):
    def calc_Q_tensor(r, dr):
        Q = 3*np.outer(r,r) - dr**2*np.identity(3)
        return Q
    T = np.array(config.symbols)
    R = config.positions
    D = soap.tools.partition.calculate_distance_mat(R, R)
    if centres is None: centres = range(len(T))
    IX = []
    for i in centres:
        # Nb positions
        nbs = np.where(D[i] <= basis.r_cut)[0]
        rc = R[i]
        types_nbs = T[nbs]
        R_nbs = np.copy(R[nbs])
        R_nbs = R_nbs-rc
        dR_nbs = np.sum(R_nbs*R_nbs, axis=1)**0.5
        weights_nbs, dweights_nbs = basis.calcWeights(dR_nbs)
        # Frame
        dim_1 = basis.type_basis.N*3
        dim_2 = basis.type_basis.N*6
        X1 = np.zeros((dim_1,))
        X2 = np.zeros((dim_2,))
        triu = np.triu_indices(3,0)
        for nb in range(len(nbs)):
            w_nb = weights_nbs[nb]
            if dR_nbs[nb] > 1e-10: w_nb *= 1./dR_nbs[nb]**2
            t = basis.type_basis.encodeType(types_nbs[nb])
            x1 = w_nb*R_nbs[nb]
            x2 = w_nb*calc_Q_tensor(R_nbs[nb], dR_nbs[nb])[triu]
            X1[t*3:t*3+3] = X1[t*3:t*3+3] + x1
            X2[t*6:t*6+6] = X2[t*6:t*6+6] + x2
        X12 = np.concatenate([X1,X2])
        IX.append(X12)
    return np.array(IX)

def expand_structure_ipoints(config, centres, config_ipoints, centres_ipoints, basis):
    R = config.positions[centres]
    R_ipoints = config_ipoints.positions[centres_ipoints]
    D = soap.tools.partition.calculate_distance_mat(R, R_ipoints)
    IX = []
    IR = []
    for ii in range(len(centres)):
        rc = R[ii]
        nbs = np.where(D[ii] <= basis.r_cut)[0]
        R_nbs = np.copy(R_ipoints[nbs])
        R_nbs = R_nbs - rc
        X = basis.expandVectors(R_nbs)
        # >>> if len(nbs):
        # >>>     basis.asXyz(outfile='basis.xyz', weights=X, R=R_nbs, translate=rc)
        # >>>     raw_input('...xyz...')
        IX.append(X)
        IR.append(R_nbs)
    return np.array(IX), np.array(IR)

