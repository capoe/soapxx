import numpy as np
import pickle
import soap
import momo
log = momo.osio

def get_types_pairs_default():
    types = 'C N O S H F Cl Br I P B'.split()
    pairs = [
        ['C:C'  , 1],  ['C:N'  , 1],  ['C:O'  , 1],  ['C:S'  , 1],  ['C:H'  , 1],  ['C:F'  , 1],  ['C:Cl' , 1],  ['C:Br' , 1],  ['C:I'  , 1],  ['C:P'  , 1],  ['C:B'  , 1],
        ['N:N'  , 1],  ['N:O'  , 1],  ['N:S'  , 1],  ['N:H'  , 1],  ['N:F'  , 1],  ['N:Cl' , 1],  ['N:Br' , 1],  ['N:I'  , 1],  ['N:P'  , 1],  ['N:B'  , 1],
        ['O:O'  , 1],  ['O:S'  , 1],  ['O:H'  , 1],  ['O:F'  , 1],  ['O:Cl' , 1],  ['O:Br' , 1],  ['O:I'  , 1],  ['O:P'  , 1],  ['O:B'  , 1],
        ['S:S'  , 1],  ['S:H'  , 1],  ['S:F'  , 1],  ['S:Cl' , 1],  ['S:Br' , 1],  ['S:I'  , 1],  ['S:P'  , 1],  ['S:B'  , 1],
        ['H:H'  , 1],  ['H:F'  , 1],  ['H:Cl' , 1],  ['H:Br' , 1],  ['H:I'  , 1],  ['H:P'  , 1],  ['H:B'  , 1],
        ['F:F'  , 1],  ['F:Cl' , 1],  ['F:Br' , 0],  ['F:I'  , 0],  ['F:P'  , 0],  ['F:B'  , 0],
        ['Cl:Cl', 1],  ['Cl:Br', 0],  ['Cl:I' , 0],  ['Cl:P' , 0],  ['Cl:B' , 0],
        ['Br:Br', 1],  ['Br:I' , 0],  ['Br:P' , 0],  ['Br:B' , 0],
        ['I:I'  , 0],  ['I:P'  , 0],  ['I:B'  , 0],
        ['P:P'  , 0],  ['P:B'  , 0],
        ['B:B'  , 0]
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

class TypeBasis(object):
    def __init__(self, types, pairs):
        self.types = types
        self.pairs = pairs
        self.N = len(self.types)
        self.NN = len(self.pairs)
        self.encoder_types = { t: tidx for tidx, t in enumerate(self.types) }
        self.encoder_pairs = { p: pidx for pidx, p in enumerate(self.pairs) }
    def encodeType(self, t):
        return self.encoder_types[t]
    def encodePair(self, t1, t2):
        t1, t2 = tuple(sorted([t1, t2], key=lambda t: self.encoder_types[t]))
        p_str = '%s:%s' % (t1, t2)
        if not p_str in self.encoder_pairs: return None
        return self.encoder_pairs[p_str]

class Basis(object):
    def __init__(self, r_excl=None, r_cut=None, r_cut_width=None, 
            sigma=None, type_basis=None, log=log):
        self.type_basis = type_basis
        self.r_excl = r_excl
        self.r_cut = r_cut
        self.r_cut_width = r_cut_width
        self.sigma = sigma
        self.ranges = None
        self.sigmas = None
        self.centres = None
        self.pair_dim = None
        self.triplet_dim = None
        self.pair_ch_dim = None
        self.triplet_ch_dim = None
        if self.r_cut is not None: self.setup(log=log)
    def setup(self, log=log):
        self.ranges = [
            [ 0, self.r_cut ],
            [ 2*self.r_excl, 2*self.r_cut ],
            [ 0, self.r_cut-self.r_excl ],
            [ self.r_excl, 2*self.r_cut ],
        ]
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
                i, len(self.centres[i]), self.sigmas[i]) << log.endl
    def calcWeights(self, dR):
        w = np.heaviside(-dR+self.r_cut, 0.0)
        transition_idcs = np.where(w*dR > self.r_cut-self.r_cut_width)[0]
        w[transition_idcs] = np.cos(0.5*(dR[transition_idcs]-self.r_cut+self.r_cut_width)/(self.r_cut_width)*np.pi)
        return w
    def expandPairs(self, pair_types, pairs):
        X = np.zeros((self.type_basis.N*self.pair_dim,))
        if len(pairs) > 0:
            assert pairs.shape[1] == 2
            G = np.exp(
                -(pairs[:,0].reshape((-1,1)) - self.centres[0])**2/(2.*self.sigmas[0]**2)
            )
            W = pairs[:,1]
            G = (G.T*W).T
            block_size = self.pair_dim
            assert G[0].shape[0] == self.pair_dim
            for i in range(len(pair_types)):
                c = self.type_basis.encodeType(pair_types[i])
                X[c*block_size:(c+1)*block_size] = X[c*block_size:(c+1)*block_size] + G[i]
        return X
    def expandTriplets(self, trip_types, trips):
        X = np.zeros((self.type_basis.NN*self.triplet_dim,))
        if len(trips) > 0:
            assert trips.shape[1] == 4
            G = []
            for i in range(3):
                Ci = self.centres[i+1]
                Si = self.sigmas[i+1]
                Gi = np.exp(
                    -(trips[:,i].reshape((-1,1)) - Ci)**2/(2.*Si**2)
                )
                G.append(Gi)
            GG = []
            for t in range(len(trips)):
                GGt = np.outer(
                    np.outer(G[0][t], G[1][t]).flatten(), G[2][t]
                ).flatten()
                GG.append(GGt)
            GG = np.array(GG)
            W = trips[:,3]
            GG = (GG.T*W).T
            assert GG[0].shape[0] == self.triplet_dim
            block_size = self.triplet_dim
            for i in range(len(trip_types)):
                c = self.type_basis.encodePair(*tuple(trip_types[i]))
                if c is None:
                    log << "Skip" << trip_types[i] << log.endl
                    continue
                X[c*block_size:(c+1)*block_size] = X[c*block_size:(c+1)*block_size] + GG[i]
        return X
    def save(self, jarfile):
        open(jarfile, 'w').write(pickle.dumps(self))
    def load(self, jarfile):
        self = pickle.load(open(jarfile, 'rb'))
        return self

def get_pairs_triplets(rc, T, R, D, basis, eps=1e-10):
    R = R-rc
    dR = np.sum(R**2, axis=1)**0.5
    # Pairs
    pair_types = []
    pairs = []
    for i in range(len(T)):
        pair_types.append(T[i])
        pairs.append([ dR[i], basis.sigma**2/dR[i]**2 if dR[i] > eps else 1. ])
    pairs = np.array(pairs)
    pairs[:,1] = pairs[:,1]*basis.calcWeights(pairs[:,0])
    # Triplets
    triplet_types = []
    triplets = []
    for i in range(len(T)):
        ri = dR[i]
        if ri < eps: continue
        for j in range(i+1, len(T)):
            rj = dR[j]
            if rj < eps: continue
            rij = D[i,j]
            triplet_types.append(sorted([T[i],T[j]]))
            triplets.append([ri+rj, np.abs(ri-rj), rij, basis.sigma**4/(ri*rj)**2, ri, rj])
    triplets = np.array(triplets)
    if len(triplets) > 0:
        triplets[:,3] = triplets[:,3]*basis.calcWeights(triplets[:,4])*basis.calcWeights(triplets[:,5])
        triplets = triplets[:,0:4]
    return pair_types, pairs, triplet_types, triplets

def expand_structure(config, basis, centres=None, log=None):
    T = np.array(config.symbols)
    R = config.positions
    D = soap.tools.partition.calculate_distance_mat(R, R)
    IX_pairs = []
    IX_trips = []
    if centres is None: centres = range(len(T))
    for i in centres:
        nbs = np.where(D[i] <= basis.r_cut)[0]
        rc = R[i]
        pair_types, pairs, triplet_types, triplets = \
            get_pairs_triplets(rc, T[nbs], R[nbs], D[nbs][:,nbs], basis=basis)
        X_pairs = basis.expandPairs(pair_types, pairs)
        if len(pair_types) > 0:
            X_pairs = X_pairs/np.dot(X_pairs, X_pairs)**0.5
        X_trips = basis.expandTriplets(triplet_types, triplets)
        if len(triplet_types) > 0:
            X_trips = X_trips/np.dot(X_trips, X_trips)**0.5
        IX_pairs.append(X_pairs)
        IX_trips.append(X_trips)
    return np.array(IX_pairs), np.array(IX_trips)

