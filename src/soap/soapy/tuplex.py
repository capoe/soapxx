#! /usr/bin/env python
import soap
import numpy as np
import scipy.special
import scipy.linalg
from . import nn 
import copy
import scipy.optimize
import momo
import multiprocessing as mp
log = momo.osio

class PyNodeTupleF(nn.PyNode):
    def __init__(self, idx, parents, props):
        nn.PyNode.__init__(self, idx, parents, props)
        self.op = "tuplex"
        self.basis = props["basis"]
        C = self.basis.type_basis.K_abc.shape[2] # number of pair type channels
        self.dim = (3+6)*self.basis.type_basis.N
        self.dX_dR = None
        self.w_grad = True
        assert len(self.parents) == 2
    def evaluate(self):
        T_mat = self.parents[0].X_out
        R_mat = self.parents[1].X_out
        assert T_mat.shape[0] == R_mat.shape[0]
        self.X_out = np.zeros((T_mat.shape[0], self.dim))
        self.dX_dR = []
        for env in range(T_mat.shape[0]):
            X, dX_dR = evaluate_tuplef(T_mat[env], R_mat[env], 
                basis=self.basis, w_grad=self.w_grad)
            self.X_out[env] = X
            self.dX_dR.append(dX_dR)
    def backpropagate(self, g_back=1., level=0, log=None):
        g_R = []
        for env in range(g_back.shape[0]):
            g = self.dX_dR[env].T.dot(g_back[env])
            g_R.append(g)
        self.parents[1].backpropagate(g_back=g_R, level=level+1, log=log)

class PyNodeTupleX(nn.PyNode):
    def __init__(self, idx, parents, props):
        nn.PyNode.__init__(self, idx, parents, props)
        self.op = "tuplex"
        self.dX_dR = None
        # Basis and dimensions
        self.basis = props["basis"]
        P = len(self.basis.centres[0])           # number of r01 basis fcts
        N = len(self.basis.centres[1])           # number of r1+r2 basis fcts
        K = len(self.basis.centres[2])           # number of r1-r2 basis fcts
        L = len(self.basis.centres[3])           # number of r3 basis fcts
        C = self.basis.type_basis.K_abc.shape[2] # number of pair type channels
        self.dim_pairs = P*self.basis.type_basis.N
        self.dim_pairs_ch = self.basis.type_basis.N
        self.dim_pairs_geom = self.basis.type_basis.N
        self.dim_trips = N*K*L*C
        self.dim_trips_ch = C
        self.dim_trips_geom = N*K*L
        self.dim = self.dim_pairs + self.dim_trips
        # Settings
        self.w_grad = nn.require(props, "w_grad", False)
        self.n_procs = nn.require(props, "n_procs", 1)
        assert len(self.parents) == 2
    def evaluate(self):
        T_mat = self.parents[0].X_out
        R_mat = self.parents[1].X_out
        assert T_mat.shape[0] == R_mat.shape[0]
        self.X_out = np.zeros((T_mat.shape[0], self.dim))
        self.dX_dR = []
        if self.n_procs == 1:
            for env in range(T_mat.shape[0]):
                X, dX_dR = evaluate_tuplex(T_mat[env], R_mat[env], 
                    basis=self.basis, w_grad=self.w_grad)
                self.X_out[env] = X
                self.dX_dR.append(dX_dR)
        else:
            pool = mp.Pool(processes=self.n_procs)
            inputs = [ (T_mat[env], R_mat[env], self.basis, self.w_grad) \
                for env in range(T_mat.shape[0]) ]
            res = pool.map(evalute_tuplex_mp, inputs)
            pool.close()
            for env in range(T_mat.shape[0]):
                self.X_out[env] = res[env][0]
                self.dX_dR.append(res[env][1])
    def backpropagate(self, g_back=1., level=0, log=None):
        if self.w_grad == False: return
        g_R = []
        for env in range(g_back.shape[0]):
            g = self.dX_dR[env].T.dot(g_back[env])
            g_R.append(g)
        self.parents[1].backpropagate(g_back=g_R, level=level+1, log=log)

def evaluate_distance(R, eps=1e-10, w_grad=True):
    dR = np.sqrt(np.sum(R*R, axis=1))
    grad_dR = (R.T/(dR+eps)).T if w_grad else None
    return dR, grad_dR

def evaluate_weight(dR, r_erf, eps=1e-4, w_grad=True, constant=False):
    if constant: return np.ones_like(dR), np.zeros_like(dR)
    t = 1. - np.exp(-dR**2/r_erf**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        w = t*(r_erf+eps)**2/(dR)**2
        near_zero = np.where(dR < eps)
        w[near_zero] = 1.0
        if w_grad:
            grad_w = 2*((1.-t)-w)/(dR)
            grad_w[near_zero] = 0.0
        else: 
            grad_w = None
    return w, grad_w

def evaluate_cutoff(dR, r_cut, r_cut_width, w_grad=True):
    w = np.heaviside(-dR+r_cut, 0.0)
    transition_idcs = np.where(w*dR > r_cut-r_cut_width)[0]
    phase = (dR[transition_idcs]-r_cut+r_cut_width)/(r_cut_width)*np.pi
    w[transition_idcs] = 0.5*(1.+np.cos(phase))
    if w_grad:
        grad_w = np.zeros_like(w)
        grad_w[transition_idcs] = -0.5*np.sin(phase)*np.pi/r_cut_width
    else: grad_w = None
    return w, grad_w

def expand_pairs(dR, basis, w_grad=True):
    rr = np.subtract.outer(basis.centres[0], dR)
    g = np.exp(-0.5*rr**2/basis.sigmas[0]**2)
    grad_g_dR = rr/basis.sigmas[0]**2*g if w_grad else None
    return g, grad_g_dR

def expand_triplets(S12p, S12m, S3, basis, w_grad=True):
    ss1 = np.subtract.outer(basis.centres[1], S12p)
    ss2 = np.subtract.outer(basis.centres[2], S12m)
    ss3 = np.subtract.outer(basis.centres[3], S3)
    g1 = np.exp(-0.5*ss1**2/basis.sigmas[1]**2)
    g2 = np.exp(-0.5*ss2**2/basis.sigmas[2]**2)
    g3 = np.exp(-0.5*ss3**2/basis.sigmas[3]**2)
    if w_grad:
        grad_g1 = ss1/basis.sigmas[1]**2*g1
        grad_g2 = ss2/basis.sigmas[2]**2*g2
        grad_g3 = ss3/basis.sigmas[3]**2*g3
    else:
        grad_g1, grad_g2, grad_g3 = (None, None, None)
    return g1, g2, g3, grad_g1, grad_g2, grad_g3

def grad_correct_for_centre(g):
    g[:,0] = -np.sum(g[:,3::3], axis=1)
    g[:,1] = -np.sum(g[:,4::3], axis=1)
    g[:,2] = -np.sum(g[:,5::3], axis=1)
    return g

def evaluate_tuplef(T_nbs, R_nbs, basis, 
        normalize=True, 
        w_grad=True):
    dim_x_in = R_nbs.shape[0]
    T_nbs = T_nbs.reshape((-1,basis.type_basis.N))
    R_nbs = R_nbs.reshape((-1,3))
    grad_R_nbs_xyz = np.identity(dim_x_in).reshape((-1,3,dim_x_in))
    assert T_nbs.shape[0] == R_nbs.shape[0]
    dR, grad_dR_xyz = evaluate_distance(R_nbs, w_grad=w_grad)
    # Cutoff and weight function
    fc, grad_fc_dR = evaluate_cutoff(dR, basis.r_cut, basis.r_cut_width, w_grad=w_grad)
    w, grad_w_dR = evaluate_weight(dR, basis.weight_decay_length, 
        w_grad=w_grad, constant=basis.weight_constant)
    wf = fc*w
    wf = wf.flatten()
    if w_grad:
        grad_wf_dR = fc*grad_w_dR + w*grad_fc_dR
        grad_wf_xyz = (grad_dR_xyz.T*grad_wf_dR).T
        grad_wf_xyz = scipy.linalg.block_diag(*tuple(list(grad_wf_xyz)))
        grad_dR_xyz = scipy.linalg.block_diag(*tuple(list(grad_dR_xyz)))
    # Frame
    M = np.einsum('i,it,ia->at', wf, T_nbs, R_nbs)
    M = M.flatten()
    Q = np.einsum('i,it,ia,ib->abt', wf, T_nbs, R_nbs, R_nbs)
    Q = Q[([0,0,0,1,1,2],[0,1,2,1,2,2])] 
    Q = Q.flatten()
    if w_grad:
        grad_M = (
            np.einsum('id,it,ia->atd', grad_wf_xyz, T_nbs, R_nbs)
          + np.einsum('i,it,iad->atd', wf, T_nbs, grad_R_nbs_xyz)
        )
        grad_M = grad_M.reshape((-1,dim_x_in))
        grad_M = grad_correct_for_centre(grad_M)
        grad_Q = (
            np.einsum('id,it,ia,ib->abtd', grad_wf_xyz, T_nbs, R_nbs, R_nbs)
          + np.einsum('i,it,iad,ib->abtd', wf, T_nbs, grad_R_nbs_xyz, R_nbs)
          + np.einsum('i,it,ia,ibd->abtd', wf, T_nbs, R_nbs, grad_R_nbs_xyz)
        )
        grad_Q = grad_Q[([0,0,0,1,1,2],[0,1,2,1,2,2])] 
        grad_Q = grad_Q.reshape((-1,dim_x_in))
        grad_Q = grad_correct_for_centre(grad_Q)
    else:
        grad_M = None
        grad_Q = None
    Q = np.concatenate([M,Q])
    if w_grad:
        grad_Q = np.concatenate([grad_M, grad_Q], axis=0)
    if normalize:
        Q, grad_Q = norm_w_grad(Q, grad_Q)
    return Q, grad_Q

def evalute_tuplex_mp(inputs):
    X, dX = evaluate_tuplex(
        T_nbs=inputs[0], R_nbs=inputs[1], 
        basis=inputs[2], w_grad=inputs[3])
    return (X, dX)

def evaluate_tuplex(T_nbs, R_nbs, basis, 
        normalize=True, 
        w_grad=True):
    dim_x_in = R_nbs.shape[0]
    T_nbs = T_nbs.reshape((-1,basis.type_basis.N))
    R_nbs = R_nbs.reshape((-1,3))
    assert T_nbs.shape[0] == R_nbs.shape[0]
    dR, grad_dR_xyz = evaluate_distance(R_nbs, w_grad=w_grad)

    # Cutoff and weight function
    fc, grad_fc_dR = evaluate_cutoff(dR, basis.r_cut, basis.r_cut_width, w_grad=w_grad)
    w, grad_w_dR = evaluate_weight(dR, basis.weight_decay_length, 
        w_grad=w_grad, constant=basis.weight_constant)
    wf = fc*w
    wf = wf.flatten()
    if w_grad:
        grad_wf_dR = fc*grad_w_dR + w*grad_fc_dR
        grad_wf_xyz = (grad_dR_xyz.T*grad_wf_dR).T
        grad_wf_xyz = scipy.linalg.block_diag(*tuple(list(grad_wf_xyz)))
        grad_dR_xyz = scipy.linalg.block_diag(*tuple(list(grad_dR_xyz)))

    # Pair expansion
    g, grad_g_dR = expand_pairs(dR, basis, w_grad=w_grad)
    P = (np.einsum('ni,i,ia->na', g, wf, T_nbs)).reshape((g.shape[0]*T_nbs.shape[1],))
    if w_grad:
        grad_P_xyz = (
            np.einsum('ni,i,ia,id->nad', grad_g_dR, wf, T_nbs, grad_dR_xyz) \
          + np.einsum('ni,ia,id->nad',   g,             T_nbs, grad_wf_xyz)
        ).reshape((g.shape[0]*T_nbs.shape[1], dim_x_in))
        grad_P_xyz = grad_correct_for_centre(grad_P_xyz)
    else: grad_P_xyz = None

    # Triplet expansion
    U_nbs = T_nbs[1:] # types
    S_nbs = R_nbs[1:]
    dS = dR[1:]
    vf = wf[1:]
    if w_grad:
        grad_dS_xyz = grad_dR_xyz[1:]
        grad_vf_xyz = grad_wf_xyz[1:]
    # Triplet types
    T12 = np.einsum('it,js->ijts', U_nbs, U_nbs)
    # Triplet weights
    V12 = np.multiply.outer(vf, vf).flatten()
    V1 = np.multiply.outer(vf, np.ones_like(vf)).flatten()
    V2 = np.multiply.outer(np.ones_like(vf), vf).flatten()
    if w_grad:
        gv1 = np.tile(grad_vf_xyz, (1,len(grad_vf_xyz))).reshape((-1,grad_vf_xyz.shape[1]))
        gv2 = np.tile(grad_vf_xyz, (len(grad_vf_xyz),1))
        grad_V12_xyz = ((gv1.T*V2 + gv2.T*V1)).T
    # Triplet (r01+r02)
    S12p = np.add.outer(dS, dS).flatten()
    if w_grad:
        grad_S12p_xyz = np.tile(grad_dS_xyz, (len(grad_dS_xyz),1)) + \
            np.tile(grad_dS_xyz, (1,len(grad_dS_xyz))).reshape((-1,grad_dS_xyz.shape[1]))
    # Triplet |r01-r02|
    S12m = np.subtract.outer(dS, dS).flatten()
    S12m_sign = np.sign(S12m)
    S12m = S12m*S12m_sign
    if w_grad:
        gs1 = np.tile(grad_dS_xyz, (1,len(grad_dS_xyz))).reshape((-1,grad_dS_xyz.shape[1]))
        gs2 = np.tile(grad_dS_xyz, (len(grad_dS_xyz),1))
        grad_S12m_xyz = ((gs1.T - gs2.T)*S12m_sign).T

    # Triplet r3
    g_in = np.concatenate([np.zeros((len(S_nbs),1)), np.identity(len(S_nbs))], axis=1)
    X12 = np.subtract.outer(S_nbs[:,0], S_nbs[:,0]).flatten()
    Y12 = np.subtract.outer(S_nbs[:,1], S_nbs[:,1]).flatten()
    Z12 = np.subtract.outer(S_nbs[:,2], S_nbs[:,2]).flatten()
    S3 = np.sqrt(X12**2+Y12**2+Z12**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        gs_X_pre = X12/S3
        gs_Y_pre = Y12/S3
        gs_Z_pre = Z12/S3
    g_in_1 = np.tile(g_in, (1,len(S_nbs))).reshape((-1,g_in.shape[1]))
    g_in_2 = np.tile(g_in, (len(S_nbs),1))
    g_in_12 = g_in_1.T - g_in_2.T
    if w_grad:
        grad_X = (g_in_12*gs_X_pre).T
        grad_Y = (g_in_12*gs_Y_pre).T
        grad_Z = (g_in_12*gs_Z_pre).T
        grad_S3_xyz = np.concatenate([ grad_X, grad_Y, grad_Z ], axis=1)
        order = np.concatenate([ 
            [ i, grad_X.shape[1]+i, 2*grad_X.shape[1]+i ] for i in range(grad_X.shape[1])
        ])
        grad_S3_xyz = grad_S3_xyz[:,order]
    # Exclude diagonal
    triu_i, triu_j = np.triu_indices(len(dS), k=1)
    triu_ij = triu_i*len(dS)+triu_j
    T12 = T12[(triu_i,triu_j)]
    V12 = V12[triu_ij]
    S12p = S12p[triu_ij]
    S12m = S12m[triu_ij]
    S3 = S3[triu_ij]
    if w_grad:
        grad_V12_xyz = grad_V12_xyz[triu_ij]
        grad_S12p_xyz = grad_S12p_xyz[triu_ij]
        grad_S12m_xyz = grad_S12m_xyz[triu_ij]
        grad_S3_xyz = grad_S3_xyz[triu_ij]
    # Triplet type and structure expansion
    C12 = np.einsum('mab,abc->mc', T12, basis.type_basis.K_abc)
    g1, g2, g3, grad_g1, grad_g2, grad_g3 = expand_triplets(
        S12p, S12m, S3, basis, w_grad=w_grad)
    N = g1.shape[0]
    K = g2.shape[0]
    L = g3.shape[0]
    M = g1.shape[1]
    D = dim_x_in
    C = C12.shape[1]
    # Contractions
    GHI = (\
        np.einsum('nm,km,lm,m,mc->nklc', g1, g2, g3, V12, C12, optimize='greedy') \
    ).reshape((N*K*L*C,))
    if w_grad:
        grad_GHI_xyz = (\
            np.einsum('nm,km,lm,m,mc,md->nklcd', 
                grad_g1, g2,      g3,      V12, C12, grad_S12p_xyz, optimize='greedy') \
          + np.einsum('nm,km,lm,m,mc,md->nklcd', 
                g1,      grad_g2, g3,      V12, C12, grad_S12m_xyz, optimize='greedy') \
          + np.einsum('nm,km,lm,m,mc,md->nklcd', 
                g1,      g2,      grad_g3, V12, C12, grad_S3_xyz,   optimize='greedy') \
          + np.einsum('nm,km,lm,mc,md->nklcd',   
                g1,      g2,      g3,           C12, grad_V12_xyz,  optimize='greedy') \
        ).reshape((N*K*L*C, dim_x_in))
        grad_GHI_xyz = grad_correct_for_centre(grad_GHI_xyz)
    else: grad_GHI_xyz = None

    # Normalize
    if normalize:
        P, grad_P_xyz = norm_w_grad(P, grad_P_xyz)
        GHI, grad_GHI_xyz = norm_w_grad(GHI, grad_GHI_xyz)

    # Concatenate
    X = np.concatenate([ P, GHI ])
    if w_grad:
        grad_X = np.concatenate([ 
            grad_P_xyz, grad_GHI_xyz ], axis=0)
    else: grad_X = None
    return X, grad_X

def norm_w_grad(X, grad_X):
    z = 1./np.sqrt(np.sum(X**2))
    Xz = X*z
    if grad_X is not None:
        grad_Xz = grad_X*z
        grad_Xz = grad_Xz - np.outer(Xz, grad_Xz.T.dot(Xz))
    else: grad_Xz = None
    return Xz, grad_Xz

def extract_environments(config, basis):
    R = config.positions
    T = np.array(config.symbols)
    R_out = []
    T_out = []
    if "centres" in config.info:
        centres = map(int, config.info["centres"].split(','))
    else:
        centres = np.where(np.array(config.symbols) != 'H')[0]
    D = soap.tools.partition.calculate_distance_mat(R[centres], R)
    L = np.heaviside(basis.r_cut - D, 0.0)
    L[(np.arange(L.shape[0]), centres)] = 0
    for cc, c in enumerate(centres):
        nbs = np.where(L[cc] > 0.5)
        R_nbs = R[nbs]
        R_nbs = R_nbs - R[c]
        Rc = np.concatenate([ np.zeros((3,)), R_nbs.flatten() ])
        Tc = np.concatenate([ T[c:c+1], T[nbs] ])
        tids = ( np.arange(len(Tc)), [ basis.type_basis.encodeType(t) for t in Tc ] )
        Tc = np.zeros((len(Tc), basis.type_basis.N))
        Tc[tids] = 1.
        Tc = Tc.flatten()
        R_out.append(Rc)
        T_out.append(Tc)
    return np.array(T_out), np.array(R_out)

def evaluate_distance_mat_3d_triu(x, w_grad):
    dim = len(x)
    n_parts = len(x)//3
    X = x[0::3]
    Y = x[1::3]
    Z = x[2::3]
    XX = np.subtract.outer(X,X)
    YY = np.subtract.outer(Y,Y)
    ZZ = np.subtract.outer(Z,Z)
    D = np.sqrt(XX**2 + YY**2 + ZZ**2)
    rows, cols = np.triu_indices(D.shape[0], k=1)
    ij = rows*dim + cols
    D = D[(rows,cols)]
    if w_grad:
        XX = XX[(rows,cols)]
        YY = YY[(rows,cols)]
        ZZ = ZZ[(rows,cols)]
        gX = scipy.linalg.block_diag(*tuple([ np.array([1.,0.,0.]) for i in range(n_parts)]))
        gY = scipy.linalg.block_diag(*tuple([ np.array([0.,1.,0.]) for i in range(n_parts)]))
        gZ = scipy.linalg.block_diag(*tuple([ np.array([0.,0.,1.]) for i in range(n_parts)]))
        gXX = np.tile(gX, (1,dim)).reshape((-1,dim)) - np.tile(gX, (dim,1))
        gYY = np.tile(gY, (1,dim)).reshape((-1,dim)) - np.tile(gY, (dim,1))
        gZZ = np.tile(gZ, (1,dim)).reshape((-1,dim)) - np.tile(gZ, (dim,1))
        gXX = gXX[ij]
        gYY = gYY[ij]
        gZZ = gZZ[ij]
        grad_D = ((gXX.T*XX + gYY.T*YY + gZZ.T*ZZ)/D).T
    else: grad_D = None
    return D, grad_D

nn.register("tuplex", PyNodeTupleX)
nn.register("tuplef", PyNodeTupleF)

