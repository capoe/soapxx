import numpy as np
import numpy.linalg as la
import kernel as kern
import soap
import multiprocessing as mp

import momo
import datasets.gdb
import datasets.soap
import libmatch.environments # Alchemy
import libmatch.structures # structure

LAGRAPH_DEBUG = False
LAGRAPH_CHECK = True

# TODO Properly construct subgraph Laplacians

class ParticleGraphVertex(object):
    def __init__(self, atom, options):
        self.atom = atom
        self.p = None
        
def sorted_eigenvalues_vectors(matrix, hermitian=False):
    # i-th column(!) of v is eigenvector to i-th eigenvalue in w
    if hermitian:
        w,V = la.eigh(matrix)
    else:
        w,V = la.eig(matrix)
    w,V = la.eig(matrix)
    order = w.argsort()
    w = w[order]
    V = V[:,order]
    return w,V

def flg_compute_S(L, Q, eta, gamma):
    L_reg = L+eta*np.identity(L.shape[0])
    L_reg_inv = la.inv(L_reg)
    S_reg = Q.T.dot(L_reg_inv).dot(Q).real
    travg_S = np.trace(S_reg)/S_reg.shape[0]
    S_reg = S_reg + gamma*np.identity(Q.shape[1])

    #DEBUG
    if LAGRAPH_DEBUG:
        print ""
        print "<flg_compute_S>"
        print "L_reg=", L_reg
        print "L_reg_inv=", L_reg_inv
        print "Identity?=", L_reg.dot(L_reg_inv)
        print "S_reg", S_reg
        raw_input('...')
    #DEBUG
    return S_reg, travg_S

def flg_compute_k_flg(S1, S2):
    S1_inv = la.inv(S1)
    S2_inv = la.inv(S2)        
    S12_inv = 0.5*(S1_inv+S2_inv)
    S12_inv_inv = la.inv(S12_inv)
    det_12 = la.det(S12_inv_inv)
    det_1_2 = la.det(S1)*la.det(S2)    
    warn = False
    if det_12 < 0. or det_1_2 < 0.:
        print "WARNING__%+1.7f<0__%+1.7f<0__" % (det_12, det_1_2)
        warn = True
    #return (la.det(S12_inv_inv))**0.5/(la.det(S1)*la.det(S2))**0.25 
    return abs(det_12)**0.5/abs(det_1_2)**0.25, warn

def adjust_regularization(graphs, options):
    print "Adjust eta, gamma based on %d graphs." % len(graphs)
    # Adjust L-regularization
    traces = []
    for g in graphs:
        travg_L = np.trace(g.L)/g.L.shape[0]
        traces.append(travg_L)
    traces = np.array(traces)
    avg = np.average(traces)
    options['laplacian']['eta'] = avg/100.
    print "Adjust eta to", options['laplacian']['eta']
    # Adjust S-regularization
    traces = []
    for g in graphs:
        print g.label
        if options['graph']['hierarchical']:
            k, travg_S1, travg_S2 = compare_graphs_hierarchical(g, g, options, return_traces=True)
        else:
            k, travg_S1, travg_S2 = compare_graphs_featurized(g, g, options, return_traces=True)
        traces.append(travg_S1)
    traces = np.array(traces)
    avg = np.average(traces)    
    options['laplacian']['gamma'] = avg/100.
    print "Adjust gamma to", options['laplacian']['regularize_gamma']    
    return

def optimize_hierarchy(graphs, options, kfct, write_out=False, log=None, verbose=False):
    if not options['graph']['optimize_hierarchy']: return
    if log: log << "Optimizing r0, n_levels based on %d graphs" % len(graphs) << log.endl
    # ETA-GAMMA PAIRS
    r0s = [ 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.5, 2. ]
    levels = [ 1,2,3,4,5 ]
    pairs = []
    for r0 in r0s:
        for l in levels:
            pairs.append((r0,l))
        pairs.append((None,None))
    if write_out: ofs = open('out.optimize_resolution.txt', 'w')
    # COMPUTE MERIT (VIA STD-DEV) FOR EACH PAIR
    merits = []
    for r0, l in pairs:
        if r0 == l == None:
            if write_out: ofs.write('\n')
            continue
        # Set
        options['graph']['r0'] = r0
        options['graph']['n_levels'] = l
        for graph in graphs:
            del graph.subgraphs
            graph.createSubgraphs(options)
        # Process
        kmat = soap.soapy.util.mp_compute_upper_triangle(
            kfct=kfct,
            g_list=graphs,
            n_procs=4,
            n_blocks=1,
            log=None,
            tstart_twall=(None, None),
            backup=False,
            options=options)
        # Analyse
        kmat = kmat + kmat.T
        np.fill_diagonal(kmat, 0.5*kmat.diagonal())
        triu_idcs = np.triu_indices(kmat.shape[0], 1)
        kmat_triu = kmat[triu_idcs]
        kmat_min = np.min(kmat_triu)
        kmat_max = np.max(kmat_triu)
        kmat_avg = np.average(kmat_triu)
        kmat_std = np.std(kmat_triu)
        kmat_med = np.median(kmat_triu)
        #kmat_ent = -np.sum(kmat_triu*np.log(kmat_triu+1e-20))
        kmat_ent = soap.soapy.math.shannon_entropy(kmat_triu, eps=1e-20, norm=True)
        if log: log << 'r0=%+1.2f n_lev=%d avg/std %+1.2e %+1.2e min/max %+1.2e %+1.2e ent %+1.2e' % \
            (r0, l, kmat_avg, kmat_std, kmat_min, kmat_max, kmat_ent) << log.endl
        if log and verbose: log << kmat << log.endl
        if write_out:
            ofs.write('%+1.2f %d avg/std %+1.7e %+1.7e min/max %+1.7e %+1.7e ent %+1.2e\n' % \
                (r0, l, kmat_avg, kmat_std, kmat_min, kmat_max, kmat_ent))
        # Store
        merits.append((r0, l, kmat_std, kmat_ent, kmat_med))
    ent_target = 0.25/(-0.5*np.log(0.5)) # <- entropy of numbers uniformly distributed over [0,1]
    std_target = (1./12.)**0.5 # <- std deviation of numbers uniformly distributed over [0,1]
    med_target = 0.5
    merits = sorted(merits, key=lambda m: -(m[2]-std_target)**2 -(m[3]-ent_target)**2 -(m[4]-med_target)**2)
    if log: log << "Optimum for r0=%+1.7e n_lev=%d : std=%+1.4e ent=%+1.4e med=%+1.4e" % merits[-1] << log.endl
    options['graph']['r0'] = merits[-1][0]
    options['graph']['n_levels'] = merits[-1][1]
    return
    

def optimize_regularization(graphs, options, write_out=False, log=None, verbose=False):
    if not options['laplacian']['optimize_eta_gamma']: return
    if log: log << "Optimizing eta, gamma based on %d graphs" % len(graphs) << log.endl
    if verbose:
        for graph in graphs: 
            print graph.label
            print graph.L
    # ETA-GAMMA PAIRS
    etas   = [ 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1., 5., 10., 50., 100., 500. ]
    gammas = [ 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1., 5., 10., 50., 100., 500. ]
    etas   = [ 10**(-7+i) for i in range(14) ]
    gammas = [ 10**(-7+i) for i in range(14) ]
    etas   = [ 10**(-3+i) for i in range(5) ]
    gammas = [ 10**(-3+i) for i in range(5) ]
    pairs = []
    for eta in etas:
        for gamma in gammas:
            pairs.append((eta,gamma))
        pairs.append((None,None))
    if write_out: ofs = open('out.optimize_regularization.txt', 'w')
    # COMPUTE MERIT (VIA STD-DEV) FOR EACH PAIR
    merits = []
    for eta, gamma in pairs:
        if eta == gamma == None:
            if write_out: ofs.write('\n')
            continue
        # Set
        options['laplacian']['eta'] = eta
        options['laplacian']['gamma'] = gamma
        # Process
        kmat = soap.soapy.util.mp_compute_upper_triangle(
            kfct=compare_graphs_laplacian_kernelized,
            g_list=graphs,
            n_procs=4,
            n_blocks=1,
            log=None,
            tstart_twall=(None, None),
            backup=False,
            options=options)
        # Analyse
        kmat = kmat + kmat.T
        np.fill_diagonal(kmat, 0.5*kmat.diagonal())
        triu_idcs = np.triu_indices(kmat.shape[0], 1)
        kmat_triu = kmat[triu_idcs]
        kmat_min = np.min(kmat_triu)
        kmat_max = np.max(kmat_triu)
        kmat_avg = np.average(kmat_triu)
        kmat_std = np.std(kmat_triu)
        kmat_med = np.median(kmat_triu)
        #kmat_ent = -np.sum(kmat_triu*np.log(kmat_triu+1e-20))
        kmat_ent = soap.soapy.math.shannon_entropy(kmat_triu, eps=1e-20, norm=True)
        if log: log << 'Eta=%+1.2e Gamma=%+1.2e avg/std %+1.2e %+1.2e min/max %+1.2e %+1.2e ent %+1.2e' % \
            (eta, gamma, kmat_avg, kmat_std, kmat_min, kmat_max, kmat_ent) << log.endl
        if log and verbose: log << kmat << log.endl
        if write_out:
            ofs.write('%+1.7e %+1.7e avg/std %+1.7e %+1.7e min/max %+1.7e %+1.7e ent %+1.2e\n' % \
                (eta, gamma, kmat_avg, kmat_std, kmat_min, kmat_max, kmat_ent))
        # Store
        merits.append((eta, gamma, kmat_std, kmat_ent, kmat_med))
    ent_target = 0.25/(-0.5*np.log(0.5)) # <- entropy of numbers uniformly distributed over [0,1]
    std_target = (1./12.)**0.5 # <- std deviation of numbers uniformly distributed over [0,1]
    med_target = 0.5
    #ent_target = 1. # TODO
    merits = sorted(merits, key=lambda m: -(m[2]-std_target)**2 -(m[3]-ent_target)**2 -(m[4]-med_target)**2)
    #merits = sorted(merits, key=lambda m: m[3])
    if log: log << "Optimum for eta=%+1.7e gamma=%+1.7e : std=%+1.4e ent=%+1.4e med=%+1.4e" % merits[-1] << log.endl
    options['laplacian']['eta'] = merits[-1][0]
    options['laplacian']['gamma'] = merits[-1][1]
    return

def compare_graphs_laplacian_kernelized(g1, g2, options):
    if options['run']['verbose']:
        print "flg(%s,%s)" % (g1.label, g2.label)
    if options['graph']['hierarchical']:
        return compare_graphs_hierarchical(g1, g2, options)
    else:
        return compare_graphs_featurized(g1, g2, options)

def compare_graphs_kernelized(L1, L2, K12, options, zero_eps=1e-10, verbose=False):

    if LAGRAPH_CHECK:   
        assert np.max(np.abs(np.sum(L1, axis=1))) < zero_eps
        assert np.max(np.abs(np.sum(L2, axis=1))) < zero_eps
    assert L1.shape[0] + L2.shape[0] == K12.shape[0]
    assert L1.shape[1] + L2.shape[1] == K12.shape[1]
    assert K12.shape[0] == K12.shape[1]
    
    # JOINT-SUB-SPACE CALCULATION
    # Non-zero-eigenvalue eigenspace
    assert np.max(np.abs(K12 - K12.T)) < zero_eps # TODO Debug mode
    eigvals, eigvecs = sorted_eigenvalues_vectors(K12, hermitian=True)    
    
    if eigvals[0] < -zero_eps:        
        e_min = np.min(eigvals)
        e_max = np.max(eigvals)
        if abs(e_min/e_max) > 0.01: #zero_eps**0.5:
            #print K12
            print "WARNING Eigvals < 0 (%+1.7e) (%+1.7e) (%+1.7e)" % (e_min, e_max, e_min/e_max)
    
    sel_idx = []
    for i in range(K12.shape[0]):
        #if abs(eigvals[i]) < zero_eps: pass # TODO Precision problems here ... even positive-definite matrices produce negative eigenvalues !?
        if eigvals[i] < zero_eps: pass
        else: sel_idx.append(i)
    eigvals_nonzero = eigvals[sel_idx]
    eigvecs_nonzero = eigvecs[:,sel_idx]    
    
    # Form subspace projection matrix Q
    Q = eigvals_nonzero**0.5*eigvecs_nonzero
    Q1 = Q[0:L1.shape[0],:]
    Q2 = Q[L1.shape[0]:K12.shape[0],:]
    
    # COMPUTE SPECTRUM OVERLAP   
    # Inverse regularized laplacian
    eta = options['laplacian']['eta']
    gamma = options['laplacian']['gamma']
    S1_reg, travg_S1 = flg_compute_S(L1, Q1, eta, gamma)
    S2_reg, travg_S2 = flg_compute_S(L2, Q2, eta, gamma)
    
    #DEBUG
    if LAGRAPH_DEBUG:
        print ""
        print "<compare_graphs_kernelized>"
        print "eigvals=", eigvals
        print "eigvecs=", eigvecs
        print "Q1=", Q1
        print "Q2=", Q2
        print "L1=", L1
        print "L2=", L2
        print "S1_reg=", S1_reg
        print "S2_reg=", S2_reg
        raw_input('...')
    #DEBUG
    
    """
    if verbose:
        print "K12", K12[0:8][:,0:8]
        print "K12", K12[8:16][:,8:16]
        print "dK12", K12 - K12.T
        print "Q1", Q1
        print "Q2", Q2
        print "S1", S1_reg
        print "S2", S2_reg
        print la.det(S1_reg)
        print la.det(S2_reg)
        #print S1_reg - S2_reg
    """
    
    # Laplacian-graph kernel
    k_flg, warn = flg_compute_k_flg(S1_reg, S2_reg)
    return k_flg, warn, travg_S1, travg_S2
    
def compare_graphs_featurized(g1, g2, options, return_traces=False):

    # EXTRACT LAPLACIANS (L), FEATURE MATRICES (P), SHAPES
    L1 = g1.L
    L2 = g2.L
    P1 = g1.P
    P2 = g2.P
    n1 = L1.shape[0]
    n2 = L2.shape[0]
    dim1 = P1.shape[1]    
    dim2 = P2.shape[1]
       
    # SANITY CHECKS
    assert P1.shape[0] == n1
    assert P2.shape[0] == n2    
    assert L1.shape[1] == n1
    assert L2.shape[1] == n2    
    assert dim1 == dim2
    n12 = n1+n2
    dim = dim1
    
    # JOINT FEATURE MATRIX => KERNEL MATRIX
    P12 = np.zeros((n12, dim))
    P12[0:n1,:] = P1
    P12[n1:n12,:] = P2
    K12 = P12.dot(P12.T)

    #DEBUG
    if LAGRAPH_DEBUG:
        print ""
        print "<compare_graphs_featurized>"
        print "P1=", P1
        print "P2=", P2
        print "L1=", L1
        print "L2=", L2
        print "K12=", K12
        raw_input('...')
    #DEBUG
    
    # KERNELIZED COMPARISON
    k_flg, warn, trace_S1, trace_S2 = compare_graphs_kernelized(L1, L2, K12, options)
    if warn: print "WARNED __%s:%s__" % (g1.label, g2.label)
    if return_traces:
        return k_flg, trace_S1, trace_S2
    else:
        return k_flg

def compare_graphs_hierarchical(g1, g2, options, return_traces=False):

    # SANITY CHECKS
    assert g1.n_levels == g2.n_levels
    n_levels = g1.n_levels
    
    n1 = g1.L.shape[0]
    n2 = g2.L.shape[0]
    n12 = n1+n2
    K12 = np.zeros((n12,n12), dtype='float64')
    K12_l_out = np.copy(K12)
    
    for l in range(n_levels):
        subgraphs_l = g1.subgraphs[l] + g2.subgraphs[l]
        assert len(subgraphs_l) == n12
        # 0th level => base-kernel-based comparison
        if l == 0:
            for i in range(n12):
                gsub_i = subgraphs_l[i]
                for j in range(i, n12):
                    gsub_j = subgraphs_l[j]
                    
                    """
                    # THIS DOES NOT WORK: NO GRAM MATRIX ANY MORE!
                    assert gsub_i.cutoff == gsub_j.cutoff
                    if i < n1 and j < n1 and g1.D[i][j] > 2*gsub_i.cutoff:
                        print i,j, "no overlap (> 1-1)"
                        k_flg = 0.
                    elif i >= n1 and j >= n1 and g2.D[i-n1][j-n1] > 2*gsub_i.cutoff:
                        print i,j, "no overlap (> 2-2)"
                        k_flg = 0.
                    #print "l=%d: i = %2d size = %2d | j = %2d size = %2d" % (
                    #    l, i, gsub_i.L.shape[0], j, gsub_j.L.shape[0])
                    else:
                        k_flg = compare_graphs_featurized(gsub_i, gsub_j, options)
                    """
                    
                    k_flg = compare_graphs_featurized(gsub_i, gsub_j, options)                    
                    K12[i,j] = k_flg
                    K12[j,i] = K12[i,j]
            # Update child kernel
            K12_l_out = np.copy(K12)
        # l-th level => child-kernel-based comparison
        else:
            for i in range(n12):
                gsub_i = subgraphs_l[i]
                idcs_sub_i = gsub_i.idcs
                for j in range(i, n12):
                    gsub_j = subgraphs_l[j]
                    
                    """
                    # THIS DOES NOT WORK: NO GRAM MATRIX ANY MORE!
                    assert gsub_i.cutoff == gsub_j.cutoff
                    if i < n1 and j < n1 and g1.D[i][j] > 2*gsub_i.cutoff:
                        print i,j, "no overlap (> 1-1)"
                        k_flg = 0.
                    elif i >= n1 and j >= n1 and g2.D[i-n1][j-n1] > 2*gsub_i.cutoff:
                        print i,j, "no overlap (> 2-2)"
                        k_flg = 0.
                    else:
                        idcs_sub_j = gsub_j.idcs
                        #print "l=%d: i = %2d size = %2d | j = %2d size = %2d" % (
                        #    l, i, gsub_i.L.shape[0], j, gsub_j.L.shape[0])
                        # Short-list K12_l_out (= create appropriate view)                    
                        idcs = np.zeros((gsub_i.size+gsub_j.size), dtype='int')
                        idcs[0:gsub_i.size] = idcs_sub_i if i < n1 else n1+idcs_sub_i
                        idcs[gsub_i.size:]  = idcs_sub_j if j < n1 else n1+idcs_sub_j
                        
                        K12_sub = K12_l_out[idcs][:,idcs] # <- TODO This is slow
                        #K12_sub = K12_l_out[np.ix_(idcs,idcs)] # <- TODO Also slow
                        # Compute kernel
                        k_flg, warn = compare_graphs_kernelized(gsub_i.L, gsub_j.L, K12_sub, options)
                        print i,j, idcs_sub_i, idcs_sub_j, idcs, k_flg
                    """

                    idcs_sub_j = gsub_j.idcs
                    #print "l=%d: i = %2d size = %2d | j = %2d size = %2d" % (
                    #    l, i, gsub_i.L.shape[0], j, gsub_j.L.shape[0])
                    # Short-list K12_l_out (= create appropriate view)                    
                    idcs = np.zeros((gsub_i.size+gsub_j.size), dtype='int')
                    idcs[0:gsub_i.size] = idcs_sub_i if i < n1 else n1+idcs_sub_i
                    idcs[gsub_i.size:]  = idcs_sub_j if j < n1 else n1+idcs_sub_j
                    K12_sub = K12_l_out[idcs][:,idcs] # <- TODO This is slow
                    #K12_sub = K12_l_out[np.ix_(idcs,idcs)] # <- TODO Also slow
                    # Compute kernel
                    k_flg, warn, trace_S1, trace_S2 = compare_graphs_kernelized(gsub_i.L, gsub_j.L, K12_sub, options)
                    
                    K12[i,j] = k_flg
                    K12[j,i] = K12[i,j]
            # Update child kernel
            K12_l_out = np.copy(K12)
        """
        print "Level l = %d complete" % l
        print K12_l_out[0:8][:,0:8]
        print K12_l_out[8:16][:,8:16]
        print K12_l_out[0:8][:,8:16]
        raw_input('...')
        """
        
    #print g1.label, g2.label
    k_flg_top, warn, trace_S1, trace_S2 = compare_graphs_kernelized(g1.L, g2.L, K12_l_out, options, verbose=True)
    
    if return_traces:
        return k_flg_top, trace_S1, trace_S2
    else:
        return k_flg_top

class ParticleSubgraph(object):
    def __init__(self, idx, parent, idcs, position, cutoff):
        self.idx = idx
        self.parent = parent
        self.idcs = idcs
        self.position = position
        self.cutoff = cutoff
        self.size = len(self.idcs)
        self.z = self.parent.Z[self.idx]
        return
    @property
    def label(self):
        return self.parent.label
    @property
    def L(self):
        L = self.parent.L[self.idcs][:,self.idcs]
        # Convert L into proper Laplacian
        np.fill_diagonal(L, 0.)
        np.fill_diagonal(L, -np.sum(L, axis=1))
        return L
    @property
    def D(self):
        return self.parent.D[self.idcs][:,self.idcs]
    @property
    def P(self):
        return self.parent.P[self.idcs]

class ParticleGraph(object):
    def __init__(self, label, atoms, options):
        self.label = label
        self.P, self.centers = self.setupVertexFeatures(atoms, options) # <- Feature matrix
        self.L, self.D = self.setupLaplacian(atoms, options) # <- Laplacian matrix
        self.K = None
        self.subgraphs = None
        self.subgraph_cutoffs = None
        self.Z = atoms.get_atomic_numbers()
        if options['graph']['hierarchical']:
            self.createSubgraphs(options)
        return
    def createSubgraphs(self, options):
        subgraphs = []
        subgraph_cutoffs = []
        n_levels = options['graph']['n_levels']   
        r0 = options['graph']['r0']
        alpha = options['graph']['alpha']
        for l in range(n_levels):
            r_cut_l = r0*alpha**l
            subgraphs.append([])
            subgraph_cutoffs.append(r_cut_l)
            for i in range(self.D.shape[0]):
                idcs_sub = np.where(self.D[i] < r_cut_l)[0]
                subgraph = ParticleSubgraph(
                    i, self, idcs_sub, self.centers[i], r_cut_l)
                subgraphs[-1].append(subgraph)
        self.subgraphs = subgraphs
        self.subgraph_cutoffs = subgraph_cutoffs
        self.n_levels = len(self.subgraphs)
        return self.subgraphs
    def computeKernel(self, options):
        assert False
        kernelfct = kern.KernelFunctionFactory[options['kernel.type']](options)
        self.K = kernelfct.computeBlock(self.P)
        return self.K
    def setupLaplacian(self, atoms, options):
        # TODO Account for PBC
        n_atoms = len(atoms)
        L = np.zeros((n_atoms, n_atoms))
        D = np.zeros((n_atoms, n_atoms))
        # Read options
        inverse_dist = options['laplacian']['inverse_dist']
        scale = options['laplacian']['scale']
        coulomb = options['laplacian']['coulomb']
        # Off-diagonal
        for i in range(n_atoms):
            ai = atoms[i]
            for j in range(i+1, n_atoms):                
                aj = atoms[j]
                rij = ai.position - aj.position
                Rij = np.dot(rij, rij)**0.5
                # Distance matrix
                D[i,j] = Rij
                D[j,i] = D[i,j]
                # Laplacian
                if coulomb: 
                    pre = ai.number*aj.number
                else: 
                    pre = 1.
                if inverse_dist:
                    L[i,j] = -1.*scale*pre * 1./Rij
                    L[j,i] = L[i,j]
                else:
                    L[i,j] = -1.*scale*pre * Rij
                    L[j,i] = L[i,j]
        # Diagonal
        d = -np.sum(L, axis=1)
        np.fill_diagonal(L, d)
        np.fill_diagonal(D, 0.)
        return L, D
    def setupVertexFeatures(self, atoms, options):
        n_atoms = len(atoms)
        positions = [ atom.position for atom in atoms ]
        descriptor_type = options['graph']['descriptor']
        options_descriptor = options['descriptor'][descriptor_type]
        if descriptor_type == 'atom_type':
            feature_map = {}
            feature_list = options_descriptor['type_map']
            dim = len(feature_list)
            P = np.zeros((n_atoms, dim))
            for idx, atom in enumerate(atoms):
                p = np.zeros((dim))
                atom_type = atom.number
                for i in range(dim):
                    if feature_list[i] == atom_type: p[i] = 1
                    else: p[i] = 0
                P[idx,:] = p
        elif descriptor_type == 'soap':
            # Structure
            structure = soap.tools.setup_structure_ase(self.label, atoms)
            # Options
            options_soap = soap.Options()
            for item in options_descriptor.items():
                key = item[0]
                val = item[1]
                if type(val) == list: continue # TODO Exclusions loaded separately, fix this
                options_soap.set(key, val)
            options_soap.excludeCenters(options_descriptor['exclude_centers'])
            options_soap.excludeTargets(options_descriptor['exclude_targets'])
            # Spectrum
            spectrum = soap.Spectrum(structure, options_soap)
            spectrum.compute()
            spectrum.computePower()
            if options_descriptor['spectrum.gradients']:
                spectrum.computePowerGradients()
            spectrum.computeGlobal()
            # Adapt spectrum
            adaptor = kern.KernelAdaptorFactory[options_soap.get('kernel.adaptor')](
                options_soap, 
                types_global=options_descriptor['type_list'])
            ix = adaptor.adapt(spectrum)
            dim = ix.shape[1]
            assert ix.shape[0] == n_atoms
            P = ix
        elif descriptor_type == 'soap-quippy':
            atoms_quippy = datasets.gdb.convert_ase2quippy_atomslist([atoms])[0]
            # Read options
            options_xml_file = options_descriptor["options_xml"]
            opt_interface = momo.OptionsInterface()
            xml_options = opt_interface.ParseOptionsFile(options_xml_file, 'options')
            # Finalize options
            xml_options.kernel.alchemy = xml_options.kernel.alchemy.As(str)
            xml_options.kernel.alchemy_rules = xml_options.kernel.alchemy
            xml_options.soap.nocenter = xml_options.soap.nocenter.As(str)
            xml_options.soap.noatom = [] # TODO
            if xml_options.soap.nocenter and xml_options.soap.nocenter != 'None':
                xml_options.soap.nocenter = map(int, xml_options.soap.nocenter.split())
            else:
                xml_options.soap.nocenter = []
            datasets.soap.finalize_options([], xml_options)
            # Process
            z_types = options_descriptor["z_type_list"]
            struct = libmatch.structures.structure(xml_options.kernel.alchemy)
            soap_raw = struct.parse(atoms_quippy,
                xml_options.soap.R.As(float),
                xml_options.soap.N.As(int),
                xml_options.soap.L.As(int),
                xml_options.soap.sigma.As(float),
                xml_options.soap.w0.As(float),
                xml_options.soap.nocenter,
                xml_options.soap.noatom,
                types = z_types,
                kit = xml_options.kernel.kit)
            # Assign raw soaps to atoms (currently stored by z-key)
            z_idx_counter = {}
            for z in soap_raw:
                #print z, soap_raw[z].shape
                z_idx_counter[z] = 0
            ix = []
            for i,z in enumerate(atoms.get_atomic_numbers()):
                z_idx = z_idx_counter[z]
                ix.append(soap_raw[z][z_idx])
                z_idx_counter[z] += 1
                #print z, z_idx
            P = np.array(ix)
            dim = P.shape[1]
            assert P.shape[0] == n_atoms
            #print P.dot(P.T)
        elif descriptor_type == 'npy_load':
            folder = options_descriptor["folder"]
            npy_file = '%s/%s.x.npy' % (folder, self.label)
            print npy_file
            P = np.load(npy_file)
            dim = P.shape[1]
            assert P.shape[0] == n_atoms
        elif descriptor_type == 'none':
            dim = 1
            P = np.zeros((n_atoms, dim))
            for idx, atom in enumerate(atoms):
                P[idx,0] = 1.
        else:
            raise NotImplementedError(descriptor_type)        
        return P, positions
        
def mp_create_graph(config, options, log):
    soap.soapy.util.MP_LOCK.acquire()
    log << log.item << "Graph('%s') PID=%d" % (\
        config.config_file, mp.current_process().pid) << log.endl
    soap.soapy.util.MP_LOCK.release()
    graph = ParticleGraph(config.config_file, config.atoms, options)
    return graph        
        
        
        
        

