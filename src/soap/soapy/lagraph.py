import numpy as np
import numpy.linalg as la
import kernel as kern
import soap


class ParticleGraphVertex(object):
    def __init__(self, atom, options):
        self.atom = atom
        self.p = None
        
def sorted_eigenvalues_vectors(matrix):
    # i-th column(!) of v is eigenvector to i-th eigenvalue in w
    w,V = la.eig(matrix)
    order = w.argsort()
    w = w[order]
    V = V[:,order]
    return w,V

def flg_compute_S(L, Q, eta, gamma):
    L_reg = L+eta*np.identity(L.shape[0])
    L_reg_inv = la.inv(L_reg)
    S_reg = Q.T.dot(L_reg_inv).dot(Q) + gamma*np.identity(Q.shape[1])
    return S_reg

def flg_compute_k_flg(S1, S2):
    S1_inv = la.inv(S1)
    S2_inv = la.inv(S2)        
    S12_inv = 0.5*(S1_inv+S2_inv)
    S12_inv_inv = la.inv(S12_inv)
    det_12 = la.det(S12_inv_inv)
    det_1_2 = la.det(S1)*la.det(S2)
    if det_12 < 0. or det_1_2 < 0.:
        print "WARNING__%+1.7f<0__%+1.7f<0__" % (det_12, det_1_2)
    #return (la.det(S12_inv_inv))**0.5/(la.det(S1)*la.det(S2))**0.25 
    return det_12**0.5/det_1_2**0.25

def compare_graphs_kernelized(L1, L2, K12, options, zero_eps=1e-10):
    
    assert L1.shape[0] + L2.shape[0] == K12.shape[0]
    assert L1.shape[1] + L2.shape[1] == K12.shape[1]
    assert K12.shape[0] == K12.shape[1]
    
    # JOINT-SUB-SPACE CALCULATION
    # Non-zero-eigenvalue eigenspace
    eigvals, eigvecs = sorted_eigenvalues_vectors(K12)
    sel_idx = []
    for i in range(K12.shape[0]):
        if abs(eigvals[i]) < zero_eps: pass
        else: sel_idx.append(i)
    eigvals_nonzero = eigvals[sel_idx]
    eigvecs_nonzero = eigvecs[:,sel_idx]
    
    # Form subspace projection matrix Q
    Q = eigvals_nonzero*eigvecs_nonzero
    Q1 = Q[0:L1.shape[0],:]
    Q2 = Q[L1.shape[0]:K12.shape[0],:]
    
    # COMPUTE SPECTRUM OVERLAP   
    # Inverse regularized laplacian
    eta = options['laplacian.regularize_eta']
    gamma = options['laplacian.regularize_gamma']    
    S1_reg = flg_compute_S(L1, Q1, eta, gamma).real
    S2_reg = flg_compute_S(L2, Q2, eta, gamma).real
    
    # Laplacian-graph kernel
    k_flg = flg_compute_k_flg(S1_reg, S2_reg)
        
    return k_flg

def compare_graphs_laplacian_kernelized(g1, g2, options):
    if options['laplacian.hierarchical']:
        return compare_graphs_hierarchical(g1, g2, options)
    else:
        return compare_graphs(g1, g2, options)

def compare_graphs(g1, g2, options):

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
    
    # KERNELIZED COMPARISON
    k_flg = compare_graphs_kernelized(L1, L2, K12, options)
    return k_flg

def compare_graphs_hierarchical(g1, g2, options):

    # SANITY CHECKS
    assert g1.n_levels == g2.n_levels
    n_levels = g1.n_levels
    
    n1 = g1.L.shape[0]
    n2 = g2.L.shape[0]
    n12 = n1+n2    
    K12 = np.zeros((n12,n12))
    K12_l_out = np.copy(K12)
    
    for l in range(n_levels):
        subgraphs_l = g1.subgraphs[l] + g2.subgraphs[l]
        #print "l =", l, len(subgraphs_l), n12
        assert len(subgraphs_l) == n12
        # 0th level => base-kernel-based comparison
        if l == 0:
            for i in range(n12):
                gsub_i = subgraphs_l[i]
                for j in range(i, n12):
                    gsub_j = subgraphs_l[j]
                    #print "l=%d: i = %2d size = %2d | j = %2d size = %2d" % (
                    #    l, i, gsub_i.L.shape[0], j, gsub_j.L.shape[0])                    
                    k_flg = compare_graphs(gsub_i, gsub_j, options)
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
                    idcs_sub_j = gsub_j.idcs
                    #print "l=%d: i = %2d size = %2d | j = %2d size = %2d" % (
                    #    l, i, gsub_i.L.shape[0], j, gsub_j.L.shape[0])
                    # Short-list K12_l_out (= create appropriate view)                    
                    idcs = np.zeros((gsub_i.size+gsub_j.size), dtype='int')
                    idcs[0:gsub_i.size] = idcs_sub_i
                    idcs[gsub_i.size:] = idcs_sub_j if j < n1 else n1 + idcs_sub_j
                    K12_sub = K12_l_out[idcs][:,idcs] # <- TODO This is slow
                    #K12_sub = K12_l_out[np.ix_(idcs,idcs)] # <- TODO Also slow
                    # Compute kernel
                    k_flg = compare_graphs_kernelized(gsub_i.L, gsub_j.L, K12_sub, options)
                    K12[i,j] = k_flg
                    K12[j,i] = K12[i,j]                    
            # Update child kernel
            K12_l_out = np.copy(K12)
    
    k_flg_top = compare_graphs_kernelized(g1.L, g2.L, K12_l_out, options)
    
    return k_flg_top

class ParticleSubgraph(object):
    def __init__(self, idx, idcs_sub, P_sub, L_sub, D_sub):
        self.idx = idx
        self.idcs = idcs_sub
        self.size = len(self.idcs)
        self.P = P_sub # <- Feature matrix
        self.L = L_sub # <- Laplacian matrix
        self.D = D_sub # <- Distance matrix
        assert self.P.shape[0] == self.size
        assert self.L.shape[0] == self.size
        assert self.D.shape[0] == self.size
        return

class ParticleGraph(object):
    def __init__(self, label, atoms, options):
        self.label = label
        self.P = self.setupVertexFeatures(atoms, options) # <- Feature matrix
        self.L, self.D = self.setupLaplacian(atoms, options) # <- Laplacian matrix
        self.K = None
        self.subgraphs = None
    def createSubgraphs(self, options):
        subgraphs = []
        n_levels = options['laplacian.n_levels']   
        r0 = options['laplacian.r0']
        alpha = options['laplacian.alpha']
        for l in range(n_levels):
            r_cut_l = r0*alpha**l
            subgraphs.append([])
            for i in range(self.D.shape[0]):
                idcs_sub = np.where(self.D[i] < r_cut_l)[0]
                D_sub = self.D[idcs_sub][:,idcs_sub]
                L_sub = self.L[idcs_sub][:,idcs_sub]
                P_sub = self.P[idcs_sub]
                subgraph = ParticleSubgraph(i, idcs_sub, P_sub, L_sub, D_sub)
                subgraphs[-1].append(subgraph)
                #print i
                #print D_sub
                #print P_sub
        self.subgraphs = subgraphs
        self.n_levels = len(self.subgraphs)
        return self.subgraphs
    def computeKernel(self, options):
        kernelfct = kern.KernelFunctionFactory[options['kernel.type']](options)
        self.K = kernelfct.computeBlock(self.P)
        return self.K
    def setupLaplacian(self, atoms, options):
        # TODO Account for PBC
        n_atoms = len(atoms)
        L = np.zeros((n_atoms, n_atoms))
        D = np.zeros((n_atoms, n_atoms))
        # Read options
        inverse_dist = options['laplacian.inverse_dist']
        scale = options['laplacian.scale']        
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
                if inverse_dist:
                    L[i,j] = -1.*scale*ai.number*aj.number * 1./Rij
                    L[j,i] = L[i,j]
                else:
                    L[i,j] = -1.*scale*ai.number*aj.number * Rij
                    L[j,i] = L[i,j]
        # Diagonal
        d = -np.sum(L, axis=1)
        np.fill_diagonal(L, d)
        np.fill_diagonal(D, 0.)
        return L, D
    def setupVertexFeatures(self, atoms, options):
        n_atoms = len(atoms)
        descriptor_type = options['laplacian.descriptor']
        options_descriptor = options['descriptor.%s' % descriptor_type]
        if descriptor_type == 'atom_type':
            feature_map = {}
            feature_list = options_descriptor['type_map']
            for idx, f in enumerate(feature_list): feature_map[f] = idx
            dim = len(feature_map.keys())
            P = np.zeros((n_atoms, dim))
            for idx, atom in enumerate(atoms):
                p = np.zeros((dim))
                atom_type = atom.number
                p[feature_map[atom_type]] = 1
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
            adaptor = kern.KernelAdaptorFactory[options_soap.get('kernel.adaptor')](options_soap)
            ix = adaptor.adapt(spectrum)
            dim = ix.shape[1]
            assert ix.shape[0] == n_atoms
            P = ix
        else:
            raise NotImplementedError(descriptor_type)        
        return P
        
        
        
        
        
        

