#! /usr/bin/env python
import soap
import json
import numpy as np
import pickle
import h5py
import time
import datetime
import os
import psutil

# ============
# GRAPH OBJECT
# ============

class Graph(object):
    def __init__(self, 
            idx=-1, 
            label='', 
            feature_mat=None, 
            feature_mat_avg=None,
            position_mat=None, 
            connectivity_mat=None, 
            vertex_info=[], 
            graph_info={}):
        # Labels
        self.idx = idx
        self.label = label
        self.graph_info = graph_info
        # Vertex data: descriptors
        self.P = feature_mat
        self.P_avg = feature_mat_avg
        self.P_type_str = str(type(self.P))
        # Vertex data: positions, labels
        self.R = position_mat
        self.vertex_info = vertex_info
        # Edge data
        self.C = connectivity_mat
        return
    def save_to_h5(self, h5f, dtype='float32'):
        group = h5f.create_group('%06d' % self.idx)
        if self.P_type_str == "<class 'soap.soapy.kernel.DescriptorMapMatrix'>":
            # Save list of descriptor maps
            g0 = group.create_group('feature_dmap')
            for dmap_idx, dmap in enumerate(self.P):
                g1 = g0.create_group('%d' % dmap_idx)
                for key in dmap:
                    g1.create_dataset(
                        key, 
                        data=dmap[key], 
                        compression='gzip', 
                        dtype=dtype)
            # Save averaged descriptor map
            if type(self.P_avg) != type(None):
                g0_avg = group.create_group('feature_dmap_avg')
                for key in self.P_avg:
                    g0_avg.create_dataset(
                        key, 
                        data=self.P_avg[key], 
                        compression='gzip', 
                        dtype=dtype)
        elif self.P_type_str == "<type 'numpy.ndarray'>":
            # Save numpy arrays
            group.create_dataset(
                'feature_mat', 
                data=self.P, 
                compression='gzip', 
                dtype=dtype)
            if type(self.P_avg) != type(None):
                group.create_dataset(
                    'feature_mat_avg', 
                    data=self.P_avg, 
                    compression='gzip', 
                    dtype=dtype)
        else: raise NotImplementedError(self.P_type_str)
        group.create_dataset('position_mat', data=self.R)
        group.create_dataset('connectivity_mat', data=self.C)
        group.attrs['idx'] = self.idx
        group.attrs['label'] = self.label
        group.attrs['vertex_info'] = json.dumps(self.vertex_info)
        group.attrs['graph_info'] = json.dumps(self.graph_info)
        group.attrs['P_type_str'] = self.P_type_str
        return
    def load_from_h5(self, h5f):
        self.idx = h5f.attrs['idx']
        self.label = h5f.attrs['label']
        self.vertex_info = json.loads(h5f.attrs['vertex_info'])
        self.graph_info = json.loads(h5f.attrs['graph_info'])
        # Determine feature matrix type
        if 'P_type_str'in h5f.attrs:
            self.P_type_str = h5f.attrs['P_type_str']
        else:
            self.P_type_str = "<type 'numpy.ndarray'>"
        if self.P_type_str == "<class 'soap.soapy.kernel.DescriptorMapMatrix'>":
            # Load list of descriptor maps
            self.P = soap.soapy.kernel.DescriptorMapMatrix()
            g0 = h5f['feature_dmap']
            for i in range(len(g0)):
                Pi = soap.soapy.kernel.DescriptorMap()
                g1 = g0['%d' % i]
                for key in g1:
                    Pi[key] = g1[key].value
                self.P.append(Pi)
            # Load averaged descriptor map
            self.P_avg = soap.soapy.kernel.DescriptorMap()
            if 'feature_dmap_avg' in h5f:
                g0_avg = h5f['feature_dmap_avg']
                for key in g0_avg:
                    self.P_avg[key] = g0_avg[key].value
        elif self.P_type_str == "<type 'numpy.ndarray'>":
            self.P = h5f['feature_mat'].value
            if 'feature_mat_avg' in h5f:
                self.P_avg = h5f['feature_mat_avg'].value
            else: self.P_avg = None
        else: raise NotImplementedError(self.P_type_str)
        self.R = h5f['position_mat'].value
        self.C = h5f['connectivity_mat'].value
        return self

def load_graphs(hdf5, n=-1):
    h = h5py.File(hdf5, 'r')
    gsec = h['graphs']
    if n == None or n < 0: n = len(gsec)
    graphs = [ Graph().load_from_h5(
        gsec['%06d' % i]) for i in range(len(gsec)) if i < n ]
    h.close()
    return graphs

def calculate_graph_connectivity(graph, zero_diagonal):
    """
    Suited for periodic-table elements:
    Computes connectivity using covalent radii
    """
    N = graph.R.shape[0]
    D = np.zeros((N,N), dtype='float64')
    for i in range(N):
        for j in range(i,N):
            dij = ((graph.R[i]-graph.R[j]).dot(graph.R[i]-graph.R[j]))**0.5
            D[i,j] = dij 
            D[j,i] = dij
    C = soap.tools.partition.calculate_connectivity_mat(D, graph.vertex_info)
    if zero_diagonal:
        np.fill_diagonal(C, False)
    return C

def filter_graph(graph, filter_fct=lambda i, g: True, log=None):
    # Vertex is kept if filter_fct evaluates to True, otherwise removed
    if log: 
        log << "Filtering graph" << log.endl
        log << "Graph size initial:" << graph.P.shape[0] << log.endl
    idcs_keep = []
    idcs_removed = [] 
    vertex_info_keep = []
    for idx, type_ in enumerate(graph.vertex_info):
        if filter_fct(idx, graph):
            idcs_keep.append(idx)
            vertex_info_keep.append(type_)
        else: 
            idcs_removed.append(idx)
    # Reduce features, positions, connecitivity
    P_keep = graph.P[idcs_keep]
    R_keep = graph.R[idcs_keep]
    graph_filtered = Graph(
        idx=graph.idx,
        label=graph.label,
        feature_mat=P_keep,
        position_mat=R_keep,
        connectivity_mat=np.array([]),
        vertex_info=vertex_info_keep,
        graph_info=graph.graph_info
    )
    if log: log << "Graph size final:" << graph_filtered.P.shape[0] << log.endl
    return graph_filtered

# ================
# KERNEL FUNCTIONS
# ================

class BaseKernelDot(object):
    def __init__(self, options):
        self.xi = options['xi']
        self.delta = options['delta']
        return
    def compute(self, X, Y):
        return self.delta**2 * X.dot(Y.T)**self.xi

class TopKernelRematch(object):
    def __init__(self, options, basekernel):
        self.gamma = options['gamma']
        self.basekernel = basekernel
        return
    def compute(self, g1, g2, log=None):
        if log: log << "[Kernel] %s %s" % (g1.graph_info['label'], g2.graph_info['label']) << log.endl
        K_base = self.basekernel.compute(g1.P, g2.P)
        # Only works with float64 (due to C-casts?) ...
        if K_base.dtype != 'float64':
            K_base = K_base.astype('float64')
        k_top = soap.linalg.regmatch(K_base, self.gamma, 1e-6)
        return k_top
    def preprocess(self, g, log=None):
        return g

class TopKernelCanonical(object):
    def __init__(self, options, basekernel):
        self.xi = options['xi']
        self.basekernel = basekernel
    def compute(self, g1, g2, log=None):
        if log: log << "[Kernel] %s %s" % (g1.graph_info['label'], g2.graph_info['label']) << log.endl
        K_base = self.basekernel.compute(g1.P, g2.P)
        k_top = soap.soapy.lamatch.reduce_kernel_canonical(K_base, self.xi)
        return k_top
    def preprocess(self, g, log=None):
        return g

class TopKernelRematchHierarchical(object):
    def __init__(self, options, basekernel):
        self.gamma = options['gamma']
        self.basekernel = basekernel
        self.bond_order = options['bond-order']
        self.concatenate = options['concatenate']
        return
    def compute(self, g1, g2, log=None):
        K_base = self.basekernel.compute(g1.P, g2.P)
        for n in self.concatenate:
            K_base = K_base*self.basekernel.compute(g1.P_avg_n_dict[n], g2.P_avg_n_dict[n])
        # Only works with float64 (due to C-casts?) ...
        if K_base.dtype != 'float64':
            K_base = K_base.astype('float64')
        k_top = soap.linalg.regmatch(K_base, self.gamma, 1e-6)
        if log: log << "[Kernel] %s %s" % (g1.graph_info['label'], g2.graph_info['label']) << k_top << log.endl
        return k_top
    def preprocess(self, g, log=None):
        if log: log << "Preprocessing %s" % (g.graph_info['label']) << log.endl
        N = g.R.shape[0]
        # Bond and distance matrix
        D = np.zeros((N,N), dtype='float64')
        B = np.zeros((N,N), dtype='bool')
        for i in range(N):
            for j in range(i, N):
                ri = g.R[i]
                ti = str(g.vertex_info[i])
                rj = g.R[j]
                tj = str(g.vertex_info[j])
                # Distance
                drij = np.dot(ri-rj,ri-rj)**0.5
                D[i,j] = drij
                D[j,i] = drij
                # Covalent radii
                rci = soap.soapy.periodic_table[ti].covrad
                rcj = soap.soapy.periodic_table[tj].covrad
                drcij = soap.tools.partition.covalent_cutoff(rci, rcj)
                # Check for bond
                if drij <= drcij: bij = True
                else: bij = False
                B[i,j] = bij
                B[j,i] = bij
        # List of bonded atoms (via idcs)
        B_idcs_0 = []
        for i in range(N):
            b_idcs = list(np.where(B[i] == True)[0])
            B_idcs_0.append(b_idcs)
        P_avg_n_list = []
        P_avg_n_list.append(g.P) # 0-th order environment ^= centre atom
        for n_bond_order in range(self.bond_order):
            B_idcs = []
            for i in range(N):
                b_idcs = list(np.where(B[i] == True)[0])
                B_idcs.append(b_idcs)
            # Higher-order bond list
            for n in range(n_bond_order-1):
                B_idcs_out = []
                for i in range(N):
                    b_idcs_in = B_idcs[i]
                    b_idcs_out = [] + b_idcs_in
                    for idx in b_idcs_in:
                        b_idcs_out = list(set(b_idcs_out + B_idcs_0[idx]))
                    B_idcs_out.append(b_idcs_out)
                B_idcs = B_idcs_out
            #if log:
            #    for idx, b_idcs in enumerate(B_idcs):
            #        log << 'index' << idx << str(b_idcs).replace(',',' ').replace('[','').replace(']','') << log.endl
            # Average neighbourhood descriptor
            P_avg_n = np.zeros(g.P.shape, dtype='float64')
            for i in range(N):
                p_avg = np.average(g.P[B_idcs[i]], axis=0)
                p_avg = p_avg/np.dot(p_avg, p_avg)**0.5
                P_avg_n[i] = p_avg
            P_avg_n_list.append(P_avg_n)
        # Store
        g.P_avg_n_dict = { n: P_avg_n_list[n] for n in self.concatenate }
        #g.P_avg_n_dict = { n: P_avg_n_list[n] for n in range(self.bond_order+1) }
        #g.B = B_idcs
        return

class TopKernelRematchAtomic(object):
    def __init__(self, options, basekernel):
        self.gamma = options['gamma']
        self.basekernel = basekernel
        return
    def compute(self, g1, g2, log=None):
        if log: log << "[Kernel] %s %s" % (g1.graph_info['label'], g2.graph_info['label']) << log.endl
        K_base = self.basekernel.compute(g1.P, g2.P)
        # Only works with float64 (due to C-casts?) ...
        if K_base.dtype != 'float64':
            K_base = K_base.astype('float64')
        P_base = soap.linalg.kernel_rematch_atomic(K_base, self.gamma, 1e-6)
        k_top = np.sum(P_base*K_base)
        return k_top, K_base, P_base
    def preprocess(self, g, log=None):
        return g

class TopKernelAverage(object):
    def __init__(self, options, basekernel):
        self.basekernel = basekernel
        return
    def compute(self, g1, g2, log=None):
        p1_avg = np.average(g1.P, axis=0)
        p2_avg = np.average(g2.P, axis=0)
        p1_avg = p1_avg/np.dot(p1_avg,p1_avg)**0.5
        p2_avg = p2_avg/np.dot(p2_avg,p2_avg)**0.5
        return self.basekernel.compute(p1_avg, p2_avg)
    def preprocess(self, g, log=None):
        return g

BaseKernelFactory = {
    'dot' : BaseKernelDot
}

TopKernelFactory = {
    'rematch' : TopKernelRematch,
    'rematch-atomic': TopKernelRematchAtomic,
    'rematch-hierarchical': TopKernelRematchHierarchical,
    'average' : TopKernelAverage,
    'canonical' : TopKernelCanonical
}

# ==================
# MP COMPUTE KERNELS
# ==================

def mp_get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def mp_compute_graph(
        config,
        fragment_based,
        descriptor_type,
        descriptor_options,
        log):
    if log: 
        soap.soapy.util.MP_LOCK.acquire()
        log << log.back << "[Graph] Processing %s" % config.info['label'] << log.endl
        soap.soapy.util.MP_LOCK.release()
    # Config => struct + connectivity matrices
    # NOTE This will reorder the atoms in config
    config, struct, top, frag_bond_mat, atom_bond_mat, frag_labels, atom_labels = \
        soap.tools.structure_from_ase(
            config, 
            do_partition=fragment_based, 
            add_fragment_com=fragment_based, 
            log=None)
    # Struct => spectrum
    compute_descriptor = DescriptorFactory[descriptor_type]
    feature_mat, position_mat, type_vec = compute_descriptor(
        struct, 
        descriptor_options, 
        fragment_based=fragment_based)
    # Config, spectrum => graph
    if fragment_based:
        connectivity_mat = frag_bond_mat
        vertex_info = frag_labels
    else:
        connectivity_mat = atom_bond_mat
        vertex_info = type_vec
    graph = Graph(
        idx = config.info['idx'],
        label = str(config.info['label']),
        feature_mat=feature_mat,
        position_mat=position_mat,
        connectivity_mat=connectivity_mat,
        vertex_info=vertex_info,
        graph_info=config.info)
    return graph

def mp_compute_kernel_idx_pair(idx1, idx2, kernel, h5f):
    g1 = Graph()
    g2 = Graph()
    g1.load_from_h5(h5f['%06d' % idx1])
    g2.load_from_h5(h5f['%06d' % idx2])
    k = kernel.compute(g1, g2)
    return k

def mp_compute_kernel_g_pair(g1, g2, kernel, h5f):
    k = kernel.compute(g1, g2)
    return k

def mp_compute_kernel_block_hdf5(block, kernel, log, dtype_result, h5_file):
    rows = list(block[0])
    cols = list(block[1])
    #soap.soapy.util.MP_LOCK.acquire()
    f = h5py.File(h5_file, 'r')
    h5_graphs = f['graphs']
    g_rows = [ Graph().load_from_h5(h5_graphs['%06d' % i]) for i in rows ]
    g_cols = [ Graph().load_from_h5(h5_graphs['%06d' % i]) for i in cols ]
    f.close()
    #soap.soapy.util.MP_LOCK.release()
    return mp_compute_kernel_block([g_rows, g_cols], kernel, log, dtype_result)

def mp_compute_kernel_block(block, kernel, log, dtype_result, symmetric=True):
    gi_block = block[0]
    gj_block = block[1]
    if log: 
        soap.soapy.util.MP_LOCK.acquire()
        log << "Processing block %4d:%-4d %4d:%-4d" % (gi_block[0].idx, gi_block[-1].idx, gj_block[0].idx, gj_block[-1].idx) << "PID =" << os.getpid() << log.endl
        soap.soapy.util.MP_LOCK.release()
    kmat = np.zeros((len(gi_block),len(gj_block)), dtype=dtype_result)
    for i,gi in enumerate(gi_block):
        for j,gj in enumerate(gj_block):
            if symmetric and gi.idx > gj.idx: pass
            else: kmat[i,j] = kernel.compute(gi,gj)
    return kmat

def matrix_blocks(array, block_size, upper_triangular):
    n_blocks = np.ceil(array.shape[0]/block_size)
    if n_blocks < 1: n_blocks = 1
    chunks = np.array_split(array, n_blocks)
    blocks = []
    for i in xrange(len(chunks)):
        for j in xrange(i if upper_triangular else 0, len(chunks)):
            blocks.append((chunks[i], chunks[j]))
    return blocks

def matrix_blocks_by_row(array, block_size, upper_triangular):
    n_blocks = np.ceil(array.shape[0]/block_size)
    if n_blocks < 1: n_blocks = 1
    chunks = np.array_split(array, n_blocks)
    blocks = []
    for i in xrange(len(chunks)):
        blocks.append([])
        for j in xrange(i if upper_triangular else 0, len(chunks)):
            blocks[-1].append((chunks[i], chunks[j]))
    return blocks

def read_graph_block(h5_graphs, block, log):
    if log: 
        soap.soapy.util.MP_LOCK.acquire()
        log << "Reading block    %4d:%-4d %4d:%-4d" % (block[0][0], block[0][-1], block[1][0], block[1][-1]) << "PID =" << os.getpid() << log.endl
        soap.soapy.util.MP_LOCK.release()
    rows = list(block[0])
    cols = list(block[1])
    if (type(h5_graphs) == list):
        g_rows = [ h5_graphs[i] for i in rows ]
        g_cols = [ h5_graphs[j] for j in cols ]
    else:
        g_rows = [ Graph().load_from_h5(h5_graphs['%06d' % i]) for i in rows ]
        g_cols = [ Graph().load_from_h5(h5_graphs['%06d' % i]) for i in cols ]
    return [g_rows, g_cols]

# ==========
# LOAD-WRITE
# ==========

def read_filter_configs(
        config_file, 
        index=':', 
        filter_types=None, 
        types=[],
        do_remove_duplicates=True, 
        key=lambda c: c.info['label'],
        log=None):
    if log: log << "Reading" << config_file << log.endl
    configs = soap.tools.io.read(config_file, index=index)
    if log: log << log.item << "Have %d initial configurations" % len(configs) << log.endl
    if do_remove_duplicates:
        configs, duplics = remove_duplicates(configs, key=key)
        if log: log << log.item << "Removed %d duplicates" % len(duplics) << log.endl
    if filter_types:
        configs_filtered = []
        for config in configs:
            types_config = config.get_chemical_symbols()
            keep = True
            for t in types_config:
                if not t in types:
                    keep = False
                    break
            if keep: configs_filtered.append(config)
        configs = configs_filtered
        if log: log << log.item << "Have %d configurations after filtering" % len(configs) << log.endl
    return configs

def remove_duplicates(array, key=lambda a: a):
    len_in = len(array)
    label = {}
    array_curated = []
    array_duplicates = []
    for a in array:
        key_a = key(a)
        if key_a in label:
            array_duplicates.append(a)
        else:
            array_curated.append(a)
            label[key_a] = True
    len_out = len(array_curated)
    return array_curated, array_duplicates

# =============
# DESC WRAPPERS
# =============

def compute_ftd(struct, options, fragment_based):
    for atom in struct:
        atom.sigma = options["fieldtensor.sigma"]
        log << atom.name << atom.type << atom.weight << atom.sigma << atom.pos << log.endl
    # OPTIONS
    options_soap = soap.Options()
    for key, val in options.items():
        if type(val) == list: continue
        options_soap.set(key, val)
    # Exclusions
    excl_targ_list = options['exclude_targets']
    excl_cent_list = options['exclude_centers']
    excl_targ_list.append('COM')
    if not fragment_based:
        excl_cent_list.append('COM')
    options_soap.excludeCenters(excl_cent_list)
    options_soap.excludeTargets(excl_targ_list)
    # SPECTRUM
    ftspectrum = soap.FTSpectrum(struct, options_soap)
    ftspectrum.compute()
    # Adapt spectra
    adaptor = soap.soapy.kernel.KernelAdaptorFactory["ftd-specific"](
        {}, 
        options["type_list"])
    IX, IR, types = adaptor.adapt(ftspectrum, return_pos_matrix=True)
    return IX, IR, types

def compute_soap(struct, options, fragment_based):
    # OPTIONS
    options_soap = soap.Options()
    for key, val in options.items():
        if type(val) == list: continue
        options_soap.set(key, val)
    # Exclusions
    excl_targ_list = options['exclude_targets']
    excl_cent_list = options['exclude_centers']
    excl_targ_list.append('COM')
    if not fragment_based:
        excl_cent_list.append('COM')
    options_soap.excludeCenters(excl_cent_list)
    options_soap.excludeTargets(excl_targ_list)
    # SPECTRUM
    spectrum = soap.Spectrum(struct, options_soap)
    # Compute density expansion
    if fragment_based:
        # Divide onto segments and their COM representatives
        seg_by_name = {}
        comseg_by_name = {}
        for seg in struct.segments:
            if seg.name.split('.')[-1] == 'COM':
                comseg_by_name[seg.name] = seg
            else:
                seg_by_name[seg.name] = seg
        # Compute segment descriptors
        for name, seg in seg_by_name.items():
            comseg = comseg_by_name[name+'.COM']
            spectrum.compute(comseg, seg)
    else:
        spectrum.compute()
    # Compute power spectrum, 
    spectrum.computePower()
    if options['spectrum.gradients']:
        spectrum.computePowerGradients()
    if options['spectrum.global']:
        spectrum.computeGlobal()
    # Adapt spectrum
    adaptor = soap.soapy.kernel.KernelAdaptorFactory[options['kernel.adaptor']](
        options_soap,
        types_global=options['type_list'])
    IX, IR, types = adaptor.adapt(spectrum, return_pos_matrix=True)
    return IX, IR, types

# TODO Add descriptors (coulomb, hierarchical-coulomb, ...)
DescriptorFactory = {
    'soap' : compute_soap,
    'ftd'  : compute_ftd
}



