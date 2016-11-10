import os
import numpy as np
import multiprocessing as mp
import functools as fct
import json
import datetime
import resource

HARTREE_TO_EV = 27.21138602
HARTREE_TO_KCALMOL = 627.509469

MP_LOCK = mp.Lock()

def mp_pool_compute_upper_triangle(
        kfct,
        g_list,
        n_procs,
        dtype='float64',
        mplog=None,
        **kwargs):
    kfct_primed=fct.partial(kfct, **kwargs)
    n_rows = len(g_list)
    kmat = np.zeros((n_rows, n_rows), dtype=dtype)
    for i in range(n_rows):
        if mplog: mplog << mplog.back << "Computing row %d" % i << mplog.endl
        g_pair_list = []
        gi = g_list[i]
        for j in range(i, n_rows):
            gj = g_list[j]
            g_pair_list.append([gi,gj])
        pool = mp.Pool(processes=n_procs)
        krow = pool.map(kfct_primed, g_pair_list)
        pool.close()
        pool.join()
        kmat[i,i:] = krow
    return kmat

def mp_compute_vector(
        kfct,
        g_list,
        n_procs,
        **kwargs):
    kfct_primed = fct.partial(
        kfct,
        **kwargs)
    pool = mp.Pool(processes=n_procs)
    kvec = pool.map(kfct_primed, g_list)
    pool.close()
    pool.join()
    return kvec

def mp_compute_column_block(gi, gj_list, kfct):
    """
    Evaluates kfct for each pair (gi, gj), with gj from gj_list

    See Also
    --------
    mp_compute_upper_triangle
    """
    krow = []
    for gj in gj_list:
        if gj.mp_idx < gi.mp_idx:
            k = 0.
        else:
            k = kfct(gi, gj)
        krow.append(k)
    return krow

def mp_compute_column_block_idx(gi, gj_list, kfct):
    """
    Evaluates kfct for each pair (i, j), with j from j_list
    """
    krow = []
    for gj in gj_list:
        if gj < gi:
            k = 0.
        else:
            k = kfct(gi, gj)
        krow.append(k)
    return krow

def compute_upper_triangle(
        kfct,
        g_list,
        **kwargs):
    dim = len(g_list)
    kmat = np.zeros((dim,dim), dtype='float64')
    kfct_primed = fct.partial(
        kfct,
        **kwargs)
    for i in range(dim):
        gi = g_list[i]
        for j in range(i, dim):
            gj = g_list[j]
            kij = kfct_primed(gi, gj)
            kmat[i,j] = kij
            kmat[j,i] = kij
    return kmat

def mp_compute_upper_triangle(
        kfct, 
        g_list, 
        n_procs, 
        n_blocks, 
        mplog=None, 
        tstart_twall=(None,None), 
        backup=True,
        verbose=True,
        embed_idx=True,
        **kwargs):
    """
    Compute kernel matrix computed from pairs of objects in object list

    Parameters
    ----------
    kfct : function reference to be evaluated between g_list items
    g_list : list of items supplied to kfct(gi, gj, **kwargs)
    n_procs: number of processors requested
    n_blocks: number of column blocks onto which computation is split
    kwargs: keyword arguments supplied to kfct
    """
    if not verbose: mplog=None
    t_start = tstart_twall[0]
    t_wall = tstart_twall[1]
    dim = len(g_list)
    kmat = np.zeros((dim,dim))
    if embed_idx:
        # Embed mp index in g-list objects
        for mp_idx, g in enumerate(g_list): g.mp_idx = mp_idx
    # Divide onto column blocks
    col_idcs = np.arange(len(g_list))
    col_div_list = np.array_split(col_idcs, n_blocks)
    for col_div_idx, col_div in enumerate(col_div_list):
        t_in = datetime.datetime.now()
        # Column start, column end
        c0 = col_div[0]
        c1 = col_div[-1]+1
        if mplog: mplog << "Column block i[%d:%d] j[%d:%d]" % (0, c1, c0, c1) << mplog.endl
        gj_list = g_list[c0:c1]
        gi_list = g_list[0:c1]
        # Prime kernel function
        kfct_primed = fct.partial(
            kfct,
            **kwargs)
        pool = mp.Pool(processes=n_procs)
        # Prime mp function
        mp_compute_column_block_primed = fct.partial(
            mp_compute_column_block if embed_idx else mp_compute_column_block_idx,
            gj_list=gj_list,
            kfct=kfct_primed)
        # Map & close
        npyfile = 'out.block_i_%d_%d_j_%d_%d.npy' % (0, c1, c0, c1)
        # ... but first check for previous calculations of same slice
        if backup and npyfile in os.listdir('./'):
            if mplog: mplog << "Load block from '%s'" % npyfile << mplog.endl
            kmat_column_block = np.load(npyfile)
        else:
            kmat_column_block = pool.map(mp_compute_column_block_primed, gi_list)
            kmat_column_block = np.array(kmat_column_block)
            if backup: np.save(npyfile, kmat_column_block)
        # Update kernel matrix
        kmat[0:c1,c0:c1] = kmat_column_block
        pool.close()
        pool.join()
        # Check time
        t_out = datetime.datetime.now()
        dt_block = t_out-t_in
        if t_start and t_wall:
            t_elapsed = t_out-t_start
            if mplog: mplog << "Time elapsed =" << t_elapsed << " (wall time = %s) (maxmem = %d)" % (t_wall, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) << mplog.endl
            if col_div_idx+1 != len(col_div_list) and t_elapsed+dt_block > t_wall-dt_block:
                mplog << "Wall time hit expected for next iteration, break ..." << mplog.endl
                break
            else: pass
    return kmat

def json_load_utf8(file_handle):
    return _byteify(
        json.load(file_handle, object_hook=_byteify),
        ignore_dicts=True
    )

def json_loads_utf8(json_text):
    return _byteify(
        json.loads(json_text, object_hook=_byteify),
        ignore_dicts=True
    )

def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data

def idcs_split_train_test(N_data, N_train, shift=0, method='stride'):
    N_test = N_data-N_train
    idcs = np.arange(0, N_data)
    if method == 'stride':
        idcs_test = idcs_select_stride(idcs, N_test, shift)
        idcs_train = idcs_select_complement(idcs, idcs_test)
    else:
        raise NotImplementedError(method)
    return idcs_train, idcs_test

def idcs_select_stride(idcs, n_sel, shift=0):
    idcs_sel = [ int(float(idcs.shape[0])/n_sel*i) for i in range(n_sel) ]
    idcs_sel = np.array(idcs_sel)
    if shift:
        idcs_sel = idcs_shift_pbc(idcs_sel, shift, idcs.shape[0])
    return idcs_sel

def idcs_shift_pbc(idcs, shift, length):
    return np.sort((idcs + shift) % length)

def idcs_select_complement(idcs, idcs_sel):
    mask = np.zeros(idcs.shape[0], dtype=bool)
    mask[idcs_sel] = True
    return idcs[~mask]








