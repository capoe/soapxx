import numpy as np
import multiprocessing as mp
import functools as fct
import json

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

def mp_compute_upper_triangle(kfct, g_list, n_procs, n_blocks, log=None, **kwargs):
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
    dim = len(g_list)
    kmat = np.zeros((dim,dim))
    # Embed mp index in g-list objects
    for mp_idx, g in enumerate(g_list): g.mp_idx = mp_idx
    # Divide onto column blocks
    col_idcs = np.arange(len(g_list))
    col_div_list = np.array_split(col_idcs, n_blocks)
    for col_div in col_div_list:
        # Column start, column end
        c0 = col_div[0]
        c1 = col_div[-1]+1
        if log: log << "Column block [%d:%d]" % (c0, c1) << log.endl
        gj_list = g_list[c0:c1]
        # Prime kernel function
        kfct_primed = fct.partial(
            kfct,
            **kwargs)
        pool = mp.Pool(processes=n_procs)
        # Prime mp function
        mp_compute_column_block_primed = fct.partial(
            mp_compute_column_block,
            gj_list=gj_list,
            kfct=kfct_primed)
        # Map & close
        kmat_column_block = pool.map(mp_compute_column_block_primed, g_list)
        kmat_column_block = np.array(kmat_column_block)
        kmat[:,c0:c1] = kmat_column_block
        pool.close()
        pool.join()
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
