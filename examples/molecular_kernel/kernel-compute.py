#! /usr/bin/env python
import soap
import soap.soapy.graph as pygraph
import ase.io
import json
import numpy as np
import pickle
import h5py
import time
import datetime
import os
import psutil

def run(log, cmdline_options, json_options):

    # Read options
    options = json_options
    n_procs = cmdline_options.n_procs
    mp_kernel_block_size = cmdline_options.mp_kernel_block_size
    mp_preload = False
    mp_hdf5_read_parallel = True
    # Type handling
    compile_types = cmdline_options.types_compile
    filter_types = False if compile_types else True
    # Selections
    n_select = cmdline_options.select
    subsample_method = 'stride' #'random'
    # Descriptor options
    descriptor_type = options['descriptor']['type']
    descriptor_options = options['descriptor'][descriptor_type]
    fragment_based = 'frag' in descriptor_type.split('-')[-1]
    soap.tools.partition.ligand_size_lower = 5 # TODO Make an option
    label_key = cmdline_options.label_key

    h5_file = cmdline_options.hdf5_out
    h5 = h5py.File(h5_file, 'w')

    # Read and select configurations
    configs_A = pygraph.read_filter_configs(
        cmdline_options.config_file, 
        index=':', 
        filter_types=filter_types, 
        types=descriptor_options['type_list'], 
        do_remove_duplicates=True, 
        key=lambda c: c.info[label_key], 
        log=log)
    configs_A, configs_A2 = soap.soapy.learn.subsample_array(
        configs_A, n_select=n_select, method=subsample_method, stride_shift=0)
    configs = configs_A

    # Index; find all chemical elements in dataset
    for idx, c in enumerate(configs):
        c.info['idx'] = idx
        if not 'label' in c.info:
            c.info['label'] = c.info[label_key]
    if compile_types:
        types_global = []
        for idx, c in enumerate(configs):
            types_global = types_global + c.get_chemical_symbols()
            types_global = list(set(types_global))
        types_global = sorted(types_global)
        log << "Compiled global types list:" << types_global << log.endl
        descriptor_options['type_list'] = types_global
    else:
        log << "Using global types from options:" << descriptor_options['type_list'] << log.endl

    # COMPUTE GRAPHS
    log << "Computing graphs ..." << log.endl
    log << "Descriptor: %s (fragment-based: %s)" % (descriptor_type, fragment_based) << log.endl
    log << json.dumps(descriptor_options, indent=2, sort_keys=True) << log.endl
    h5.attrs['options'] = json.dumps(options)
    h5.attrs['options.descriptor'] = json.dumps(descriptor_options)
    h5_graphs = h5.create_group('/graphs')
    # Create chunks to reduce memory storage
    def chunk_list(l, n):
        return [ l[i:i+n] for i in xrange(0, len(l), n) ]
    chunks = chunk_list(configs, 30)
    if n_procs < 2:
        log << "Computing graphs, single-threaded ..." << log.endl
        for chunk in chunks:
            graphs = []
            for config in chunk:
                graph = pygraph.mp_compute_graph(
                    config=config,
                    fragment_based=fragment_based,
                    descriptor_type=descriptor_type,
                    descriptor_options=descriptor_options,
                    log=log)
                graphs.append(graph)
            log << "Save chunk to h5 ..." << log.endl
            for g in graphs:
                g.save_to_h5(h5_graphs)
    else:
        log << "Computing graphs, multi-threaded (n_procs=%d) ..." % n_procs << log.endl
        for chunk in chunks:
            graphs = soap.soapy.util.mp_compute_vector(
                kfct=pygraph.mp_compute_graph,
                g_list=chunk,
                n_procs=n_procs,
                fragment_based=fragment_based,
                descriptor_type=descriptor_type,
                descriptor_options=descriptor_options,
                log=soap.soapy.momo.OSIO())
            log << "Save chunk to h5 ..." << log.endl
            for g in graphs:
                g.save_to_h5(h5_graphs)
            del graphs
    log << "Graphs complete." << log.endl
    
    # SAVE NAMES, CLASS LABELS
    labels = np.zeros((len(h5_graphs),), dtype=[('idx','i8'),('tag','a32')])
    for g in h5_graphs.iteritems():
        idx = int(g[0])
        tag = g[1].attrs['label']
        labels[idx] = (idx, tag)
    h5_labels = h5.create_group('labels')
    h5_labels.create_dataset('label_mat', data=labels)

    if cmdline_options.kernel:
        # COMPUTE KERNEL
        basekernel_type = options["basekernel"]["type"]
        basekernel_options = options["basekernel"][basekernel_type]
        basekernel = pygraph.BaseKernelFactory[basekernel_type](basekernel_options)
        topkernel_type = options["topkernel"]["type"]
        topkernel_options = options["topkernel"][topkernel_type]
        topkernel = pygraph.TopKernelFactory[topkernel_type](topkernel_options, basekernel)

        t_in = datetime.datetime.now()
        if n_procs < 2:
            kmat = np.zeros((len(h5_graphs),len(h5_graphs)), dtype='float32')
            if mp_kernel_block_size > 0:
                blocks = matrix_blocks_by_row(np.arange(len(h5_graphs)), block_size=mp_kernel_block_size, upper_triangular=True)
                for row_block in blocks:
                    if len(row_block) < 1: continue
                    # LOAD ROW ITEMS
                    rows = list(row_block[0][0])
                    g_rows = [ pygraph.Graph().load_from_h5(h5_graphs['%06d' % i]) for i in rows ]
                    # PREPROCESS ROWS
                    for g in g_rows:
                        topkernel.preprocess(g, log)
                    for block in row_block: 
                        # LOAD COL ITEMS
                        cols = list(block[1])
                        g_cols = [ pygraph.Graph().load_from_h5(h5_graphs['%06d' % i]) for i in cols ]
                        # PREPROCESS COLS
                        for g in g_cols:
                            topkernel.preprocess(g, log)
                        #print rows
                        #print cols
                        #raw_input('...')
                        for idx, i in enumerate(rows):
                            for jdx, j in enumerate(cols):
                                if j < i: continue
                                kmat[i,j] = topkernel.compute(g_rows[idx], g_cols[jdx], log)
            else:
                # LOAD
                graphs = []
                for i in range(len(h5_graphs)):
                    g = pygraph.Graph().load_from_h5(h5_graphs['%06d' % i])
                    graphs.append(g)
                # PREPROCESS
                for i in range(len(graphs)):
                    topkernel.preprocess(graphs[i], log)
                # KERNEL
                for i in range(len(graphs)):
                    for j in range(i, len(graphs)):
                        kmat[i,j] = topkernel.compute(graphs[i], graphs[j], log)
            kmat = kmat + kmat.T
            np.fill_diagonal(kmat, kmat.diagonal()*0.5)
        else:
            log << "Computing kernel, multi-threaded (n_procs=%d) ..." % n_procs << log.endl
            import multiprocessing as mp
            import functools as fct
            import itertools
            # Split kernel into blocks
            batch_size = n_procs*4
            block_size = mp_kernel_block_size
            if mp_preload:
                mp_graphs = []
                for i in range(len(h5_graphs)):
                    g = pygraph.Graph().load_from_h5(h5_graphs['%06d' % i])
                    mp_graphs.append(g)
            else:
                mp_graphs = h5_graphs
            blocks = matrix_blocks(np.arange(len(mp_graphs)), block_size=block_size, upper_triangular=True)
            if mp_hdf5_read_parallel:
                log << "Creating mp_hdf5_read_parallel kfct_primed" << log.endl
                h5.close()
                jobs = blocks
                kfct_primed = fct.partial(
                    pygraph.mp_compute_kernel_block_hdf5,
                    kernel=topkernel,
                    log=soap.soapy.momo.OSIO(),
                    dtype_result='float32',
                    h5_file=h5_file)
            else:
                jobs = (read_graph_block(mp_graphs, block, log) for block in blocks)
                # Kernel function, primed
                kfct_primed = fct.partial(
                    pygraph.mp_compute_kernel_block, 
                    kernel=topkernel, 
                    log=soap.soapy.momo.OSIO(), 
                    dtype_result='float32')
            # Compute
            kmat_blocks = []
            if mp_hdf5_read_parallel:
                job_batches = chunk_list(jobs, batch_size)
                # Pool
                for batch_idx, batch in enumerate(job_batches):
                    pool = mp.Pool(n_procs)
                    # Map
                    log << "Starting batch %d/%d ..." % (batch_idx+1, len(job_batches)) << log.endl
                    kmat_blocks_batch = pool.map(kfct_primed, batch)
                    log << "Done with batch %d/%d." % (batch_idx+1, len(job_batches)) << log.endl
                    # Collect
                    kmat_blocks.extend(kmat_blocks_batch)
                    # Close & join
                    pool.close()
                    pool.join()
            else:
                while True:
                    # Pool
                    pool = mp.Pool(n_procs)
                    log << "Starting batch ..." << log.endl
                    # Map
                    #kmat_blocks_batch = pool.imap(kfct_primed, itertools.islice(jobs, batch_size), chunksize=1)
                    kmat_blocks_batch = pool.map(kfct_primed, itertools.islice(jobs, batch_size))
                    # Collect
                    kmat_blocks_batch = list(kmat_blocks_batch)
                    if kmat_blocks_batch:
                        kmat_blocks.extend(kmat_blocks_batch)
                    else:
                        log << "Slice done, break." << log.endl
                        break
                    log << "Slice done." << log.endl
                    # Close & join
                    pool.close()
                    pool.join()
            if mp_hdf5_read_parallel:
                h5 = h5py.File(h5_file, 'a')
                mp_graphs = h5['graphs']
            # Store results
            kmat = np.zeros((len(mp_graphs),len(mp_graphs)), dtype='float32')
            for block, kmat_block in zip(blocks, kmat_blocks):
                r0 = block[0][0]
                r1 = block[0][-1]+1
                c0 = block[1][0]
                c1 = block[1][-1]+1
                kmat[r0:r1,c0:c1] = kmat_block
            kmat = kmat + kmat.T
            np.fill_diagonal(kmat, kmat.diagonal()*0.5)
        t_out = datetime.datetime.now()
        dt = t_out - t_in
        log << "Time for kernel computation: %s" % dt << log.endl

        # SAVE KERNEL
        h5_kernel = h5.create_group('kernel')
        h5_kernel.create_dataset('kernel_mat', data=kmat)
        print kmat
    h5.close()

if __name__ == "__main__":

    # Parallelization parameters
    # How many processors?                  n_procs
    # How many tasks per batch?             batch_size
    # How many tasks read upon demand?      chunk_size
    # Kernel matrix block size per task?    mp_kernel_block_size x mp_kernel_block_size
    
    # Command-line options
    log = soap.soapy.momo.osio
    log.Connect()
    log.AddArg('folder', typ=str, help="Data folder as execution target")
    log.AddArg('config_file', typ=str, help="xyz-trajectory file")
    log.AddArg('types_compile', typ=bool, help="Whether or not to compile particle types from dataset or use those specified in options file")
    log.AddArg('label_key', typ=str, help="Key storing unique identifier in <config_file>")
    log.AddArg('options', typ=str, help="Options file (json)")
    log.AddArg('hdf5_out', typ=str, help="Output hdf5 file name")
    log.AddArg('select', typ=int, default=-1, help="Actives to select")
    log.AddArg('n_procs', typ=int, default=1, help="Number of processors")
    log.AddArg('mp_kernel_block_size', typ=int, default=-1, help="Linear block size for kernel computation")
    log.AddArg('kernel', typ=bool, default=True, help="Whether or not to compute kernel")
    cmdline_options = log.Parse()
    json_options = soap.soapy.util.json_load_utf8(open(cmdline_options.options))

    # Run
    log.cd(cmdline_options.folder)
    soap.silence()
    run(log=log, cmdline_options=cmdline_options, json_options=json_options)
    log.root()

