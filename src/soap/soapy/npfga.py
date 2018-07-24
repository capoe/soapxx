import numpy as np
import soap

# TODO Clean this module up, split onto shorter functions
# TODO Generic function for calculating cumulative distributions

def generate_graph(
        features_with_props, 
        uop_list, 
        bop_list, 
        unit_min_exp, 
        unit_max_exp):
    assert len(uop_list) == len(bop_list)
    fgraph_options = soap.Options()
    fgraph_options.set("unit_min_exp", unit_min_exp)
    fgraph_options.set("unit_max_exp", unit_max_exp)
    fgraph = soap.FGraph(fgraph_options)
    for f in features_with_props:
        fgraph.addRootNode(f[0], f[1], f[2], f[3], f[4])
    for lidx in range(len(uop_list)):
        fgraph.addLayer(uop_list[lidx], bop_list[lidx])
    fgraph.generate()
    return fgraph

def calculate_null_distribution(
        fgraph, 
        rand_IX_list, 
        rand_Y, 
        options, 
        log,
        file_out=False):
    npfga_dtype = rand_IX_list[0].dtype
    rand_covs = np.zeros((len(rand_IX_list), len(fgraph)), dtype=npfga_dtype)
    rand_Y = rand_Y.reshape((-1,1))
    for i, rand_IX in enumerate(rand_IX_list):
        log << log.back << "Randomized control, instance" << i << log.flush
        rand_covs[i,:] = fgraph.applyAndCorrelate(rand_IX, rand_Y, str(npfga_dtype))[:,0]
    log << log.endl
    # Dimensions and threshold
    n_channels = len(fgraph)
    n_samples = rand_covs.shape[0]
    p_threshold = 1. - options.tail_fraction
    i_threshold = int(p_threshold*n_samples+0.5)
    log << "Tail contains %d samples" % (n_samples-i_threshold) << log.endl
    # Random-sampling convariance matrix
    # Rows -> sampling instances
    # Cols -> feature channels
    rand_cov_mat = np.copy(rand_covs)
    rand_cov_mat = np.abs(rand_cov_mat)
    # Sort covariance observations for each channel
    rand_covs = np.abs(rand_covs)
    rand_covs = np.sort(rand_covs, axis=0)
    # Cumulative distribution for each channel
    rand_cum = np.ones((n_samples,1), dtype=npfga_dtype)
    rand_cum = np.cumsum(rand_cum, axis=0)
    rand_cum = (rand_cum-0.5) / rand_cum[-1,0]
    rand_cum = rand_cum[::-1,:]
    if file_out: np.savetxt('out_sis_channel_cov.hist', np.concatenate((rand_cum, rand_covs), axis=1))
    # Establish threshold for each channel
    thresholds = rand_covs[-int((1.-p_threshold)*n_samples),:]
    thresholds[np.where(thresholds < 1e-2)] = 1e-2
    t_min = np.min(thresholds)
    t_max = np.max(thresholds)
    t_std = np.std(thresholds)
    t_avg = np.average(thresholds)
    log << "Channel-dependent thresholds: min avg max +/- std = %1.2f %1.2f %1.2f +/- %1.4f" % (
        t_min, t_avg, t_max, t_std) << log.endl
    # Peaks over threshold: calculate excesses for random samples
    log << "Calculating excess for random samples" << log.endl
    pots = rand_covs[i_threshold:n_samples,:]
    rand_exs_mat = np.zeros((n_samples,n_channels), dtype=npfga_dtype)
    for s in range(n_samples):
        log << log.back << "- Sample %d/%d" % (s+1, n_samples) << log.flush
        rand_cov_sample = rand_cov_mat[s]
        exs = -np.average((pots+1e-10-rand_cov_sample)/(pots+1e-10), axis=0)
        rand_exs_mat[s,:] = exs
    # Random excess distributions
    rand_exs = np.sort(rand_exs_mat, axis=1) # n_samples x n_channels
    rand_exs_cum = np.ones((n_channels,1), dtype=npfga_dtype) # n_channels x 1
    rand_exs_cum = np.cumsum(rand_exs_cum, axis=0)
    rand_exs_cum = (rand_exs_cum-0.5) / rand_exs_cum[-1,0]
    rand_exs_cum = rand_exs_cum[::-1,:]
    rand_exs_avg = np.average(rand_exs, axis=0)
    rand_exs_std = np.std(rand_exs, axis=0)
    # Rank distributions
    rand_exs_rank = np.sort(rand_exs, axis=0) # n_samples x n_channels
    rand_exs_rank = rand_exs_rank[:,::-1]
    rand_exs_rank_cum = np.ones((n_samples,1), dtype=npfga_dtype) # n_channels x 1
    rand_exs_rank_cum = np.cumsum(rand_exs_rank_cum, axis=0)
    rand_exs_rank_cum = (rand_exs_rank_cum-0.5) / rand_exs_rank_cum[-1,0]
    rand_exs_rank_cum = rand_exs_rank_cum[::-1,:]
    if file_out: np.savetxt('out_exs_rank_rand.txt', np.concatenate([ rand_exs_rank_cum, rand_exs_rank ], axis=1))
    # ... Histogram
    if file_out: np.savetxt('out_exs_rand.txt', np.array([rand_exs_cum[:,0], rand_exs_avg, rand_exs_std]).T)
    log << log.endl
    return pots, rand_exs_cum, rand_exs_rank, rand_exs_rank_cum

def rank_ptest(
        tags,
        covs,
        exs, 
        exs_cum, 
        rand_exs_rank, 
        rand_exs_rank_cum, 
        q_threshold, # lower confidence threshold
        file_out=False):
    n_channels = exs.shape[0]
    idcs_sorted = np.argsort(exs)[::-1]
    p_first_list = np.zeros((n_channels,))
    p_rank_list = np.zeros((n_channels,))
    for rank, c in enumerate(idcs_sorted):
        # Calculate probability to observe feature given its rank
        ii = np.searchsorted(rand_exs_rank[:,rank], exs[c])
        if ii >= rand_exs_rank_cum.shape[0]:
            p0 = rand_exs_rank_cum[ii-1,0]
            p1 = 0.0
        elif ii <= 0:
            p0 = 1.0
            p1 = rand_exs_rank_cum[ii,0]
        else:
            p0 = rand_exs_rank_cum[ii-1,0]
            p1 = rand_exs_rank_cum[ii,0]
        p_rank = 0.5*(p0+p1)
        # Calculate probability to observe feature as highest-ranked
        ii = np.searchsorted(rand_exs_rank[:,0], exs[c])
        if ii >= rand_exs_rank_cum.shape[0]:
            p0 = rand_exs_rank_cum[ii-1,0]
            p1 = 0.0
        elif ii <= 0:
            p0 = 1.0
            p1 = rand_exs_rank_cum[ii,0]
        else:
            p0 = rand_exs_rank_cum[ii-1,0]
            p1 = rand_exs_rank_cum[ii,0]
        p_first = 0.5*(p0+p1)
        p_first_list[c] = p_first
        p_rank_list[c] = p_rank
    if file_out: 
        np.savetxt('out_exs_phys.txt', np.array([
            exs_cum[::-1,0], 
            exs[idcs_sorted], 
            p_rank_list[idcs_sorted], 
            p_first_list[idcs_sorted]]).T)
    cleaned = np.where(p_first_list <= 1.-q_threshold)[0]
    cleaned = sorted(cleaned, key=lambda c: (1.-p_first_list[c])*-np.abs(covs[c]))
    q_values = [ 1.-p_first_list[c] for c in range(n_channels) ]
    return cleaned, q_values, p_first_list[idcs_sorted[0]], exs[idcs_sorted[0]]

