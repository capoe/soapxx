import numpy as np
import soap
from soap.soapy.math import zscore

class PyFGraph(object):
    def __init__(self, fgraph):
        self.fgraph_ = fgraph
        self.fnodes = []
        self.fnode_map = {}
        self.generations = None
        self.root_nodes = None
        self.map_generations = None
        self.extract()
        self.rank()
    def extract(self):
        self.fnodes = [ PyFNode(_) for _ in self.fgraph_ ]
        self.root_nodes = filter(lambda f: f.is_root, self.fnodes)
        self.map_generations = {}
        self.generations = sorted(list(set([ f.generation for f in self.fnodes ])))
        self.map_generations = { g: [] for g in self.generations }
        for f in self.fnodes:
            self.map_generations[f.generation].append(f)
        print "FGraph of size %d" % len(self.fnodes)
        for g in self.generations:
            print "  %4d nodes at generation %d" % (
                len(self.map_generations[g]), g)
        self.fnode_map = { f.expr: f for f in self.fnodes }
        for f in self.fnodes: f.resolveParents(self.fnode_map)
        return
    def rank(self, key=lambda f: np.abs(f.cov*f.confidence)):
        scores = [ key(f) for f in self.fnodes ]
        scores_cum = np.cumsum(sorted(scores))
        ranked = sorted(self.fnodes, key=key)
        for idx, r in enumerate(ranked):
            r.rank = scores_cum[idx]/scores_cum[-1]*key(r)
            print r.rank
            #r.rank = float(idx+1)/len(ranked)
        return

class PyFNode(object):
    def __init__(self, fnode):
        self.fnode_ = fnode
        self.parents_ = self.fnode_.getParents()
        self.parents = []
        self.is_root = fnode.is_root
        self.generation = fnode.generation
        self.expr = fnode.expr
        self.cov = fnode.cov
        self.confidence = fnode.q
        self.rank = -1
    def resolveParents(self, fnode_map):
        self.parents = [ fnode_map[p.expr] for p in self.parents_ ]

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
    rand_exs_rank_cum = np.ones((n_samples,1), dtype=npfga_dtype) # n_samples x 1
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
    q_values = [ 1.-p_first_list[c] for c in range(n_channels) ]
    q_values_nth = [ 1.-p_rank_list[c] for c in range(n_channels) ]
    return q_values, q_values_nth

class PyFGraphStats(object):
    def __init__(self, tags, covs, exs, q_values, q_values_nth, null_exs):
        self.tags = tags
        self.covs = covs
        self.exs = exs
        self.q_values = np.array(q_values)
        self.q_values_nth = np.array(q_values_nth)
        self.null_exs = null_exs # matrix of size SxC (samples by channels)
        self.order = np.argsort(-np.abs(self.covs))
        self.evaluateTopNode()
    def evaluateTopNode(self):
        self.top_idx = self.order[0]
        self.top_tag = self.tags[self.top_idx]
        self.top_cov = self.covs[self.top_idx]
        self.top_q = self.q_values[self.top_idx]
        self.top_exceedence = self.exs[self.top_idx]
        self.top_avg_null_exceedence = np.average(self.null_exs[:,0])
        self.top_rho_harm = np.abs(self.top_cov)/(1.+self.top_exceedence)
        self.top_avg_null_cov = self.top_cov*(1.+self.top_avg_null_exceedence)/(1.+self.top_exceedence)
        self.percentiles = np.arange(0,110,10)
        self.top_avg_null_exc_percentiles = [ np.percentile(self.null_exs[:,0], p) for p in self.percentiles ]
        self.top_avg_null_cov_percentiles = [ self.top_cov*(1.+e)/(1.+self.top_exceedence) for e in self.top_avg_null_exc_percentiles ]
        return
    def calculateExpectedCovExceedence(self, rank, rank_null=0):
        idx = self.order[rank]
        cov = np.abs(self.covs[idx])
        exc = self.exs[idx]
        avg_null_exceedence = np.average(self.null_exs[:,rank_null])
        rho_harm = cov/(1.+exc) # harmonic tail cov for this channel
        avg_null_cov = cov*(1.+avg_null_exceedence)/(1.+exc)
        return cov-avg_null_cov
    def summarize(self, log):
        log << "Top-ranked node: '%s'" % (self.top_tag) << log.endl
        log << "  [phys]   cov  = %+1.4f     exc  = %+1.4f     q = %1.4f" % (self.top_cov, self.top_exceedence, self.top_q) << log.endl
        log << "  [null]  <cov> = %+1.4f    <exc> = %+1.4f" % (self.top_avg_null_cov, self.top_avg_null_exceedence) << log.endl
        log << "Percentiles"
        for idx, p in enumerate(self.percentiles):
            log << "  [null] p = %1.2f  <cov>_p = %+1.4f  <exc>_p = %+1.4f" % (
                0.01*p, self.top_avg_null_cov_percentiles[idx], self.top_avg_null_exc_percentiles[idx]) << log.endl
        return

def run_npfga(fgraph, IX, Y, rand_IX_list, rand_Y, options, log):
    # C = #channels, S = #samples, T = tail length
    pots_TxC, ranks_Cx1, null_exs_SxC, ranks_Sx1 = soap.soapy.npfga.calculate_null_distribution(
        fgraph=fgraph,
        rand_IX_list=rand_IX_list,
        rand_Y=rand_Y.reshape((-1,1)),
        options=options,
        log=log)
    covs = fgraph.applyAndCorrelate(
        IX,
        Y.reshape((-1,1)),
        str(IX.dtype))[:,0]
    covs_abs = np.abs(covs)
    exs = -np.average((pots_TxC+1e-10-covs_abs)/(pots_TxC+1e-10), axis=0)
    tags = [ f.expr for f in fgraph ]
    q_values, q_values_nth = soap.soapy.npfga.rank_ptest(
        tags=tags,
        covs=covs_abs,
        exs=exs,
        exs_cum=ranks_Cx1,
        rand_exs_rank=null_exs_SxC,
        rand_exs_rank_cum=ranks_Sx1,
        q_threshold=options.confidence_threshold)
    for fidx, fnode in enumerate(fgraph):
        fnode.q = q_values[fidx]
        fnode.cov = covs[fidx]
    return tags, covs, q_values, PyFGraphStats(tags, covs, exs, q_values, q_values_nth, null_exs_SxC)

def run_cov_decomposition(fgraph, fnode, IX, Y, rand_IX_list, rand_Y, ftag_to_idx, log):
    roots = fnode.getRoots()
    root_tags = [ (r.tag[2:-1] if r.tag.startswith("(-") else r.tag) for r in roots ]
    y_norm = (Y-np.average(Y))/np.std(Y)
    # Marginalization tuples
    input_tuples = []
    for size in range(len(root_tags)+1):
        input_tuples.extend(soap.soapy.math.find_all_tuples_of_size(size, root_tags))
    covs = []
    # Calculated expectations
    for tup in input_tuples:
        rand_covs = []
        xs = []
        ys = []
        for i in range(len(rand_IX_list)):
            rand_IX = np.copy(IX)
            for tag in tup:
                rand_IX[:,ftag_to_idx[tag]] = rand_IX_list[i][:,ftag_to_idx[tag]]
            rand_x = fgraph.evaluateSingleNode(fnode, rand_IX, str(rand_IX.dtype))
            rand_x_norm = (rand_x[:,0] - np.average(rand_x[:,0]))/np.std(rand_x[:,0])
            xs = xs + list(rand_x[:,0])
            ys = ys + list(y_norm)
            rand_cov = np.dot(rand_x_norm, y_norm)/y_norm.shape[0]
            #rand_covs.append(np.abs(rand_cov))
            rand_covs.append(rand_cov)
        xs = np.array(xs)
        ys = np.array(ys)
        xs = (xs - np.average(xs))/np.std(xs)
        ys = (ys - np.average(ys))/np.std(ys)
        covs.append(np.average(rand_covs))
        #covs.append(np.dot(xs,ys)/ys.shape[0])
        print "Marginalizing over", tup, "=>", covs[-1]
    # Solve linear system for decomposition
    col_names = [ ":".join(tup)+":" for tup in input_tuples ]
    # Setup linear system A*x = b and solve for x (margin terms)
    A = np.ones((len(col_names),len(col_names))) # coeff_matrix
    for i, tup in enumerate(input_tuples):
        for j, col_name in enumerate(col_names):
            zero = False
            for tag in tup:
                if tag+":" in col_name:
                    zero = True
                    break
            if zero: A[i,j] = 0.0
    print "Coefficient matrix"
    print A
    b = covs # 
    x = np.linalg.solve(A,b)
    order = np.argsort(x)[::-1]
    for i in order:
        print "%-50s = %+1.4f" % ("<rho[%s]>" % (":".join(input_tuples[i])), x[i])
    return

def run_factor_analysis(mode, fgraph, fnode, IX, Y, rand_IX_list, rand_Y, ftag_to_idx, log):
    roots = fnode.getRoots()
    root_tags = [ (r.tag[2:-1] if r.tag.startswith("(-") else r.tag) for r in roots ]
    # Covariance for true instantiation
    x = fgraph.evaluateSingleNode(fnode, IX, str(IX.dtype))
    x_norm = (x[:,0]-np.average(x[:,0]))/np.std(x)
    y_norm = (Y-np.average(Y))/np.std(Y)
    cov = np.dot(x_norm, y_norm)/y_norm.shape[0]
    # Null dist
    rand_covs_base = []
    for i in range(len(rand_IX_list)):
        rand_x = fgraph.evaluateSingleNode(fnode, rand_IX_list[i], str(rand_IX_list[i].dtype))
        rand_x_norm = (rand_x[:,0] - np.average(rand_x[:,0]))/np.std(rand_x[:,0])
        rand_cov = np.dot(rand_x_norm, y_norm)/y_norm.shape[0]
        rand_covs_base.append(np.abs(rand_cov))
    rand_covs_base = np.array(sorted(rand_covs_base))
    np.savetxt('out_null.txt', np.array([np.arange(len(rand_covs_base))/float(len(rand_covs_base)), rand_covs_base]).T)
    # Analyse each factor
    factor_map = {}
    for root_tag in root_tags:
        rand_covs = []
        for i in range(len(rand_IX_list)):
            rand_IX = np.copy(IX)
            if mode == "randomize_this":
                rand_IX[:,ftag_to_idx[root_tag]] = rand_IX_list[i][:,ftag_to_idx[root_tag]]
            elif mode == "randomize_other":
                for tag in root_tags:
                    if tag == root_tag: pass
                    else: rand_IX[:,ftag_to_idx[tag]] = rand_IX_list[i][:,ftag_to_idx[tag]]
            else: raise ValueError(mode)
            rand_x = fgraph.evaluateSingleNode(fnode, rand_IX, str(rand_IX.dtype))
            rand_x_norm = (rand_x[:,0] - np.average(rand_x[:,0]))/np.std(rand_x[:,0])
            rand_cov = np.dot(rand_x_norm, y_norm)/y_norm.shape[0]
            rand_covs.append(np.abs(rand_cov))
        # Test
        rand_covs = np.array(sorted(rand_covs))
        np.savetxt('out_%s_%s.txt' % (mode.split("_")[1], root_tag), np.array([np.arange(len(rand_covs))/float(len(rand_covs)), rand_covs]).T)
        rank = np.searchsorted(rand_covs, np.abs(cov))
        q_value = float(rank)/len(rand_covs)
        factor_map[root_tag] = { "q_value": q_value, "min_cov": rand_covs[0], "max_cov": rand_covs[-1] }
    for factor, r in factor_map.iteritems():
        log << "%-50s c=%+1.4f q%-20s = %+1.4f  [random min=%1.2f max=%1.2f]" % (
            fnode.expr, cov, "(%s)" % factor, r["q_value"], r["min_cov"], r["max_cov"]) << log.endl
    return factor_map

def represent_graph_2d(fgraph):
    # POSITION NODES
    dphi_root = 2*np.pi/len(fgraph.map_generations[0])
    radius_root = 1.0
    radius_scale = 2.5
    for gen in fgraph.generations:
        nodes = fgraph.map_generations[gen]
        print "Positioning generation", gen
        for idx, node in enumerate(nodes):
            if gen == 0:
                node.radius = radius_root
                node.phi = idx*dphi_root
                print "r=%1.2f phi=%1.2f %s" % (node.radius, node.phi, node.expr)
            elif len(node.parents) == 1:
                # Unary case
                par = node.parents[0]
                node.radius = (1.+gen)**2*radius_root + radius_scale*(
                    np.abs(node.cov*node.confidence))*radius_root
                node.phi = par.phi + (
                    np.abs(node.cov*node.confidence))*dphi_root/node.radius
            elif len(node.parents) == 2:
                # Binary case
                p1 = node.parents[0]
                p2 = node.parents[1]
                phi_parents = sorted([ p.phi for p in node.parents ])
                dphi = phi_parents[1]-phi_parents[0]
                if dphi <= np.pi:
                    node.phi = phi_parents[0] + 0.5*dphi
                else:
                    node.phi = (phi_parents[1] + 0.5*(2*np.pi - dphi)) % (2*np.pi)
                node.radius = (1.+gen)**2*radius_root + radius_scale*(
                    np.abs(node.cov*node.confidence))*radius_root
                node.phi = node.phi + (
                    np.abs(node.cov*node.confidence))*dphi_root/node.radius
    # LINKS BETWEEN NODES
    def connect_straight(f1, f2):
        x1 = f1.radius*np.cos(f1.phi)
        y1 = f1.radius*np.sin(f1.phi)
        x2 = f2.radius*np.cos(f2.phi)
        y2 = f2.radius*np.sin(f2.phi)
        #w = np.abs(f2.cov*f2.confidence)
        w = f2.rank
        return [ [x1,y1,w], [x2,y2,w] ]
    def connect_arc(f0, f1, f2, samples=15):
        x0 = f0.radius*np.cos(f0.phi)
        y0 = f0.radius*np.sin(f0.phi)
        x1 = f1.radius*np.cos(f1.phi)
        y1 = f1.radius*np.sin(f1.phi)
        x2 = f2.radius*np.cos(f2.phi)
        y2 = f2.radius*np.sin(f2.phi)
        #w = np.abs(f2.cov*f2.confidence)
        w = f2.rank
        r1 = ((x1-x0)**2+(y1-y0)**2)**0.5
        r2 = ((x2-x0)**2+(y2-y0)**2)**0.5
        phi1 = np.arctan2(y1-y0, x1-x0)
        phi2 = np.arctan2(y2-y0, x2-x0)
        if phi1 < 0.: phi1 = 2*np.pi + phi1
        if phi2 < 0.: phi2 = 2*np.pi + phi2
        phi_start = phi1
        dphi = phi2-phi1
        if dphi >= np.pi:
            dphi = 2*np.pi - dphi
            phi_end = phi_start-dphi
        elif dphi <= -np.pi:
            dphi = 2*np.pi + dphi
            phi_end = phi_start+dphi
        else:
            phi_end = phi_start + dphi
        coords = []
        for i in range(samples):
            phi_i = phi_start + float(i)/(samples-1)*(phi_end-phi_start)
            rad_i = r1 + float(i)/(samples-1)*(r2-r1)
            x_i = x0 + rad_i*np.cos(phi_i)
            y_i = y0 + rad_i*np.sin(phi_i)
            coords.append([x_i, y_i, w])
        return coords
    curves = []
    for fnode in fgraph.fnodes:
        if len(fnode.parents) == 1:
            curves.append(connect_straight(fnode.parents[0], fnode))
        elif len(fnode.parents) == 2:
            curves.append(connect_arc(fnode.parents[0], fnode.parents[1], fnode))
            curves.append(connect_arc(fnode.parents[1], fnode.parents[0], fnode))
        else: pass
    # Sort curves so important ones are in the foreground
    curves = sorted(curves, key=lambda c: c[0][-1])
    return fgraph, curves





