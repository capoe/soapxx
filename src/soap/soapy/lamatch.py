import numpy as np
import numpy.linalg as la
import soap
from .. import linalg as perm
import kernel as kern
import lagraph

def compare_graphs_average(g1, g2, options):
    # TODO Use KernelFunctionFactory here
    xi = options['basekernel']['kernel.xi']
    P1_avg = np.average(g1.P, axis=0)
    P1_avg = P1_avg / np.dot(P1_avg, P1_avg)**0.5
    P2_avg = np.average(g2.P, axis=0)
    P2_avg = P2_avg / np.dot(P2_avg, P2_avg)**0.5
    #print P1.shape, P2.shape, P1_avg.shape, P2_avg.shape
    return np.dot(P1_avg, P2_avg)**xi

def compare_graphs_global(g1, g2, options):
    # TODO Use KernelFunctionFactory here
    xi = options['basekernel']['kernel.xi']
    assert g1.P.shape[0] == 1 # Global descriptor computed?
    assert g2.P.shape[0] == 1 # Global descriptor computed?
    return np.dot(g1.P[0], g2.P[0])**xi

def compare_graphs_rematch(g1, g2, options):
    if options['graph']['hierarchical']:
        k = compare_graphs_rematch_hierarchical(g1, g2, options)
    else:
        k = compare_graphs_rematch_base(g1, g2, options)
    return k

def compare_graphs_rematch_base(g1, g2, options):
    # Kernel matrix of feature vectors
    kfunc_type = options['basekernel']['kernel.type']
    kfunc = kern.KernelFunctionFactory[kfunc_type](options['basekernel'])
    K12_base = kfunc.evaluate(g1.P, g2.P)
    # >>>>>> KERNEL-SPECIFIC <<<<<<<<<<<<<<<<<<<<<<<<<
    # Match
    gamma = options['lamatch']['gamma']
    k_rematch = perm.regmatch(K12_base, gamma, 1e-6)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    return k_rematch

def compare_graphs_rematch_hierarchical(g1, g2, options):

    # Read options   
    gamma = options['lamatch']['gamma']

    # Sizes and levels 
    n1 = g1.L.shape[0]
    n2 = g2.L.shape[0]
    n_levels = g1.n_levels
    assert g1.n_levels == g2.n_levels

    # Base kernel
    kfunc_type = options['basekernel']['kernel.type']
    kfunc = kern.KernelFunctionFactory[kfunc_type](options['basekernel'])
    K12_base = kfunc.evaluate(g1.P,g2.P)
    K12 = np.zeros((n1,n2), dtype='float64')
    assert K12_base.shape == K12.shape
    K12 = np.copy(K12_base)

    #print "Base", K12_base

    # Interlevel kernels
    for l in range(1, n_levels):
        # Compare all pairs of subgraphs of this level
        for i in range(n1):
            gsub_i = g1.subgraphs[l][i]
            idcs_i = gsub_i.idcs
            for j in range(n2):
                gsub_j = g2.subgraphs[l][j]
                idcs_j = gsub_j.idcs
                # Look-up kernel slice and re-match
                ksub = K12_base[idcs_i,:][:,idcs_j]
                # >>>>>>>>>>>> KERNEL-SPECIFIC <<<<<<<<<<<<<<
                kij = perm.regmatch(ksub, gamma, 1e-6)
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                K12[i,j] = kij
                #print ""
                #print "Level =", l, "i =", i, "j =", j
                #print "Li =", gsub_i.L
                #print "Lj =", gsub_j.L
                #print "idcs_i =", idcs_i
                #print "idcs_j =", idcs_j
                #print "ksub =", ksub
                #print "kij =", kij
                #print ""
                #raw_input('...')
        # Update base kernel
        K12_base = np.copy(K12)
        #print ""
        #print "New base kernel =", K12_base
        #print ""

    # >>>>>>>>>>> KERNEL-SPECIFIC <<<<<<<<<<<<<<<<<
    # Top-level comparison
    k_re_top = perm.regmatch(K12_base, gamma, 1e-6)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    return k_re_top

def optimize_rematch(graphs, options, write_out=False, log=None, verbose=False):
    if not options['lamatch']['optimize_rematch']: return
    if log: log << "Optimizing gamma based on %d graphs" % len(graphs) << log.endl
    # ETA-GAMMA PAIRS
    gammas = [ 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 2., 3., 4., 5. ]
    pairs = []
    for gamma in gammas:
        pairs.append((gamma,"__empty__"))
        pairs.append((None, None))
    if write_out: ofs = open('out.optimize_rematch.txt', 'w')
    # KERNEL STATISTICS TARGETS
    ent_target = 0.25/(-0.5*np.log(0.5)) # <- entropy of numbers uniformly distributed over [0,1]
    std_target = (1./12.)**0.5 # <- std deviation of numbers uniformly distributed over [0,1]
    med_target = 0.5
    if log:
        log << "Objective: med(target)=%+1.2f std(target)=%+1.2f ent(target)=%+1.2f" % (\
            med_target, std_target, ent_target) << log.endl
    # COMPUTE MERIT (VIA STD-DEV) FOR EACH PAIR
    merits = []
    for gamma, empty in pairs:
        if gamma == empty == None:
            if write_out: ofs.write('\n')
            continue
        # Set
        options['lamatch']['gamma'] = gamma
        # Process
        kmat = soap.soapy.util.mp_compute_upper_triangle(
            kfct=compare_graphs_rematch,
            g_list=graphs,
            n_procs=4,
            n_blocks=1,
            log=None,
            tstart_twall=(None, None),
            backup=False,
            options=options)
        # Finalise kmat
        kmat = kmat + kmat.T
        np.fill_diagonal(kmat, 0.5*kmat.diagonal())
        # Analyse merit
        kmat_avg, kmat_std, kmat_med, kmat_ent, kmat_min, kmat_max = soap.soapy.math.kernel_statistics(kmat, triu=True, full=True)
        merit = -(kmat_std-std_target)**2 -(kmat_ent-ent_target)**2 -0.25*(kmat_med-med_target)**2
        merits.append((gamma, kmat_std, kmat_ent, kmat_med, merit))
        if log: log << 'gamma=%+1.2e avg/std %+1.2f %+1.2f ext %+1.2f %+1.2f ent %+1.2e med %+1.2e q %+1.2e' % \
            (gamma, kmat_avg, kmat_std, kmat_min, kmat_max, kmat_ent,  kmat_med, merit) << log.endl
        if log and verbose: log << kmat << log.endl
        if write_out:
            ofs.write('%+1.2e avg/std %+1.7e %+1.7e min/max %+1.7e %+1.7e ent %+1.2e\n' % \
                (gamma, kmat_avg, kmat_std, kmat_min, kmat_max, kmat_ent))
    merits = sorted(merits, key=lambda m: m[-1])
    if log: log << "Optimum for gamma=%+1.7e : std=%+1.4e ent=%+1.4e med=%+1.4e q=%+1.4e" % merits[-1] << log.endl
    options['lamatch']['gamma'] = merits[-1][0]
    return

GraphKernelFactory = {
'rematch' : compare_graphs_rematch,
'average': compare_graphs_average,
'global': compare_graphs_global,
'laplacian': lagraph.compare_graphs_laplacian_kernelized
}

