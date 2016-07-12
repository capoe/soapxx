import numpy as np

def shannon_entropy(K, norm=True, eps=1e-20):
    k = K.flatten()
    s = -np.sum(k*np.log(k+eps))
    if norm: s = s/(-0.5*np.log(0.5)*k.shape[0])
    return s

def kernel_statistics(K, triu=True, full=False):
    if triu:
        triu_idcs = np.triu_indices(K.shape[0], 1)
        kmat = K[triu_idcs]
    else:
        kmat = K.flatten()
    avg = np.average(kmat)
    std = np.std(kmat)
    med = np.median(kmat)
    ent = shannon_entropy(kmat, norm=True)
    kmin = np.min(kmat)
    kmax = np.max(kmat)
    if full: 
        return avg, std, med, ent, kmin, kmax
    return avg, std, med, ent
        
