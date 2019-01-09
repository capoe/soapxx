import numpy as np

class CDF(object):
    def __init__(self, samples, transform=None, maxlen=None):
        self.samples = samples
        if transform != None:
            self.samples = transform(self.samples)
        if maxlen != None:
            np.random.shuffle(self.samples)
            self.samples = self.samples[0:maxlen]
        self.samples = np.sort(self.samples)
        self.length = len(self.samples)
        self.epsilon = 1./self.length
    def evaluate(self, x):
        upper_edge_idcs = np.searchsorted(self.samples, x)
        return (upper_edge_idcs+0.5)/(self.length+1)
    def evaluateComplement(self, x):
        return 1. - self.evaluate(x)

def find_all_tuples_of_size(n, array):
    combos = []
    if n == 0:
        combos.append([])
    elif n > len(array):
        pass
    elif n == len(array):
        combos.append(array)
    else:
        for i in range(len(array)-n+1):
            first_element = array[i]
            sub_combos = find_all_tuples_of_size(n-1, array[i+1:])
            for sub in sub_combos:
                tup = [first_element]+list(sub)
                combos.append(tup)
    return combos

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
        
def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0
        return c

def zscore(IX, ddof=0):
    X_mean = np.mean(IX, axis=0)
    X_std = np.std(IX, axis=0, ddof=ddof)
    return div0(IX-X_mean, X_std), X_mean, X_std

