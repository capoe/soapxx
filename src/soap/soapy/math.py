import numpy as np

def shannon_entropy(K, norm=True, eps=1e-20):
    k = K.flatten()
    s = -np.sum(k*np.log(k+eps))
    if norm: s = s/(-0.5*np.log(0.5)*k.shape[0])
    return s
    
