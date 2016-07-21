#! /usr/bin/env python
import os
import sys
import numpy as np
import logging
import io

def pca_epsilon_threshold(IX):
    x_const = (1./IX.shape[1])**0.5
    eps = x_const*1e-5
    return eps

def pca_compute(IX, eps=0., log=None, norm_div_std=True, norm_sub_mean=True):
    """
    To check result consider:
    IX_norm_pca.T.dot(IX_norm_pca)/IX_norm_pca.shape[0]
    """
    # Normalize: mean, std
    if log: log << "PCA: Normalize ..." << log.endl
    if norm_sub_mean:
        IX_norm = IX - IX.mean(0)
    if norm_div_std:
        IX_norm = IX_norm/(IX.std(0)+eps)
    # Correlation matrix
    if log: log << "PCA: Correlate ..." << log.endl
    S = IX_norm.T.dot(IX_norm)/IX_norm.shape[0]
    # Diagonalize
    if log: log << "PCA: Eigenspace ..." << log.endl
    lambda_, U = np.linalg.eigh(S)
    idcs = lambda_.argsort()[::-1]
    lambda_ = lambda_[idcs]
    U = U[:,idcs]
    L = np.identity(lambda_.shape[0])*lambda_
    #print L-U.T.dot(S).dot(U) # <- should be zero
    #print S-U.dot(L).dot(U.T) # <- should be zero
    # Transform
    if log: log << "PCA: Transform ..." << log.endl
    IX_norm_pca = U.T.dot(IX_norm.T).T
    return IX_norm_pca, IX_norm, L, U

class PCA(object):
    def __init__(self):
        self.IX = None
        self.IX_norm = None
        self.X_mean = None
        self.X_std = None
        self.eps = None
        return
    def compute(self, IX, normalize_mean=True, normalize_std=True, prefix='out.pca', eps=0.):
        # Normalize
        self.IX = IX
        self.X_mean = IX.mean(0)
        self.X_std= IX.std(0)
        self.eps = eps
        if not normalize_mean:
            self.X_mean.fill(0.)
        if not normalize_std:
            self.X_std.fill(1.)
        self.IX_norm = (IX - self.X_mean)/(self.X_std+eps)
        # Correlation matrix / moment matrix
        Sigma = np.dot(self.IX_norm.T,self.IX_norm)/self.IX_norm.shape[0]
        eigvals, eigvecs = np.linalg.eigh(Sigma)        
        # Sort largest -> smallest eigenvalue
        idcs = eigvals.argsort()[::-1]
        eigvals = eigvals[idcs]
        eigvecs = eigvecs[:,idcs]
        eigvecs.transpose()
        # Store eigenvalues, eigenvectors
        self.eigvals = eigvals # eigvals[i] => i-th eigenvalue, where
        self.eigvecs = eigvecs # eigvecs[i] => i-th eigenvector
        eigvals_cumulative = np.cumsum(eigvals)
        np.savetxt('%s.sigma.txt' % prefix, Sigma)
        np.savetxt('%s.eigvals_cum.txt' % prefix, eigvals_cumulative/eigvals_cumulative[-1])
        return
    def expand(self, X, idx_cutoff=None):
        assert False
        eigvecs_sel = self.eigvecs[0:idx_cutoff] if idx_cutoff else self.eigvecs
        eigvals_sel = self.eigvals[0:idx_cutoff] if idx_cutoff else self.eigvals
        # Normalize
        X_norm = X-self.X_mean
        X_norm = X_norm/(self.X_std+self.eps)
        # Expand
        eigencoeffs = eigvecs_sel.dot(X_norm)
        # Reconstruct
        X_recon = eigencoeffs.dot(eigvecs_sel)
        X_recon = self.unnorm(X_recon)        
        return X_recon
    def unnorm(self, X):
        return X*self.X_std+self.X_mean
    def expandBlock(self, IX, idx_cutoff=None):
        eigvecs_sel = self.eigvecs[0:idx_cutoff] if idx_cutoff else self.eigvecs
        eigvals_sel = self.eigvals[0:idx_cutoff] if idx_cutoff else self.eigvals
        # Normalize
        IX_norm = IX-self.X_mean
        IX_norm = IX_norm/(self.X_std+self.eps)
        # Expand
        eigencoeffs = IX_norm.dot(eigvecs_sel.T)
        return eigencoeffs
    def reconstructEigenvec(self, idx):
        X_recon = self.eigvecs[idx]
        X_recon = self.unnorm(X_recon)
        return X_recon
    def reconstructEigenvecs(self):
        X_eig_recon = self.unnormBlock(self.eigvecs)
        return X_eig_recon
    def reconstructBlock(self, eigencoeffs):
        # Order such that: eigencoeffs[i] => eigencoefficients of sample i
        n_samples = eigencoeffs.shape[0]
        n_components = eigencoeffs.shape[1]        
        idx_cutoff = n_components
        eigvecs_sel = self.eigvecs[0:idx_cutoff]
        eigvals_sel = self.eigvals[0:idx_cutoff]     
        # Reconstruct   
        X_recon = eigencoeffs.dot(eigvecs_sel)
        X_recon = self.unnormBlock(X_recon)
        return X_recon
    def unnormBlock(self, IX):
        return IX*self.X_std+self.X_mean
    def dot(self, e1, e2):
        # e1, e1 => vectors with eigencoefficients of two samples
        assert e1.shape == e2.shape
        n_components = e1.shape[0]
        assert False and not "SENSE"

class IPCA(PCA):
    def compute(self, IX, normalize_mean=True, normalize_std=True):
        print "Compute invariant"
        # Normalize
        self.IX = IX
        self.X_mean = IX.mean(0)
        self.X_std= IX.std(0)
        if not normalize_mean:
            self.X_mean.fill(0.)
        if not normalize_std:
            self.X_std.fill(1.)
        self.IX_norm = (IX - self.X_mean)/self.X_std
        # Correlation matrix / moment matrix
        Sigma = np.dot(self.IX_norm.T,self.IX_norm)/self.IX_norm.shape[0]        
        # Invariance step
        d = (1./np.diag(Sigma))**0.5
        D = np.zeros(Sigma.shape)
        np.fill_diagonal(D, d)
        Sigma = D.dot(Sigma).dot(D)
        eigvals, eigvecs = np.linalg.eigh(Sigma)        
        # Sort largest -> smallest eigenvalue
        idcs = eigvals.argsort()[::-1]
        eigvals = eigvals[idcs]
        eigvecs = eigvecs[:,idcs]
        eigvecs.transpose()
        # Store eigenvalues, eigenvectors
        self.eigvals = eigvals # eigvals[i] => i-th eigenvalue, where
        self.eigvecs = eigvecs # eigvecs[i] => i-th eigenvector
        eigvals_cumulative = np.cumsum(eigvals)
        np.savetxt('out.pca.sigma.txt', Sigma)
        np.savetxt('out.pca.eigvals_cum.txt', eigvals_cumulative/eigvals_cumulative[-1])
        return




