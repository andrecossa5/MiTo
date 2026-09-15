"""Minimal remnant of the kNN caller: the pieces still imported elsewhere."""
import numpy as np


def _cosine_D(X, G, sq, j=None, skip=False):
    """Cosine distances with variant j optionally held out, via rank-1 Gram downdate."""
    if j is None or skip:
        Gm, s = G, sq
    else:
        xj = X[:, j]; Gm, s = G - np.outer(xj, xj), sq - xj**2
    nj = np.sqrt(np.maximum(s, 0.)); den = np.outer(nj, nj)
    D = np.ones_like(Gm)
    np.divide(Gm, den, out=D, where=den > 1e-12)
    D = np.clip(1 - D, 0., 2.)
    D[nj <= 1e-12, :] = 1.; D[:, nj <= 1e-12] = 1.
    return D


def _phi(ad, cov):
    """Dispersion index of alt counts against a binomial at the pooled rate."""
    p = float(ad.sum()/max(cov.sum(), 1e-9)); p = min(max(p, 1e-9), 1-1e-9)
    return float(np.sum((ad-cov*p)**2/np.maximum(cov*p*(1-p), 1e-12)))/max(len(ad)-1, 1)
