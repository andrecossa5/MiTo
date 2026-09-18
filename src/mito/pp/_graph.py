"""
Cell-cell graph helpers shared by genotyping, variant QC and imputation.

Private module: every function here works on plain arrays, so that the stages that
need a leave-one-out view of the data (i.e., a graph built WITHOUT the variant under
test) can share one implementation.
"""

import numpy as np
from scipy.sparse import csr_matrix

##


def cosine_distances(X, G=None, sq=None, j=None):
    """
    Cosine distances between cells, optionally holding variant `j` out.

    The hold-out is a rank-1 downdate of the Gram matrix (G = X @ X.T) and of the
    squared norms, so a graph without variant j costs one outer product instead of a
    full recomputation. This is what makes the leave-one-out tests affordable.
    """

    X = np.asarray(X, dtype=float)
    G = X @ X.T if G is None else G
    sq = (X**2).sum(1) if sq is None else sq

    if j is not None:
        xj = X[:,j]
        G = G - np.outer(xj, xj)
        sq = sq - xj**2

    nj = np.sqrt(np.maximum(sq, 0.))
    den = np.outer(nj, nj)
    D = np.ones_like(G)
    np.divide(G, den, out=D, where=den>1e-12)
    D = np.clip(1-D, 0., 2.)
    D[nj<=1e-12,:] = 1.
    D[:,nj<=1e-12] = 1.

    return D


##


def knn_adjacency(D, k):
    """
    Symmetric, unweighted kNN adjacency from a dense distance matrix.
    """

    n = D.shape[0]
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    k = min(k, n-1)
    idx = np.argpartition(D, k, axis=1)[:,:k]
    rows = np.repeat(np.arange(n), k)
    A = csr_matrix((np.ones(rows.size), (rows, idx.ravel())), shape=(n,n))

    return ((A+A.T)>0).astype(float)


##


def knn_kernel(X, k=30, normalize=True):
    """
    Gaussian kNN kernel over cell profiles (the weighted counterpart of `knn_adjacency`).

    Each cell's k neighbours are weighted by exp(-(d/d_max)^2), so the nearest ones
    dominate. With normalize=True rows sum to 1, and W @ f is the neighbourhood mean
    of any per-cell quantity f.
    """

    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    D = cosine_distances(X)
    np.fill_diagonal(D, np.inf)
    k = min(k, n-1)
    idx = np.argpartition(D, k, axis=1)[:,:k]
    dist = np.take_along_axis(D, idx, 1)
    sig = np.maximum(dist.max(1, keepdims=True), 1e-9)
    W = np.zeros((n,n))
    np.put_along_axis(W, idx, np.exp(-(dist/sig)**2), 1)

    return W/np.maximum(W.sum(1, keepdims=True), 1e-12) if normalize else W


##
