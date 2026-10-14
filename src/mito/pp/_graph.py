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


def _cosine_from_gram(G, n_rows, n_cols):
    """1 - G / (|x_i| |x_l|), with distance 1 to and from all-zero profiles, in [0, 2]."""

    den = np.outer(n_rows, n_cols)
    D = np.ones_like(G)
    np.divide(G, den, out=D, where=den>1e-12)
    D = np.clip(1-D, 0., 2.)
    D[n_rows<=1e-12,:] = 1.
    D[:,n_cols<=1e-12] = 1.

    return D


def _topk_rows(D, k):
    """
    Column indices of the k smallest values of each row of D, ties broken by the lower
    column index (so that the graph does not depend on how it was computed).
    """

    kth = np.partition(D, k-1, axis=1)[:,k-1:k]
    below = D<kth
    need = k-below.sum(1, keepdims=True)
    at = D==kth
    sel = below | (at & (np.cumsum(at, axis=1)<=need))
    rows, cols = np.nonzero(sel)

    return cols.reshape(D.shape[0], k)


class LeaveOneOutKNN:
    """
    Cosine kNN graphs of the cells, each with one variant held out, without recomputing
    the cell x cell distances for every variant.

    Holding out variant j changes the profile of the cells that carry it (X[:, j] != 0)
    and nothing else: the distance between two non-carriers is the same with or without
    j. So the cells' nearest neighbours are computed once, in row blocks (top k + margin,
    never a dense n x n matrix), and for each variant only

    * the carriers' rows are recomputed (c x n),
    * the non-carriers whose neighbourhood a carrier enters or leaves are re-ranked over
      their cached neighbours and the carriers, with an exact fallback to the full row
      when the cache holds fewer than k non-carriers.

    The graphs are identical to those built from the full distance matrix
    (`cosine_distances(X, j=j)` -> k nearest), at O(n (c + k)) time and O(n (k + margin))
    memory per variant, against O(n^2).

    Ties are frequent - two cells sharing no variant are at distance 1 from each other -
    and are broken by a random order of the cells drawn from `seed`: by cell index, the
    first cells would become everyone's neighbour.
    """

    def __init__(self, X, k, margin=None, block=1024, seed=0):
        X = np.asarray(X, dtype=float)
        # Work in a random order of the cells, so that "lower index first" on ties is a
        # random tie-break; results are mapped back to the input order
        self.perm = np.random.default_rng(seed).permutation(X.shape[0])
        self.inv = np.argsort(self.perm)
        self.X = X[self.perm]
        n = self.X.shape[0]
        self.n = n
        self.k = min(k, n-1)
        L = min(n-1, self.k + (margin if margin is not None else 2*self.k))
        self.L = L
        self.sq = (self.X**2).sum(1)
        self.norm = np.sqrt(self.sq)

        self.idx = np.empty((n, L), dtype=np.int64)
        self.dist = np.empty((n, L))
        for s in range(0, n, block):
            r = np.arange(s, min(n, s+block))
            D = _cosine_from_gram(self.X[r] @ self.X.T, self.norm[r], self.norm)
            D[np.arange(r.size), r] = np.inf
            top = _topk_rows(D, L)
            d = np.take_along_axis(D, top, 1)
            order = np.lexsort((top, d), axis=1)            # by distance, then index
            self.idx[r] = np.take_along_axis(top, order, 1)
            self.dist[r] = np.take_along_axis(d, order, 1)
        self.base = self.idx[:,:self.k].copy()
        self.kth = self.dist[:,self.k-1].copy()

    def neighbours(self, j, return_dist=False):
        """
        (n, k) neighbour indices of every cell in the graph without variant j, and their
        distances if `return_dist`.
        """

        out = self._neighbours(j, return_dist)
        idx, dist = out if return_dist else (out, None)
        idx = self.perm[idx][self.inv]
        return (idx, dist[self.inv]) if return_dist else idx

    def _neighbours(self, j, return_dist=False):
        xj = self.X[:,j]
        C = np.flatnonzero(xj)
        if C.size == 0:
            return (self.base, self.dist[:,:self.k].copy()) if return_dist else self.base
        k = self.k
        norm_new = self.norm.copy()
        norm_new[C] = np.sqrt(np.maximum(self.sq[C] - xj[C]**2, 0.))

        # Carriers: their whole row changes
        G_C = self.X[C] @ self.X.T - np.outer(xj[C], xj)
        D_C = _cosine_from_gram(G_C, norm_new[C], norm_new)
        D_C[np.arange(C.size), C] = np.inf
        out = self.base.copy()
        out[C] = _topk_rows(D_C, k)
        dist = self.dist[:,:k].copy() if return_dist else None
        if return_dist:
            dist[C] = np.take_along_axis(D_C, out[C], 1)

        # Non-carriers whose neighbourhood may change: a carrier among their current
        # neighbours, or a carrier at least as close as their k-th neighbour
        is_carrier = np.zeros(self.n, bool)
        is_carrier[C] = True
        to_carrier = D_C.T                                  # (n, c): cell -> carrier
        affected = is_carrier[self.base].any(1) | (to_carrier.min(1)<=self.kth)
        affected &= ~is_carrier
        rows = np.flatnonzero(affected)
        if rows.size == 0:
            return (out, dist) if return_dist else out

        cache_idx = self.idx[rows]
        cache_d = np.where(is_carrier[cache_idx], np.inf, self.dist[rows])
        cand_idx = np.concatenate([cache_idx, np.broadcast_to(C, (rows.size, C.size))], 1)
        cand_d = np.concatenate([cache_d, to_carrier[rows]], 1)
        order = np.lexsort((cand_idx, cand_d), axis=1)[:,:k]
        out[rows] = np.take_along_axis(cand_idx, order, 1)
        if return_dist:
            dist[rows] = np.take_along_axis(cand_d, order, 1)

        # Exact fallback: fewer than k non-carriers in the cache, so a non-carrier outside
        # it could belong to the k nearest
        short = np.isfinite(cache_d).sum(1)<k
        if short.any():
            fb = rows[short]
            G = self.X[fb] @ self.X.T - np.outer(xj[fb], xj)
            D = _cosine_from_gram(G, norm_new[fb], norm_new)
            D[np.arange(fb.size), fb] = np.inf
            out[fb] = _topk_rows(D, k)
            if return_dist:
                dist[fb] = np.take_along_axis(D, out[fb], 1)

        return (out, dist) if return_dist else out


def adjacency_from_neighbours(idx, n):
    """Symmetric, unweighted adjacency from (n, k) neighbour indices."""

    rows = np.repeat(np.arange(n), idx.shape[1])
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
