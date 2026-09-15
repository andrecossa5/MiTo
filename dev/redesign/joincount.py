"""
Join-count test on a kNN graph: a local replacement for Moran's I with one p-value threshold.

  JC_j = x_j' A x_j   number of (weighted) kNN edges joining two carriers of j

Null: carriers placed at random (permutation of x_j over cells, same carrier count).
p = P(JC_null >= JC_obs). Keep if p <= alpha.

Unlike Moran's I the statistic only sums over carrier pairs on a SPARSE kNN graph:
carriers far from the others add nothing, and absences never contribute, so a
dense cluster is significant whatever happens elsewhere in the graph.
"""
import numpy as np
from scipy.sparse import csr_matrix

from caller import _cosine_D


def knn_adjacency(D, k):
    n = D.shape[0]
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    A = csr_matrix((np.ones(rows.size), (rows, idx.ravel())), shape=(n, n))
    A = ((A + A.T) > 0).astype(float)                     # symmetric kNN graph
    return A


def join_count_test(X, B, k=15, graph='loo', n_perm=999, seed=0, eps=0.1):
    """
    graph='loo'    graph rebuilt without variant j (no self-confirmation)
    graph='fixed'  one graph from all variants, j included
    graph='eps'    j included with weight eps (places only cells with no other signal)
    Returns observed JC, null mean, p-value per variant.
    """
    rng = np.random.default_rng(seed)
    Bb = B > 0
    n, m = Bb.shape
    G = X @ X.T
    sq = (X**2).sum(1)
    A_fixed = knn_adjacency(_cosine_D(X, G, sq), k) if graph == 'fixed' else None
    obs = np.full(m, np.nan); null_mean = np.full(m, np.nan); pv = np.ones(m)
    for j in range(m):
        x = Bb[:, j].astype(float)
        c = int(x.sum())
        if c < 2:
            continue
        if graph == 'fixed':
            A = A_fixed
        elif graph == 'loo':
            A = knn_adjacency(_cosine_D(X, G, sq, j), k)
        else:
            # j kept in the graph but down-weighted: it only places cells that have
            # nothing else (rank-1 update of the Gram matrix and norms)
            xj = X[:, j]
            Ge = G - (1 - eps**2)*np.outer(xj, xj)
            sqe = sq - (1 - eps**2)*xj**2
            A = knn_adjacency(_cosine_D(X, Ge, sqe), k)
        obs[j] = x @ (A @ x)
        R = np.zeros((n_perm, n))
        for p in range(n_perm):
            R[p, rng.choice(n, c, replace=False)] = 1.0
        null = np.einsum('pi,pi->p', R @ A, R)
        null_mean[j] = null.mean()
        pv[j] = (1 + (null >= obs[j]).sum())/(1 + n_perm)
    return obs, null_mean, pv


def _stratified_sets(strata, carriers, n_perm, rng):
    """Random cell sets with the same number of cells per stratum as `carriers`."""
    n = strata.size
    R = np.zeros((n_perm, n))
    for s in np.unique(strata):
        pool = np.flatnonzero(strata == s)
        c = int(carriers[pool].sum())
        if c == 0:
            continue
        for p in range(n_perm):
            R[p, rng.choice(pool, c, replace=False)] = 1.0
    return R


def carrier_nonrandomness(X, B, cell_depth=None, k=15, alpha=0.01, n_perm=999, n_strata=5, seed=0,
                          n_rounds=1, ref=None):
    """
    Keep a variant if its carriers are non-random with respect to the REST of the data.

    Two permutation tests, both leaving variant j out, sharing one null (random cell
    sets with the carrier count of j, drawn within cell-depth strata so that cell
    quality cannot masquerade as structure):

      join count    carriers joined by more kNN edges than random sets     (enrichment)
                    -> clones marked by several variants, recurrent variants
      exclusivity   carriers carry FEWER other calls than random sets       (depletion)
                    -> clones marked by j alone: their cells lack every other
                       clone's markers, while noise carriers are ordinary cells

    Scattered noise is neither. keep if min(p_join, p_excl) <= alpha/2 (Bonferroni over
    the two tests), so the rule is a single threshold.

    "The rest of the data" is only as good as the variants it is made of: when most
    candidates are noise (MDA_PT: ~60%), cells are placed and counted by noise calls
    and real markers lose power. With n_rounds > 1, the graph and the exclusivity
    counts of the next round are built only from the variants kept in the previous
    one, and EVERY variant is retested against them. Stops when the kept set repeats.

    Returns keep (bool), p_join, p_excl.
    """
    rng = np.random.default_rng(seed)
    Bb = B > 0
    n, m = Bb.shape
    if cell_depth is None:
        strata = np.zeros(n, int)
    else:
        qs = np.quantile(cell_depth, np.linspace(0, 1, n_strata + 1)[1:-1])
        strata = np.searchsorted(qs, cell_depth, side='right')
    ref = np.ones(m, bool) if ref is None else np.asarray(ref, bool).copy()   # variants defining "the rest"
    seen = []
    for _ in range(n_rounds):
        Xr = np.where(ref[None, :], X, 0.0)
        total_calls = (Bb & ref[None, :]).sum(1).astype(float)
        G = Xr @ Xr.T
        sq = (Xr**2).sum(1)
        p_join = np.ones(m); p_excl = np.ones(m)
        for j in range(m):
            x = Bb[:, j].astype(float)
            if x.sum() < 2:
                continue
            other = total_calls - (x if ref[j] else 0.0)
            D = _cosine_D(Xr, G, sq, j if ref[j] else None)
            A = knn_adjacency(D, k)
            R = _stratified_sets(strata, Bb[:, j], n_perm, rng)
            jc_obs = x @ (A @ x)
            jc_null = np.einsum('pi,pi->p', R @ A, R)
            p_join[j] = (1 + (jc_null >= jc_obs).sum())/(1 + n_perm)
            ex_obs = (x @ other)/x.sum()
            ex_null = (R @ other)/R.sum(1)
            p_excl[j] = (1 + (ex_null <= ex_obs).sum())/(1 + n_perm)
        keep = np.minimum(p_join, p_excl) <= alpha/2
        if any((keep == s_).all() for s_ in seen) or keep.sum() < 2:
            break
        seen.append(keep.copy())
        ref = keep.copy()
    return keep, p_join, p_excl


def morans_i_test(D, B, n_perm=999, seed=0):
    """Shipped-style Moran's I: dense weights W = 1 - D over all cell pairs, permutation p."""
    rng = np.random.default_rng(seed)
    W = 1.0 - D
    np.fill_diagonal(W, 0.0)
    S0 = W.sum()
    n, m = B.shape
    I = np.full(m, np.nan); pv = np.ones(m)
    for j in range(m):
        x = B[:, j].astype(float)
        if x.sum() < 2 or x.sum() > n - 2:
            continue
        z = x - x.mean()
        I[j] = n/S0*(z @ W @ z)/(z @ z)
        Z = np.stack([rng.permutation(z) for _ in range(n_perm)])
        null = n/S0*np.einsum('pi,pi->p', Z @ W, Z)/(z @ z)
        pv[j] = (1 + (null >= I[j]).sum())/(1 + n_perm)
    return I, pv
