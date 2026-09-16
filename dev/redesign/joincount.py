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
from scipy.sparse.csgraph import connected_components

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


def carrier_concentration(A, carriers):
    """
    Share of connected carriers that sit in ONE lineage region of the kNN graph.

    Carriers are linked by DIRECT kNN edges (graph built without the variant). Linking
    through overlapping neighbourhoods was tried first and chains across lineages:
    on MDA_PT variants spread over several barcode clones scored 0.93, like clone
    markers. With direct edges: spread 0.35, mid-clone markers 0.83, whole large
    clones 0.80, tiny clones 1.00, subclones 0.52 (removing those is harmless).
    Carriers linked to nobody (stray calls) are ignored. Returns largest connected
    group / carriers in groups of size >= 2 (NaN if no group).
    """
    C = np.flatnonzero(carriers)
    if C.size < 2:
        return np.nan
    _, lab = connected_components(A[C][:, C], directed=False)
    sizes = np.bincount(lab)
    grouped = sizes[sizes >= 2]
    return float(grouped.max()/grouped.sum()) if grouped.size else np.nan


def carrier_nonrandomness(X, B, cell_depth=None, k=15, alpha=0.01, n_perm=999, n_strata=5, seed=0,
                          n_rounds=1, ref=None, min_concentration=None, return_concentration=False):
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

    Scattered noise is neither. A variant passes a test if its p <= alpha/2 (Bonferroni
    over the two tests).

    Concentration (min_concentration, default None = off). Variants carried by SEVERAL
    lineages are non-random too -- dense in each -- so the join count keeps them, but in
    the tree they tie unrelated clones together and fragment large ones (MDA_PT oracle
    ablation: removing 12 such variants took ARI 0.61 -> 0.75, large clones recovered
    4/10 -> 8/10). A variant kept by the join count must therefore have most of its
    linked carriers in one lineage region (carrier_concentration >= min_concentration).
    Variants kept by exclusivity are exempt: sole-marker carriers have no other calls,
    so their neighbourhoods -- and hence their concentration -- are undefined.
    min_concentration=None disables the criterion. It is OFF by default: baked into the
    QC at 0.5 it removed 11/19 spread variants on MDA_PT but also a whole-clone marker
    and spread variants that carry real signal, with no ARI gain on MDA_PT (AD>0
    carriers 0.609 -> 0.603, cells 61% -> 50%), none on MDA_clones / MDA_lung, and a
    loss on 50-clone simulated polytomies (0.75 -> 0.66).

      keep = (p_excl <= alpha/2) | ((p_join <= alpha/2) & (concentration >= min_concentration))

    "The rest of the data" is only as good as the variants it is made of: when most
    candidates are noise (MDA_PT: ~60%), cells are placed and counted by noise calls
    and real markers lose power. With n_rounds > 1, the graph and the exclusivity
    counts of the next round are built only from the variants kept in the previous
    one, and EVERY variant is retested against them. Stops when the kept set repeats.

    Returns keep (bool), p_join, p_excl [, concentration].
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
        p_join = np.ones(m); p_excl = np.ones(m); conc = np.full(m, np.nan)
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
            conc[j] = carrier_concentration(A, Bb[:, j])
        by_join = p_join <= alpha/2
        if min_concentration is not None:
            by_join &= np.nan_to_num(conc, nan=0.0) >= min_concentration
        keep = (p_excl <= alpha/2) | by_join
        if any((keep == s_).all() for s_ in seen) or keep.sum() < 2:
            break
        seen.append(keep.copy())
        ref = keep.copy()
    return (keep, p_join, p_excl, conc) if return_concentration else (keep, p_join, p_excl)


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
