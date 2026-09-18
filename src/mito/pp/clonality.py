"""
Variant QC on the cell kNN graph: keep only MT-SNVs whose carriers are non-random.
"""

import logging

import numpy as np
from anndata import AnnData

from ._graph import cosine_distances, knn_adjacency

##


def _stratified_sets(strata, carriers, n_perm, rng):
    """
    Random cell sets with the same number of cells per stratum as `carriers`, so that
    cell quality (coverage) cannot masquerade as lineage structure in the null.
    """

    n = strata.size
    R = np.zeros((n_perm, n))
    for s in np.unique(strata):
        pool = np.flatnonzero(strata==s)
        c = int(carriers[pool].sum())
        if c == 0:
            continue
        for p in range(n_perm):
            R[p, rng.choice(pool, c, replace=False)] = 1.0

    return R


##


def filter_non_clonal_variants(
    afm: AnnData,
    alpha: float = 0.05,
    k: int = 15,
    n_perm: int = 999,
    n_strata: int = 5,
    seed: int = 0,
    copy: bool = False
    ) -> AnnData | None:
    """
    Keep MT-SNVs whose carriers are non-random with respect to the rest of the data.

    Two permutation tests are run per variant, both on a kNN graph rebuilt WITHOUT the
    variant under test, and both against the same null (random cell sets of the same
    size, drawn within cell-coverage strata):

    * **join count**: carriers are joined by more kNN edges than random sets of cells
      would be (enrichment). This catches clones marked by several variants, and
      recurrent variants.
    * **exclusivity**: carriers carry *fewer other calls* than random cells do
      (depletion). This catches clones marked by a single variant, whose cells lack
      every other clone's markers - a case the join count has no power for.

    A variant is retained if either test gives p <= `alpha`/2 (Bonferroni over the two).
    Scattered noise is neither clustered nor exclusive, and fails both.

    This replaces the Moran's I filter of the published pipeline. Moran's I weights every
    cell pair and includes the variant under test in the distances, so variants confirm
    themselves: 96% of scattered simulated noise passed it. Excluding the variant and
    truncating the weights to a kNN graph rejects 98% of that noise while keeping 98% of
    multi-marker and 93% of sole-marker clonal variants.

    `alpha` sets the smallest clone that can be kept: a variant carried by very few cells
    cannot reach a small p-value in a permutation test, whatever its quality.

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix, genotyped with `mito.pp.call_genotypes`.
    alpha : float, optional
        Significance threshold, split over the two tests. Default is 0.05.
    k : int, optional
        Neighbours per cell in the graph. Default is 15.
    n_perm : int, optional
        Permutations per variant. Default is 999.
    n_strata : int, optional
        Cell-coverage strata for the null. Default is 5.
    seed : int, optional
        Random seed. Default is 0.
    copy : bool, optional
        Return a modified copy instead of updating `afm` in place. Default is False.

    Returns
    -------
    AnnData | None
        Updated AFM if `copy` is True, otherwise None. Adds the .var columns "p_join",
        "p_exclusive", "p_clonal" and "clonal", and subsets the MT-SNVs.
    """

    afm = afm.copy() if copy else afm
    if 'bin' not in afm.layers:
        raise ValueError('Run mito.pp.call_genotypes before the clonality filter.')

    rng = np.random.default_rng(seed)
    X = afm.X.toarray()
    B = afm.layers['bin'].toarray()>0
    n, m = B.shape

    DP = afm.layers['DP']
    depth = (DP.toarray() if hasattr(DP, 'toarray') else np.asarray(DP)).mean(1)
    if n_strata>1 and np.unique(depth).size>1:
        qs = np.quantile(depth, np.linspace(0, 1, n_strata+1)[1:-1])
        strata = np.searchsorted(qs, depth, side='right')
    else:
        strata = np.zeros(n, int)

    total_calls = B.sum(1).astype(float)
    G = X @ X.T
    sq = (X**2).sum(1)
    p_join = np.ones(m)
    p_excl = np.ones(m)

    for j in range(m):
        x = B[:,j].astype(float)
        if x.sum()<2:
            continue
        other = total_calls-x                        # calls of the OTHER variants
        A = knn_adjacency(cosine_distances(X, G, sq, j), k)
        R = _stratified_sets(strata, B[:,j], n_perm, rng)
        jc_obs = x @ (A @ x)
        jc_null = np.einsum('pi,pi->p', R @ A, R)
        p_join[j] = (1+(jc_null>=jc_obs).sum()) / (1+n_perm)
        ex_obs = (x @ other) / x.sum()
        ex_null = (R @ other) / R.sum(1)
        p_excl[j] = (1+(ex_null<=ex_obs).sum()) / (1+n_perm)

    p_clonal = 2*np.minimum(p_join, p_excl)
    test = p_clonal<=alpha
    afm.var['p_join'] = p_join
    afm.var['p_exclusive'] = p_excl
    afm.var['p_clonal'] = p_clonal
    afm.var['clonal'] = test
    afm.uns['clonality'] = {
        'alpha':alpha, 'k':k, 'n_perm':n_perm, 'n_strata':n_strata, 'seed':seed,
        'n_clonal':int(test.sum()), 'n_tested':int(m)
    }

    logging.info(
        f'Clonality filter (alpha={alpha}): retain {int(test.sum())}/{m} MT-SNVs '
        f'({int((p_join<=alpha/2).sum())} by join count, {int((p_excl<=alpha/2).sum())} by exclusivity)'
    )
    afm._inplace_subset_var(test)

    return afm if copy else None


##
