"""
Phylogenetic compatibility of the retained MT-SNVs (four-gamete / infinite-sites).
"""

import logging
from itertools import combinations

import numpy as np
from anndata import AnnData

##


def filter_incompatible_variants(
    afm: AnnData,
    min_gamete: int = 10,
    copy: bool = False
    ) -> AnnData | None:
    """
    Remove MT-SNVs that violate the infinite-sites assumption, greedily.

    Two variants are incompatible when all four gametes (1/1, 1/0, 0/1, 0/0) are observed
    in more than `min_gamete` cells each: under infinite sites no tree can carry both, so
    one of them is either recurrent, lost, or a genotyping artefact. The variant involved
    in the largest number of conflicts is dropped, and the conflict graph is recomputed,
    until no conflict remains.

    `min_gamete` is a tolerance, not a test: a handful of cells showing the fourth gamete
    is what dropouts and stray calls produce, and requiring exactly zero would remove
    almost every real marker.

    Which of the two conflicting variants to drop is decided by their signal over
    background (.var["snr"], from `mito.pp.filter_low_signal_variants`): among the
    variants still in conflict, the one standing least above its own error rate goes
    first, ties broken by conflict count and then by prevalence.

    Dropping by conflict count alone favours sparse noise variants over prevalent
    markers: on one of our test datasets it removed three markers carrying 613 of 1272
    genotyped cells, which then had no character left at all.

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix, genotyped with `mito.pp.call_genotypes`.
    min_gamete : int, optional
        Cells required in each of the four gametes for a conflict to count. Default is 10.
    copy : bool, optional
        Return a modified copy instead of updating `afm` in place. Default is False.

    Returns
    -------
    AnnData | None
        Updated AFM if `copy` is True, otherwise None. Adds the .var column "n_conflicts"
        and subsets the MT-SNVs.
    """

    afm = afm.copy() if copy else afm
    if 'bin' not in afm.layers:
        raise ValueError('Run mito.pp.call_genotypes before the compatibility filter.')

    B = afm.layers['bin'].toarray()>0
    B = B[B.sum(1)>=1]
    m = B.shape[1]

    inc = np.zeros((m,m), bool)
    for i, j in combinations(range(m), 2):
        x, y = B[:,i], B[:,j]
        if min(int((x&y).sum()), int((x&~y).sum()), int((~x&y).sum()))>min_gamete:
            inc[i,j] = inc[j,i] = True

    afm.var['n_conflicts'] = inc.sum(1)
    score = afm.var['snr'].values if 'snr' in afm.var.columns else np.zeros(m)
    score = np.where(np.isfinite(score), score, 0.0)
    prevalence = B.sum(0)

    alive = np.ones(m, bool)
    while True:
        idx = np.flatnonzero(alive)
        sub = inc[np.ix_(alive, alive)]
        if not sub.any():
            break
        conflicted = idx[sub.any(1)]
        n_conf = sub.sum(1)[sub.any(1)]
        # Drop the variant with the least signal over its own background; ties go to the
        # most conflicted, then to the least prevalent - a sparse variant carries less
        # information than a prevalent marker, and the order must not depend on the
        # column order of the AFM.
        order = np.lexsort((prevalence[conflicted], -n_conf, score[conflicted]))
        alive[conflicted[order[0]]] = False

    logging.info(f'Remove {int((~alive).sum())} MT-SNVs incompatible with an infinite-sites tree')
    afm.uns['compatibility'] = {'min_gamete':min_gamete, 'n_removed':int((~alive).sum())}
    afm._inplace_subset_var(alive)

    return afm if copy else None


##
