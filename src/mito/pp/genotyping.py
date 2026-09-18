"""
Single-cell MT-SNVs genotyping: calls, signal-to-background filter and dropout imputation.
"""

import logging

import numpy as np
from anndata import AnnData
from scipy import stats
from scipy.sparse import csr_matrix

from ._graph import cosine_distances

##


def _dense(X):
    """Layers may be sparse (AD) or dense (DP)."""
    return X.toarray() if hasattr(X, 'toarray') else np.asarray(X)


##


def _counts(afm):
    """
    Alt and total counts as dense integer arrays. Coverage is floored at AD, so that a
    site whose coverage was computed on a different read set can never give AD > DP.
    """

    for layer in ('AD', 'DP'):
        if layer not in afm.layers:
            raise ValueError(
                f'Genotyping needs read counts in afm.layers["{layer}"]. This AFM has no '
                f'counts: provide genotypes in afm.layers["bin"] instead.'
            )

    AD = _dense(afm.layers['AD']).astype(np.int64)
    COV = np.maximum(_dense(afm.layers['DP']).astype(np.int64), AD)

    return AD, COV


##


def _error_rates(AD, COV, carrier):
    """
    Per-variant sequencing-error rate: pooled alt/total counts over NON-carrier cells,
    with a one-read pseudocount so a site that never shows an alt read has a finite rate.
    """

    bg_ad = np.where(carrier, 0, AD).sum(0)
    bg_cov = np.where(carrier, 0, COV).sum(0)

    return (bg_ad+1.0) / (bg_cov+1.0)


##


def call_genotypes(
    afm: AnnData,
    alpha: float = 1e-3,
    max_iter: int = 50,
    copy: bool = False
    ) -> AnnData | None:
    """
    Call MT-SNVs genotypes, testing each cell against the variant's own error rate.

    A cell carries variant j if its alternative reads are too many to come from
    sequencing error alone: P(X >= AD | COV, error rate) <= `alpha`, one-sided.
    The error rate of each variant is the pooled alt/total ratio over the cells NOT
    currently called carriers, re-estimated until the carrier set stops changing, so
    that a clone does not inflate the background of its own marker.

    No mixture model and no prior are involved. A binomial mixture is unidentifiable
    when most cells have AD = 0: the mixing weight drifts and the prior alone calls
    cells with zero alternative reads (61% of the calls on one of our test datasets).
    Thresholding AD > 0 instead counts single reads at error-prone sites, which is how
    noise variants at AF ~0.006 acquire carriers. Here a single read is enough at a
    clean site and not at a noisy one.

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix, with alternative counts in .layers["AD"] and site
        coverage in .layers["DP"].
    alpha : float, optional
        Significance of the per-cell binomial test. Default is 1e-3.
    max_iter : int, optional
        Maximum number of background re-estimations. Default is 50.
    copy : bool, optional
        Return a modified copy instead of updating `afm` in place. Default is False.

    Returns
    -------
    AnnData | None
        Updated AFM if `copy` is True, otherwise None. Adds .layers["bin"] and
        .layers["imputed"], and the .var columns "error_rate", "n_carriers",
        "n_imputed" and "prevalence".
    """

    afm = afm.copy() if copy else afm
    AD, COV = _counts(afm)

    carrier = np.zeros(AD.shape, bool)
    converged = False
    for i in range(max_iter):
        p0 = _error_rates(AD, COV, carrier)
        pval = np.where(
            AD>0, stats.binom.sf(AD-1, COV, np.clip(p0, 1e-12, 1-1e-12)[None,:]), 1.0
        )
        new = pval<=alpha
        if (new==carrier).all():
            converged = True
            break
        carrier = new

    if not converged:
        # The carrier sets only ever grow (the background can only shrink as carriers
        # leave it), so this is a truncation, not an oscillation: the calls are a subset
        # of the fixed point's.
        logging.warning(
            f'Genotyping: the background did not converge in max_iter={max_iter} iterations. '
            f'Calls are a subset of the fixed point; raise max_iter.'
        )

    B = carrier.astype(np.int8)
    afm.layers['bin'] = csr_matrix(B)
    afm.layers['imputed'] = csr_matrix(np.zeros(B.shape, dtype=np.int8))
    afm.var['error_rate'] = p0
    afm.var['n_carriers'] = B.sum(0)
    afm.var['n_imputed'] = 0
    afm.var['prevalence'] = B.sum(0) / max(B.shape[0], 1)
    afm.uns['genotyping'] = {
        'method':'binomial', 'alpha':alpha, 'max_iter':max_iter,
        'n_iter':i+1, 'converged':converged
    }

    logging.info(
        f'Genotyping (binomial, alpha={alpha}): {int(B.sum())} calls, '
        f'{(B.sum(1)>0).sum()} cells with at least one, {i+1} background iterations'
    )

    return afm if copy else None


##


def filter_low_signal_variants(
    afm: AnnData,
    min_snr: float = 10.0,
    copy: bool = False
    ) -> AnnData | None:
    """
    Drop MT-SNVs whose carriers do not stand above the variant's own error rate.

    snr = (median AF of the cells with at least one alternative read) / error rate.

    The median is taken over cells with reads, not over called cells: calls are exactly
    the cells that beat the background, so their AF is high by construction and every
    variant would pass. Over all cells with reads, a broad heteroplasmic variant -
    present at similar AF in cells of every clone - scores low, while a clone marker
    scores in the tens to thousands.

    This protects the four-gamete filter downstream: such a variant gets a random-looking
    subset of cells called, conflicts with everything, and the greedy resolution deletes
    the real markers of the largest clone instead of it.

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix, genotyped with `mito.pp.call_genotypes`.
    min_snr : float, optional
        Minimum signal-to-background ratio. Default is 10.
    copy : bool, optional
        Return a modified copy instead of updating `afm` in place. Default is False.

    Returns
    -------
    AnnData | None
        Updated AFM if `copy` is True, otherwise None. Adds the .var column "snr" and
        subsets the MT-SNVs.
    """

    afm = afm.copy() if copy else afm
    if 'error_rate' not in afm.var.columns:
        raise ValueError('Run mito.pp.call_genotypes before filtering on signal-to-background.')

    AD, _ = _counts(afm)
    X = afm.X.toarray()
    with np.errstate(all='ignore'):
        med = np.nanmedian(np.where(AD>=1, X, np.nan), axis=0)
    ratio = np.where(
        np.isfinite(med), med/np.maximum(afm.var['error_rate'].values, 1e-12), 0.0
    )
    afm.var['snr'] = ratio

    test = ratio>=min_snr
    logging.info(f'Remove {int((~test).sum())} MT-SNVs with signal-to-background <{min_snr}')
    afm._inplace_subset_var(test)

    return afm if copy else None


##


def impute_dropouts(
    afm: AnnData,
    k: int = 30,
    thr: float = 0.8,
    min_support: int = 1,
    copy: bool = False
    ) -> AnnData | None:
    """
    Impute genotype dropouts from each cell's neighbourhood.

    For every cell without a call at variant j, the weighted fraction of its k nearest
    neighbours carrying j is computed on a graph built WITHOUT j (leave-one-out, so the
    variant cannot vouch for itself), and the call is added if that fraction is at least
    `thr`. The cell must already carry `min_support` other calls: without it, cells with
    no evidence at all are handed a genotype by their neighbourhood alone.

    Imputation trades precision for coverage and its value is dataset-dependent: it
    recovered 14.6 points of assigned cells on one of our test datasets, was neutral on
    another and cost 0.04 ARI in simulations. It is therefore off by default in
    `mito.pp.filter_afm`.

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix, genotyped with `mito.pp.call_genotypes`.
    k : int, optional
        Neighbours per cell. Default is 30.
    thr : float, optional
        Minimum weighted fraction of carriers in the neighbourhood. Default is 0.8.
    min_support : int, optional
        Other calls required in the cell. Default is 1.
    copy : bool, optional
        Return a modified copy instead of updating `afm` in place. Default is False.

    Returns
    -------
    AnnData | None
        Updated AFM if `copy` is True, otherwise None. Updates .layers["bin"],
        .layers["imputed"] and the .var count columns.
    """

    afm = afm.copy() if copy else afm
    if 'bin' not in afm.layers:
        raise ValueError('Run mito.pp.call_genotypes before imputing dropouts.')

    X = afm.X.toarray()
    B = _dense(afm.layers['bin'])
    was_imputed = _dense(afm.layers['imputed'])>0
    n, m = X.shape

    G = X @ X.T
    sq = (X**2).sum(1)
    Bf = (B>0).astype(float)
    support = Bf.sum(1)
    P = np.zeros((n,m))
    for j in range(m):
        D = cosine_distances(X, G, sq, j)
        np.fill_diagonal(D, np.inf)
        kk = min(k, n-1)
        idx = np.argpartition(D, kk, axis=1)[:,:kk]
        dist = np.take_along_axis(D, idx, 1)
        sig = np.maximum(dist.max(1, keepdims=True), 1e-9)
        w = np.exp(-(dist/sig)**2)
        w = w/np.maximum(w.sum(1, keepdims=True), 1e-12)
        P[:,j] = (w*Bf[idx,j]).sum(1)

    other = support[:,None]-Bf
    add = (B==0) & (P>=thr) & (other>=min_support)
    B = np.where(add, 1, B).astype(np.int8)
    imputed = (was_imputed | add).astype(np.int8)

    afm.layers['bin'] = csr_matrix(B)
    afm.layers['imputed'] = csr_matrix(imputed)
    afm.var['n_imputed'] = imputed.sum(0)
    afm.var['prevalence'] = (B>0).sum(0) / max(n, 1)
    afm.uns['imputation'] = {'k':k, 'thr':thr, 'min_support':min_support, 'n_imputed':int(add.sum())}

    logging.info(f'Impute dropouts (k={k}, thr={thr}): {int(add.sum())} calls added')

    return afm if copy else None


##
