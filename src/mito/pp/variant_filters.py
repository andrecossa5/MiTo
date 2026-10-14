"""
Read-level MT-SNVs filters: annotation, quality, candidate selection, known artefacts.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import binom, fisher_exact
from scipy.stats import chi2 as chi2_dist
from statsmodels.stats.multitest import multipletests

from mito.ut.positions import mask_mt_sites
from mito.ut.utils import load_common_dbSNP, load_edits_REDIdb, load_rsrs50

##


def _dense(X):
    """AD is sparse, DP is dense: read either."""
    return X.toarray() if hasattr(X, 'toarray') else np.asarray(X)


##


def filter_small_clones(
    afm: AnnData,
    column: str = 'GBC',
    min_cell_number: int = 10,
    copy: bool = False
    ) -> AnnData | None:
    """
    Retain only cells from groups in afm.obs[`column`] with at least `min_cell_number` cells.
    """

    afm = afm.copy() if copy else afm
    logging.info(f'Filtering cells from {column} groups with >={min_cell_number} cells')

    n0 = afm.shape[0]
    cell_counts = afm.obs.groupby(column).size()
    clones_to_retain = cell_counts[cell_counts>=min_cell_number].index
    test = afm.obs[column].isin(clones_to_retain).values
    afm._inplace_subset_obs(test)

    logging.info(f'Removed other {n0-afm.shape[0]} cells')
    logging.info(f'Retaining {afm.obs[column].unique().size} discrete categories (i.e., {column}) for the analysis.')

    return afm if copy else None


##


def annotate_vars(afm: AnnData, overwrite: bool = False):
    """
    Annotate MT-SNVs properties as in Weng et al., 2024, and Miller et al. 2022 before.
    Updates .var in place.
    """

    if afm.shape[0] == 0:
        raise ValueError(
            'Cannot annotate variants: the AFM has no cells left. '
            'Relax the cell filters (e.g. lower min_cell_number, or the coverage thresholds).'
        )

    # Columns this function derives itself, and therefore recomputes on overwrite.
    # Anything else in .var was put there by another step (e.g. per-variant QC statistics,
    # or per-lineage enrichment) and must survive re-annotation.
    _DERIVED = [
        'mean_af', 'mean_cov',
        'n_cells',
        'median_af_in_positives', 'mean_AD_in_positives', 'mean_DP_in_positives',
    ]

    if 'mean_af' in afm.var.columns:
        if not overwrite:
            return
        else:
            logging.info('Re-annotate variants in afm')
            afm.var = afm.var.drop(columns=_DERIVED, errors='ignore').copy()

    afm.var['mean_af'] = afm.X.mean(axis=0).A1

    if 'DP' in afm.layers:
        afm.var['mean_cov'] = _dense(afm.layers['DP']).mean(axis=0)

    afm.var['n_cells'] = (afm.X>0).sum(axis=0).A1   # cells with any alternative signal

    # Mean AF, AD and DP in +cells
    # NB: keep the AF values and the positivity mask separate. Using the mask in
    # place of the values makes median_af_in_positives collapse to a constant 1.
    AF = afm.X.toarray()
    positive = AF > 0
    with warnings.catch_warnings():        # all-negative variants yield empty slices
        warnings.simplefilter('ignore', RuntimeWarning)
        afm.var['median_af_in_positives'] = np.nanmean(
            np.where(positive, AF, np.nan), axis=0
        )
        afm.var['mean_AD_in_positives'] = np.nanmean(
            np.where(positive, _dense(afm.layers['AD']), np.nan), axis=0
        )
        afm.var['mean_DP_in_positives'] = np.nanmean(
            np.where(positive, _dense(afm.layers['DP']), np.nan), axis=0
        )
    del AF, positive


##


def filter_low_quality_variants(
    afm: AnnData,
    min_site_cov: int = 5,
    min_var_quality: int = 30,
    min_n_positive: int = 2,
    only_genes: bool = False,
    copy: bool = False
    ) -> AnnData | None:
    """
    Baseline MT-SNVs filter: site coverage, base call quality, positive cells, and
    (MAESTER) only sites in MT-genes. Sites with more than one alternative allele are
    dropped as well, since they cannot be genotyped independently.

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix, annotated with `mito.pp.annotate_vars`.
    min_site_cov : int, optional
        Minimum mean site coverage across cells. Default is 5.
    min_var_quality : int, optional
        Minimum mean base call quality across cells. Default is 30.
    min_n_positive : int, optional
        Minimum number of cells with AF > 0. Default is 2.
    only_genes : bool, optional
        Retain only MT-SNVs inside MT-gene bodies. This is the region a targeted assay
        (MAESTER) enriches; assays that cover the genome uniformly (mtscATAC, ReDeeM)
        would lose real variants, so it is off by default and `mito.pp.filter_afm` turns
        it on only for targeted AFMs. Default is False.
    copy : bool, optional
        Return a modified copy instead of updating `afm` in place. Default is False.

    Returns
    -------
    AnnData | None
        Updated AFM if `copy` is True, otherwise None.
    """

    afm = afm.copy() if copy else afm
    annotate_vars(afm)

    # What is tested depends on what the AFM carries, not on which assay produced it:
    # base-call quality is only available from read-level pre-processing, and the MT-gene
    # mask only applies when variants are annotated with their position.
    if only_genes and 'pos' in afm.var.columns:
        afm._inplace_subset_var(np.asarray(mask_mt_sites(afm.var['pos'])))

    test = (afm.var['mean_cov']>=min_site_cov) & (afm.var['n_cells']>=min_n_positive)
    if 'quality' in afm.var.columns:
        test &= afm.var['quality']>=min_var_quality
    else:
        logging.info('No per-variant base quality available: skipping the quality threshold.')
    afm._inplace_subset_var(test.values)

    # Exclude sites with more than one alt allele observed
    var_sites = afm.var_names.map(lambda x: x.split('_')[0])
    afm._inplace_subset_var((var_sites.value_counts()[var_sites]==1).values)

    # Exclude variants and cells not observed at all
    afm._inplace_subset_obs((afm.X>0).sum(axis=1).A1>0)
    afm._inplace_subset_var((afm.X>0).sum(axis=0).A1>0)

    return afm if copy else None


##


def filter_candidate_variants(
    afm: AnnData,
    min_cov: float = 5,
    min_var_quality: float = 30,
    min_frac_negative: float = 0.5,
    min_n_positive: int = 5,
    af_confident_detection: float = .01,
    min_n_confidently_detected: int = 2,
    min_mean_AD_in_positives: float = 1.25,
    min_mean_DP_in_positives: float = 10,
    copy: bool = False
    ) -> AnnData | None:
    """
    Select candidate MT-SNVs from their read-level statistics.

    Filters variants with:

    * at least `min_cov` mean site coverage (across cells)
    * at least `min_var_quality` mean base call quality (across cells)
    * at least n cells * `min_frac_negative` negative cells
    * at least `min_n_positive` cells with AF > 0
    * at least `min_n_confidently_detected` cells with AF >= `af_confident_detection`
    * at least `min_mean_AD_in_positives` mean AD in positive cells
    * at least `min_mean_DP_in_positives` mean DP in positive cells

    These thresholds only have to make a variant *worth testing*: whether its carriers
    are a clone is decided downstream, by genotyping and by the clonality filter. They
    are deliberately looser than the published MiTo defaults (mean DP in positives 25,
    negative fraction 0.2, confident AF 0.02), which discarded real markers of small
    clones before any test could see them.

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix.
    min_cov : float, optional
        Minimum mean site coverage. Default is 5.
    min_var_quality : float, optional
        Minimum mean base call quality. Default is 30.
    min_frac_negative : float, optional
        Minimum fraction of negative cells. Default is 0.5.
    min_n_positive : int, optional
        Minimum number of cells with AF > 0. Default is 5.
    af_confident_detection : float, optional
        AF threshold for a confident detection. Default is 0.01.
    min_n_confidently_detected : int, optional
        Cells required at that AF. Default is 2.
    min_mean_AD_in_positives : float, optional
        Minimum mean AD in positive cells. Default is 1.25.
    min_mean_DP_in_positives : float, optional
        Minimum mean DP in positive cells. Default is 10.
    copy : bool, optional
        Return a modified copy instead of updating `afm` in place. Default is False.

    Returns
    -------
    AnnData | None
        Updated AFM if `copy` is True, otherwise None.
    """

    afm = afm.copy() if copy else afm
    annotate_vars(afm, overwrite=True)
    # Kept local: these two counts only serve the thresholds below
    n_negative = afm.shape[0] - (afm.X>0).sum(axis=0).A1
    n_confident = (afm.X>=af_confident_detection).sum(axis=0).A1

    test = (
        (afm.var['mean_cov']>=min_cov) & \
        (n_negative>=min_frac_negative*afm.shape[0]) & \
        (afm.var['n_cells']>=min_n_positive) & \
        (n_confident>=min_n_confidently_detected) & \
        (afm.var['mean_AD_in_positives']>=min_mean_AD_in_positives) & \
        (afm.var['mean_DP_in_positives']>=min_mean_DP_in_positives)
    )
    if 'quality' in afm.var.columns:
        test &= afm.var['quality']>=min_var_quality

    afm._inplace_subset_var(test.values)

    return afm if copy else None


##


# rCRS regions of recurrent artefacts: the 303-315 and 513-525 homopolymers, the 3107N
# reference gap, and the 16182-16193 homopolymer (redeemR blacklist).
BLACKLIST_REGIONS = ((302, 315), (513, 525), (3105, 3109), (16182, 16187))


def _qvalue(p):
    """Storey q-values, with pi0 estimated at lambda = .5 (as redeemR, via qvalue)."""
    p = np.asarray(p, float)
    if p.size == 0:
        return p
    pi0 = min(1.0, np.mean(p>0.5) / 0.5)
    return np.minimum(multipletests(p, method='fdr_bh')[1] * pi0, 1.0)


def _binomial_gof(counts, depth):
    """
    Chi-square goodness of fit of a variant's per-cell alt counts to one binomial at the
    site's mean depth (redeemR `goodness_of_fit_test`): sequencing noise is binomial,
    a heteroplasmy carried by some cells only is not.
    """
    depth = max(int(depth), int(counts.max()))
    obs = np.bincount(counts, minlength=depth+1)[:depth+1]
    p_hat = counts.sum() / (counts.size * depth)
    exp = counts.size * binom.pmf(np.arange(depth+1), depth, p_hat)
    exp[exp==0] = 1e-318
    chi2 = ((obs-exp)**2 / exp).sum()
    return chi2, chi2_dist.sf(chi2, df=depth)


def overdispersion_stats(afm: AnnData, min_cell_coverage: float = 10, min_n_positive: int = 2) -> pd.DataFrame:
    """
    Per-variant statistics of the over-dispersion candidate stage (redeemR Filter-2, Weng
    et al., 2024), computed over the cells with mean MT coverage >= `min_cell_coverage`.

    Returns a table indexed by variant: carriers, carrier fraction among covered cells,
    homoplasmy flag, median depth, the chi-square statistic and its p- and q-value
    (variants not tested get statistic -inf, p- and q-value 1).
    """

    if 'mean_site_coverage' in afm.obs.columns:
        cov = afm.obs['mean_site_coverage'].values
    else:
        cov = _dense(afm.layers['DP']).mean(1)
    qualified = cov>=min_cell_coverage

    AD = afm.layers['AD'][qualified]
    AD = AD.tocsc() if hasattr(AD, 'tocsc') else AD
    DP = _dense(afm.layers['DP'])[qualified]
    carriers = (AD>0)
    n_carriers = np.asarray(carriers.sum(0)).ravel()
    n_covered = np.maximum((DP>0).sum(0), 1)
    carrier_frac = n_carriers / n_covered
    tab = pd.DataFrame({
        'n_carriers' : n_carriers, 'carrier_frac' : carrier_frac,
        'median_depth' : np.median(DP, axis=0), 'homoplasmic' : False,
        'stat' : -np.inf, 'p' : 1.0, 'q' : 1.0,
    }, index=afm.var_names)

    candidates = np.flatnonzero(n_carriers>=min_n_positive)
    if candidates.size == 0:
        return tab

    A = _dense(AD[:, candidates]).astype(np.int64)
    D = np.maximum(DP[:, candidates], A).astype(np.int64)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        af = np.where(A>0, A / np.maximum(D, 1), np.nan)
        mean_af = np.nanmean(af, 0)
        vmr = np.nanvar(af, 0, ddof=1) / mean_af
    homo = (carrier_frac[candidates]>0.75) & (mean_af>0.75) & (vmr<0.01)
    tab.iloc[candidates, tab.columns.get_loc('homoplasmic')] = homo

    # As redeemR: counts over the cells carrying any heteroplasmic candidate
    tested = ~homo
    rows = (A[:, tested]>0).any(1)
    stat = np.full(candidates.size, -np.inf)
    p = np.ones(candidates.size)
    for j in np.flatnonzero(tested):
        stat[j], p[j] = _binomial_gof(A[rows, j], D[:, j].mean())
    q = np.ones(candidates.size)
    q[tested] = _qvalue(p[tested])

    tab.iloc[candidates, tab.columns.get_loc('stat')] = stat
    tab.iloc[candidates, tab.columns.get_loc('p')] = p
    tab.iloc[candidates, tab.columns.get_loc('q')] = q

    return tab


def filter_overdispersed_variants(
    afm: AnnData,
    min_cell_coverage: float = 10,
    min_n_positive: int = 2,
    q_max: float = 0.05,
    min_median_depth: float = 5,
    max_carrier_frac: float = 0.75,
    blacklist: bool = True,
    ancestral: bool = True,
    copy: bool = False
    ) -> AnnData | None:
    """
    Select candidate MT-SNVs whose counts are over-dispersed across cells.

    The variant-level stages of redeemR Filter-2 (Weng et al., 2024), the input of
    MitoDrift, on any AFM with molecule counts (MAESTER, ReDeeM):

    * statistics over the cells with mean MT coverage >= `min_cell_coverage`
    * carried (AD > 0) by at least `min_n_positive` of them, and not homoplasmic
    * per-cell alt counts over-dispersed relative to one binomial at the site's mean
      depth (chi-square goodness of fit, Storey q <= `q_max`): sequencing errors spread
      evenly over cells, a heteroplasmy is concentrated in the cells that carry it
    * outside the blacklisted artefact regions, and not an ancestral RSRS variant
    * median depth >= `min_median_depth`, and carried by at most `max_carrier_frac` of
      the covered cells

    Unlike `mito.pp.filter_candidate_variants`, no floor on the number of carriers above
    two, nor on alt counts per carrier: markers of two-cell clones, seen once in each,
    stay testable. Statistics are stored in .var (`od_stat`, `od_q`).

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix, with AD and DP layers.
    min_cell_coverage : float, optional
        Mean MT coverage of the cells the statistics are computed on. Default is 10.
    min_n_positive : int, optional
        Minimum number of carriers. Default is 2.
    q_max : float, optional
        Maximum q-value of the over-dispersion test. Default is 0.05.
    min_median_depth : float, optional
        Minimum median depth at the site. Default is 5.
    max_carrier_frac : float, optional
        Maximum fraction of covered cells carrying the variant. Default is 0.75.
    blacklist : bool, optional
        Remove variants in `BLACKLIST_REGIONS`. Default is True.
    ancestral : bool, optional
        Remove the 50 RSRS-defining variants (MITOMAP RSRS50). Default is True.
    copy : bool, optional
        Return a modified copy instead of updating `afm` in place. Default is False.

    Returns
    -------
    AnnData | None
        Updated AFM if `copy` is True, otherwise None.
    """

    afm = afm.copy() if copy else afm
    tab = overdispersion_stats(afm, min_cell_coverage=min_cell_coverage, min_n_positive=min_n_positive)

    keep = (tab['n_carriers']>=min_n_positive) & ~tab['homoplasmic'] & (tab['q']<=q_max)
    keep &= (tab['median_depth']>=min_median_depth) & (tab['carrier_frac']<=max_carrier_frac)
    pos = afm.var['pos'].values if 'pos' in afm.var.columns else \
          afm.var_names.str.split('_').str[0].astype(int).values
    if blacklist:
        for start, end in BLACKLIST_REGIONS:
            keep &= ~((pos>=start) & (pos<=end))
    if ancestral:
        keep &= ~afm.var_names.isin(load_rsrs50())

    afm.var['od_stat'] = tab['stat'].values
    afm.var['od_q'] = tab['q'].values
    logging.info(f'Over-dispersed candidates: {int(keep.sum())}/{afm.shape[1]}')
    afm._inplace_subset_var(keep.values)

    return afm if copy else None


def cap_candidates(afm: AnnData, max_candidates: int = 1000, min_cell_coverage: float = 10,
                   copy: bool = False) -> AnnData | None:
    """
    Keep at most `max_candidates` MT-SNVs, the most over-dispersed ones (`od_stat`, from
    `mito.pp.filter_overdispersed_variants`, computed here if missing).
    """

    afm = afm.copy() if copy else afm
    if max_candidates is None or afm.shape[1]<=max_candidates:
        return afm if copy else None
    if 'od_stat' not in afm.var.columns:
        afm.var['od_stat'] = overdispersion_stats(afm, min_cell_coverage=min_cell_coverage)['stat'].values
    order = np.argsort(-afm.var['od_stat'].values, kind='stable')
    keep = np.zeros(afm.shape[1], bool)
    keep[order[:max_candidates]] = True
    logging.warning(
        f'{afm.shape[1]} candidate MT-SNVs: keeping the {max_candidates} most over-dispersed '
        f'(max_candidates). Raise it to test more, at a cost in time and memory.'
    )
    afm._inplace_subset_var(keep)

    return afm if copy else None


##


def filter_known_artefacts(
    afm: AnnData,
    dbsnp: bool = True,
    rna_edits: bool = True,
    copy: bool = False
    ) -> AnnData | None:
    """
    Remove MT-SNVs annotated as common germline variants (dbSNP) or RNA edits (REDIdb).

    Counts of removed variants are stored in .uns["known_artefacts"].
    """

    afm = afm.copy() if copy else afm
    d = {'n_dbSNP':0, 'n_REDIdb':0}

    if dbsnp:
        common = load_common_dbSNP()
        test = afm.var_names.isin(common)
        d['n_dbSNP'] = int(test.sum())
        logging.info(f'Exclude {d["n_dbSNP"]} common SNVs events (dbSNP)')
        afm._inplace_subset_var(~test)

    if rna_edits:
        edits = load_edits_REDIdb()
        test = afm.var_names.isin(edits)
        d['n_REDIdb'] = int(test.sum())
        logging.info(f'Exclude {d["n_REDIdb"]} common RNA editing events (REDIdb)')
        afm._inplace_subset_var(~test)

    afm.uns['known_artefacts'] = d

    return afm if copy else None


##


def compute_lineage_biases(
    afm: AnnData,
    lineage_column: str,
    target_lineage: str,
    alpha: float = .05
    ) -> pd.DataFrame:
    """
    Compute MT-SNVs enrichment scores for a given lineage category using Fisher's exact test.

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix, genotyped with `mito.pp.call_genotypes`.
    lineage_column : str
        Field in afm.obs containing the 'lineage' categorical variable.
    target_lineage : str
        The category in afm.obs[lineage_column] to test for MT-SNV enrichment.
    alpha : float, optional
        Family-wise error rate for p-value correction. Default is 0.05.

    Returns
    -------
    results : pd.DataFrame
        Prevalence, odds ratio and FDR of each MT-SNV in the target lineage.
    """

    if lineage_column not in afm.obs.columns:
        raise ValueError(f'{lineage_column} not present in cell metadata!')
    if 'bin' not in afm.layers:
        raise ValueError('Run mito.pp.call_genotypes before computing lineage biases.')

    muts = afm.var_names
    prevalences_array = np.zeros(muts.size)
    target_ratio_array = np.zeros(muts.size)
    oddsratio_array = np.zeros(muts.size)
    pvals = np.zeros(muts.size)

    G = afm.layers['bin'].toarray()
    for i in range(muts.size):

        test_mut = G[:,i] == 1
        test_lineage = afm.obs[lineage_column] == target_lineage
        n_mut_lineage = np.sum(test_mut & test_lineage)
        n_mut_no_lineage = np.sum(test_mut & ~test_lineage)
        n_no_mut_lineage = np.sum(~test_mut & test_lineage)
        n_no_mut_no_lineage = np.sum(~test_mut & ~test_lineage)
        prevalences_array[i] = n_mut_lineage / test_lineage.sum()
        target_ratio_array[i] = n_mut_lineage / test_mut.sum() if test_mut.sum() else np.nan

        oddsratio, pvalue = fisher_exact(
            [
                [n_mut_lineage, n_mut_no_lineage],
                [n_no_mut_lineage, n_no_mut_no_lineage],
            ],
            alternative='greater',
        )
        oddsratio_array[i] = oddsratio
        pvals[i] = pvalue

    pvals = multipletests(pvals, alpha=alpha, method="fdr_bh")[1]

    results = (
        pd.DataFrame({
            'prevalence' : prevalences_array,
            'perc_in_target_lineage' : target_ratio_array,
            'odds_ratio' : oddsratio_array,
            'FDR' : pvals,
            'lineage_bias' : -np.log10(pvals)
        }, index=muts
        )
        .sort_values('lineage_bias', ascending=False)
    )

    return results


##


def select_gt_enriched_variants(
    afm: AnnData,
    lineage_column: str = None,
    fdr_treshold: float = .1,
    n_enriched_groups: int = 2,
    copy: bool = False
    ) -> AnnData | None:
    """
    Select MT-SNVs significantly enriched in at most `n_enriched_groups` ground truth
    lineages (Fisher's exact test, FDR <= `fdr_treshold`), and the cells of those lineages.

    This is a benchmarking utility: it uses the ground truth to pick variants, and must
    not be part of a lineage inference run.
    """

    afm = afm.copy() if copy else afm
    if lineage_column is None or lineage_column not in afm.obs.columns:
        raise ValueError(f'{lineage_column} not available in afm.obs!')

    L = []
    lineages = afm.obs[lineage_column].dropna().unique()
    for target_lineage in lineages:
        logging.info(f'Computing variants enrichment for lineage {target_lineage}...')
        res = compute_lineage_biases(afm, lineage_column, target_lineage)
        L.append(res['FDR']<=fdr_treshold)

    df_enrich = pd.concat(L, axis=1)
    df_enrich.columns = lineages
    test = df_enrich.apply(lambda x: np.sum(x>0)>0 and np.sum(x>0)<=n_enriched_groups, axis=1)
    vois = df_enrich.loc[test].index.unique()
    id_lineages = df_enrich.loc[test].sum(axis=0).loc[lambda x: x>0].index.to_list()
    cells = afm.obs[lineage_column].loc[lambda x: x.isin(id_lineages)].index

    afm._inplace_subset_obs(afm.obs_names.isin(cells))
    afm._inplace_subset_var(afm.var_names.isin(vois))

    return afm if copy else None


##
