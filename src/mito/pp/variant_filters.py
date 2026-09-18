"""
Read-level MT-SNVs filters: annotation, quality, candidate selection, known artefacts.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import fisher_exact
from statsmodels.sandbox.stats.multicomp import multipletests

from mito.ut.positions import mask_mt_sites
from mito.ut.utils import load_common_dbSNP, load_edits_REDIdb

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
