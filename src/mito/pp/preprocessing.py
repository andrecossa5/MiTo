"""
Pre-process AFMs: cell filtering, and the MT-SNVs filtering pipeline (`filter_afm`).
"""

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd
from anndata import AnnData

from mito.ut.provenance import record
from mito.ut.utils import Timer

from .clonality import filter_non_clonal_variants
from .compatibility import filter_incompatible_variants
from .distances import compute_distances
from .genotyping import call_genotypes, filter_low_signal_variants, impute_dropouts
from .variant_filters import (
    annotate_vars,
    filter_candidate_variants,
    filter_known_artefacts,
    filter_low_quality_variants,
    filter_small_clones,
)

##


def filter_cells(
    afm: AnnData,
    cell_subset: Iterable[str] = None,
    cell_filter: str = 'filter1',
    nmads: int = 5,
    mean_cov_all: float = 20,
    median_cov_target: int = 25,
    min_perc_covered_sites: float = .75,
    copy: bool = False
    ) -> AnnData | None:
    """
    Filter cells from a MAESTER/RedeeM Allele Frequency Matrix.

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix.
    cell_subset : Iterable[str], optional
        Subset of cells to retain. Default is None.
    cell_filter : str, optional
        Cell filtering strategy. Options are:
        - "filter1": Filter cells based on mean MT-genome coverage (all sites).
        - "filter2": Filter cells based on median target MT-sites coverage and minimum percentage of target sites covered (MAESTER only).
        Default is "filter1".
    nmads : int, optional
        Number of Minimum Absolute Deviations to filter cells with high MT-library UMI counts. Default is 5.
    mean_cov_all : int, optional
        Minimum mean consensus (at least 3-supporting-reads) UMI coverage across the MT-genome per cell. Default is 20.
    median_cov_target : int, optional
        Minimum median UMI coverage at target MT-sites (only for MAESTER data). Default is 25.
    min_perc_covered_sites : float, optional
        Minimum fraction of MT target sites covered (only for MAESTER data). Default is 0.75.
    copy : bool, optional
        Return a modified copy instead of updating `afm` in place. Default is False.

    Returns
    -------
    AnnData | None
        Filtered AFM if `copy` is True, otherwise None.
    """

    afm = afm.copy() if copy else afm

    if cell_subset is not None:
        cells = set(cell_subset) & set(afm.obs_names)
        logging.info(f'Filter provided cell subset. Valid CBs: {len(cells)}')
        afm._inplace_subset_obs(afm.obs_names.isin(cells))

    logging.info(f'scLT system: {afm.uns.get("scLT_system", "unknown")}')
    n_in = afm.shape[0]
    params = {'cell_filter':cell_filter}

    # Which criterion applies is decided by the coverage metrics the AFM carries: the
    # target-panel columns only exist for assays that enrich a panel of MT sites.
    if cell_filter == 'filter1':
        x = afm.obs['mean_site_coverage']
        median = np.median(x)
        MAD = np.median(np.abs(x-median))
        test = (x>=mean_cov_all) & (x<=median+nmads*MAD)
        afm._inplace_subset_obs(test.values)
        logging.info(f'Filtered cells (mean MT-genome coverage >={mean_cov_all} and <={median+nmads*MAD:.2f}): {afm.shape[0]}')
        params.update({'nmads':nmads, 'mean_cov_all':mean_cov_all})

    elif cell_filter == 'filter2':
        needed = ['median_target_site_coverage', 'frac_target_site_covered']
        if not all(c in afm.obs.columns for c in needed):
            raise ValueError(
                f'Cell filter "filter2" needs the target-panel coverage metrics {needed} '
                f'in .obs, which this AFM does not have (they are written for targeted '
                f'assays only). Use cell_filter="filter1".'
            )
        test1 = afm.obs['median_target_site_coverage'] >= median_cov_target
        test2 = afm.obs['frac_target_site_covered'] >= min_perc_covered_sites
        afm._inplace_subset_obs((test1 & test2).values)
        logging.info(f'Filtered cells (median target site coverage >={median_cov_target}, covered sites >={min_perc_covered_sites}): {afm.shape[0]}')
        params.update({'median_cov_target':median_cov_target, 'min_perc_covered_sites':min_perc_covered_sites})

    else:
        logging.info(f'Skipping cell filters: {cell_filter} not available. Cells: {afm.shape[0]}')

    # Ensure each site has been observed from at least one cell
    afm._inplace_subset_var((afm.X>0).sum(axis=0).A1>0)
    record(afm, 'filter_cells', {**params, 'n_cells_in':int(n_in), 'n_cells_out':int(afm.shape[0])})

    return afm if copy else None


##


def filter_afm(
    afm: AnnData,
    *,
    # cells and grouping
    lineage_column: str = None,
    min_cell_number: int = 0,
    # 1. candidate MT-SNVs
    cand_min_site_cov: int = 5,
    cand_min_quality: int = 30,
    cand_min_frac_negative: float = 0.5,
    cand_min_n_positive: int = 5,
    cand_af_confident: float = 0.01,
    cand_min_n_confident: int = 2,
    cand_min_AD_in_positives: float = 1.25,
    cand_min_DP_in_positives: float = 10,
    cand_only_genes: bool = None,
    filter_artefacts: bool = True,
    # 2-4. genotyping and variant QC
    qc_alpha: float = 0.05,
    geno_alpha: float = 1e-3,
    min_snr: float = 10.0,
    # 5. dropout imputation
    impute: bool = False,
    # 6. characters
    max_prevalence: float = 0.5,
    min_n_var: int = 1,
    # output
    metric: str = 'weighted_jaccard',
    ncores: int = 8,
    seed: int = 0,
    copy: bool = False
    ) -> AnnData | None:
    """
    Filter an Allele Frequency Matrix down to the MT-SNVs characters used for lineage inference.

    The pipeline runs, in this order:

    1. **candidate MT-SNVs** (`mito.pp.filter_low_quality_variants`,
       `mito.pp.filter_candidate_variants`, `mito.pp.filter_known_artefacts`): read-level
       statistics decide which variants are worth testing at all.
    2. **provisional genotypes** (`mito.pp.call_genotypes` at `geno_alpha`/10): stricter
       calls, so that the graph the QC runs on is not built out of noise.
    3. **clonality filter** (`mito.pp.filter_non_clonal_variants`): keeps the variants whose
       carriers are clustered on the cell kNN graph, or mutually exclusive with the other
       variants' carriers. Replaces the Moran's I filter of the published pipeline.
    4. **final genotypes** (`mito.pp.call_genotypes` at `geno_alpha`) and the
       signal-to-background filter (`mito.pp.filter_low_signal_variants`).
    5. **dropout imputation** (`mito.pp.impute_dropouts`), optional.
    6. **characters**: prevalence cap, at least two calls, and four-gamete compatibility
       (`mito.pp.filter_incompatible_variants`); then cell-cell distances.

    Each stage is a public function and can be run on its own; this wrapper fixes their
    order, exposes the parameters worth tuning, and records provenance in
    .uns["mito"]["filter_afm"]. Stage parameters left out here (permutations, graph sizes,
    imputation k and threshold, background iterations) are available on the individual
    functions.

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix, after `mito.pp.filter_cells`.
    lineage_column : str, optional
        Ground truth lineage column in afm.obs. If given, per-variant enrichment in its
        categories is computed and stored in .var. Default is None.
    min_cell_number : int, optional
        Minimum number of cells per category of `lineage_column`. Default is 0.
    cand_min_site_cov, cand_min_quality, cand_min_frac_negative, cand_min_n_positive, cand_af_confident, cand_min_n_confident, cand_min_AD_in_positives, cand_min_DP_in_positives
        Candidate MT-SNVs thresholds. See `mito.pp.filter_candidate_variants`.
    cand_only_genes : bool, optional
        Restrict to MT-SNVs inside MT-gene bodies. None (default) decides from the AFM:
        on for targeted assays, which only enrich that region, off for assays covering
        the genome uniformly (mtscATAC, ReDeeM).
    filter_artefacts : bool, optional
        Remove common dbSNP variants and REDIdb RNA edits. Default is True.
    qc_alpha : float, optional
        Significance of the clonality filter. This sets the smallest clone that can be
        kept: markers of very small clones cannot reach small permutation p-values.
        Default is 0.05.
    geno_alpha : float, optional
        Significance of the per-cell genotyping test. Default is 1e-3.
    min_snr : float, optional
        Minimum signal-to-background ratio of a variant. Default is 10.
    impute : bool, optional
        Impute genotype dropouts from each cell's neighbourhood. Dataset-dependent: it
        buys coverage at some cost in precision. Default is False.
    max_prevalence : float, optional
        Drop characters called in more than this fraction of cells. Default is 0.5.
    min_n_var : int, optional
        Retain cells with at least this number of characters. Default is 1.
    metric : str, optional
        Distance metric. Default is "weighted_jaccard".
    ncores : int, optional
        Cores for the distance computation. Default is 8.
    seed : int, optional
        Random seed of the permutation tests. Default is 0.
    copy : bool, optional
        Return a modified copy instead of updating `afm` in place. Default is False.

    Returns
    -------
    AnnData | None
        Filtered AFM if `copy` is True, otherwise None.
    """

    afm = afm.copy() if copy else afm
    T = Timer()
    T.start()
    flow = []

    def _record(stage, n_vars_before):
        flow.append((stage, int(n_vars_before), int(afm.shape[1])))
        logging.info(f'{stage}: n cells={afm.shape[0]}, n MT-SNVs={afm.shape[1]}')

    annotate_vars(afm)
    logging.info(
        f'Filter AFM: {afm.uns.get("scLT_system", "unknown")} / '
        f'{afm.uns.get("pp_method", "unknown")}, '
        f'n cells={afm.shape[0]}, n MT-SNVs={afm.shape[1]}'
    )
    n_cells_in = afm.shape[0]

    # 1. Candidate MT-SNVs
    if min_cell_number>0 and lineage_column not in [None, 'null']:
        filter_small_clones(afm, column=lineage_column, min_cell_number=min_cell_number)
        annotate_vars(afm, overwrite=True)

    if cand_only_genes is None:
        # Targeted assays enrich the MT-gene bodies, and their coverage metrics say so
        cand_only_genes = 'median_target_site_coverage' in afm.obs.columns
        logging.info(f'Restrict to MT-gene bodies: {cand_only_genes}')

    n0 = afm.shape[1]
    filter_low_quality_variants(
        afm, min_site_cov=cand_min_site_cov, min_var_quality=cand_min_quality,
        only_genes=cand_only_genes
    )
    _record('quality filter', n0)
    if afm.shape[1] == 0 or afm.shape[0] == 0:
        # Checked here because the next stage re-annotates the variants, and would
        # otherwise fail on an empty object rather than say what went missing.
        raise ValueError(
            f'Nothing left after the quality filter ({afm.shape[0]} cells, {afm.shape[1]} '
            f'MT-SNVs): relax the cand_min_site_cov / cand_min_quality thresholds, or '
            f'check that the input AFM has any alternative allele at all.'
        )

    n0 = afm.shape[1]
    filter_candidate_variants(
        afm,
        min_cov=cand_min_site_cov,
        min_var_quality=cand_min_quality,
        min_frac_negative=cand_min_frac_negative,
        min_n_positive=cand_min_n_positive,
        af_confident_detection=cand_af_confident,
        min_n_confidently_detected=cand_min_n_confident,
        min_mean_AD_in_positives=cand_min_AD_in_positives,
        min_mean_DP_in_positives=cand_min_DP_in_positives,
    )
    _record('candidate filter', n0)

    if filter_artefacts:
        n0 = afm.shape[1]
        filter_known_artefacts(afm)
        _record('known artefacts', n0)

    if afm.shape[1] == 0:
        raise ValueError(
            'No candidate MT-SNV survived the read-level filters. Relax the cand_* '
            'thresholds, or check that the input AFM has enough signal.'
        )

    # 2-3. Provisional genotypes, then the clonality filter. The QC runs on stricter
    # calls than the final ones: a noise call placed on the graph is a cell placed in
    # the wrong neighbourhood, and the test is only as good as the graph it uses.
    call_genotypes(afm, alpha=geno_alpha/10)
    n0 = afm.shape[1]
    filter_non_clonal_variants(afm, alpha=qc_alpha, seed=seed)
    _record('clonality filter', n0)

    if afm.shape[1] < 2:
        raise ValueError(
            f'The clonality filter retained {afm.shape[1]} MT-SNVs: not enough to build a '
            f'tree. Raise qc_alpha, or relax the candidate filters.'
        )

    # 4. Final genotypes and signal over background
    call_genotypes(afm, alpha=geno_alpha)
    n0 = afm.shape[1]
    filter_low_signal_variants(afm, min_snr=min_snr)
    _record('signal-to-background filter', n0)

    # 5. Dropout imputation
    if impute:
        impute_dropouts(afm)

    # 6. Characters: prevalence cap, at least two calls, four-gamete compatibility.
    # NB: a character with fewer than two calls carries no information, and would give
    # weighted_jaccard an undefined weight.
    n0 = afm.shape[1]
    B = afm.layers['bin'].toarray()>0
    afm._inplace_subset_var((B.mean(0)<=max_prevalence) & (B.sum(0)>=2))
    _record('prevalence and call-count filters', n0)

    n0 = afm.shape[1]
    filter_incompatible_variants(afm)
    _record('compatibility filter', n0)

    if afm.shape[1] == 0:
        raise ValueError('No character left after the character filters: relax max_prevalence or qc_alpha.')

    # Cells with at least min_n_var characters
    afm._inplace_subset_obs((afm.layers['bin']>0).sum(axis=1).A1>=min_n_var)
    logging.info(f'Retain cells with at least {min_n_var} MT-SNVs: {afm.shape[0]}')
    if afm.shape[0] == 0:
        raise ValueError(
            f'No cell carries at least min_n_var={min_n_var} MT-SNVs after filtering. '
            f'Lower min_n_var, or relax the variant filters.'
        )

    annotate_vars(afm, overwrite=True)

    # Per-cell summaries of the final character set: how much evidence each cell carries
    B = afm.layers['bin'].toarray()>0
    imputed = afm.layers['imputed'].toarray()>0 if 'imputed' in afm.layers else np.zeros(B.shape, bool)
    afm.obs['n_characters'] = B.sum(1)
    afm.obs['n_imputed'] = imputed.sum(1)

    compute_distances(afm, metric=metric, ncores=ncores)

    if lineage_column is not None and lineage_column in afm.obs.columns:
        logging.info(
            f'Ground truth column "{lineage_column}" present: per-variant enrichment is '
            f'available from mito.pp.compute_lineage_biases (not stored on the AFM).'
        )

    # Statistics the stages needed to make their decisions, but that nothing downstream
    # reads: the p-values of the clonality tests, the conflict counts, and the
    # intermediate call counts. They stay available when a stage is run on its own.
    afm.var = afm.var.drop(
        columns=['p_join', 'p_exclusive', 'p_clonal', 'clonal', 'n_conflicts'], errors='ignore'
    )

    # Provenance
    record(afm, 'filter_afm', {
        'params' : {
            'candidates' : {
                'min_site_cov':cand_min_site_cov, 'min_quality':cand_min_quality,
                'min_frac_negative':cand_min_frac_negative, 'min_n_positive':cand_min_n_positive,
                'af_confident':cand_af_confident, 'min_n_confident':cand_min_n_confident,
                'min_AD_in_positives':cand_min_AD_in_positives,
                'min_DP_in_positives':cand_min_DP_in_positives,
                'only_genes':cand_only_genes, 'filter_artefacts':filter_artefacts
            },
            'genotyping' : {'alpha':geno_alpha, 'alpha_qc':geno_alpha/10},
            'clonality' : {'alpha':qc_alpha, 'seed':seed},
            'signal' : {'min_snr':min_snr},
            'imputation' : {'enabled':impute},
            'characters' : {'max_prevalence':max_prevalence, 'min_calls':2, 'min_n_var':min_n_var},
        },
        'flow' : pd.DataFrame(flow, columns=['stage', 'n_vars_in', 'n_vars_out']),
        'converged' : bool(afm.uns['genotyping']['converged']),
        'n_cells_in' : int(n_cells_in),
        'n_cells_out' : int(afm.shape[0]),
        'seconds' : T.stop()
    })

    # Per-stage keys are folded into the single record above
    for key in ('genotyping', 'clonality', 'compatibility', 'imputation', 'known_artefacts'):
        afm.uns.pop(key, None)

    logging.info(f'Final afm: n cells={afm.shape[0]}, n characters={afm.shape[1]}')
    logging.info(f'AFM filtering complete: {afm.uns["mito"]["filter_afm"]["seconds"]}')

    return afm if copy else None


##
