"""
All filters: variants/cells.
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.sandbox.stats.multicomp import multipletests
from mquad.mquad import *
from .distances import *
from ..ut.positions import mask_mt_sites
from ..ut.utils import load_edits_REDIdb, load_common_dbSNP


##


filtering_options = [
    'baseline',
    'CV',
    'miller2022', 
    'weng2024',
    'MQuad', 
    'MiTo',
    'GT_enriched'

    # DEPRECATED
    # 'ludwig2019', 
    # 'velten2021', 
    # 'seurat', 
    # 'MQuad_optimized',
    # 'density',
    # 'GT_stringent'
]



##


def nans_as_zeros(afm):
    """
    Fill nans with zeros.
    """
    X_copy = afm.X.copy()
    X_copy[np.isnan(X_copy)] = 0
    afm.X = X_copy
    return afm


##


def filter_cells_with_at_least_one(
    afm: AnnData, 
    bin_method: str = 'vanilla', 
    binarization_kwargs: Dict[str,Any] = {}
    ) -> AnnData:
    """
    Filter cells with at least one variant (genotypes from `bin_method`).
    """
    X = call_genotypes(a=afm.copy(), bin_method=bin_method, **binarization_kwargs)
    afm = afm[afm.obs_names[X.sum(axis=1).A1>=1],:]
    afm.uns['per_position_coverage'] = afm.uns['per_position_coverage'].loc[afm.obs_names,:]
    afm.uns['per_position_quality'] = afm.uns['per_position_quality'].loc[afm.obs_names,:]

    return afm


##


def filter_cell_clones(
    afm: AnnData, 
    column: str = 'GBC', 
    min_cell_number: int = 10
    ) -> AnnData:
    """
    Filter only cells from groups in afm.obs[`column`] with more 
    than `min_cell_number` cells.
    """
    
    logging.info(f'Filtering cells from {column} groups with >={min_cell_number} cells')
    
    n0 = afm.shape[0]
    cell_counts = afm.obs.groupby(column).size()
    clones_to_retain = cell_counts[cell_counts>=min_cell_number].index 
    test = afm.obs[column].isin(clones_to_retain)
    afm = afm[test,:].copy()

    logging.info(f'Removed other {n0-afm.shape[0]} cells')
    logging.info(f'Retaining {afm.obs[column].unique().size} discrete categories (i.e., {column}) for the analysis.')
          
    return afm


##


def annotate_vars(afm: AnnData, overwrite: bool = False):
    """
    Annotate MT-SNVs properties as in in Weng et al., 2024, and Miller et al. 2022 before.
    Create vars_df and update .var.
    """

    if 'mean_af' in afm.var.columns:
        if not overwrite:
            return
        else:
            logging.info('Re-annotate variants in afm')
            afm.var = afm.var.iloc[:,:3].copy()

    # Initialize vars_df

    # vars.tib <- tibble(var = rownames(af.dm),
    #                    mean_af = rowMeans(af.dm),
    #                    mean_cov = rowMeans(assays(maegtk)[["coverage"]])[as.numeric(cutf(rownames(af.dm), d = "_"))],
    #                    quality = qual.num)

    afm.var['mean_af'] = afm.X.mean(axis=0).A1

    if 'site_coverage' in afm.layers:
        afm.var['mean_cov'] = afm.layers['site_coverage'].mean(axis=0).A1
    if 'qual' in afm.layers:    # NB: not computed for redeem data
        qual = afm.layers['qual'].toarray()
        afm.var['quality'] = np.nanmean(np.where(qual>0, qual, np.nan), axis=0)
        del qual

    # Calculate the number of cells that exceed VAF thresholds 0, 1, 5, 10, 50 as in Weng et al., 2024

    # vars.tib <- vars.tib %>%
    #     mutate(n0 = apply(af.dm, 1, function(x) sum(x == 0))) %>%  # NEGATIVE CELLS
    #     mutate(n1 = apply(af.dm, 1, function(x) sum(x > 1))) %>%
    #     mutate(n5 = apply(af.dm, 1, function(x) sum(x > 5))) %>%
    #     mutate(n10 = apply(af.dm, 1, function(x) sum(x > 10))) %>%
    #     mutate(n50 = apply(af.dm, 1, function(x) sum(x > 50)))
    # Variant_CellN<-apply(af.dm,1,function(x){length(which(x>0))})
    # vars.tib<-cbind(vars.tib,Variant_CellN)

    afm.var['n0'] = (afm.X==0).sum(axis=0).A1       # NEGATIVE CELLS
    afm.var['n1'] = (afm.X>.01).sum(axis=0).A1           
    afm.var['n2'] = (afm.X>.02).sum(axis=0).A1           
    afm.var['n5'] = (afm.X>.05).sum(axis=0).A1           
    afm.var['n10'] = (afm.X>.1).sum(axis=0).A1           
    afm.var['n50'] = (afm.X>.5).sum(axis=0).A1           
    afm.var['Variant_CellN'] = (afm.X>0).sum(axis=0).A1

    # Add mean AF, AD and DP in +cells
    X = afm.X.toarray()>0
    afm.var['median_af_in_positives'] = np.nanmean(np.where(X>0, X, np.nan), axis=0)
    afm.var['mean_AD_in_positives'] = np.nanmean(
        np.where(X>0, afm.layers['AD'].toarray(), np.nan), axis=0
    )
    afm.var['mean_DP_in_positives'] = np.nanmean(
        np.where(X>0, afm.layers['DP'].toarray(), np.nan), axis=0
    )
    del X


##


def filter_baseline(
    afm: AnnData, 
    min_site_cov: int = 5, 
    min_var_quality: int = 30, 
    min_n_positive: int = 2, 
    only_genes: bool = True
    ) -> AnnData:
    """
    Compute summary stats and baseline filter MT-SNVs (MAESTER, redeem).
    """

    if afm.uns['scLT_system'] == 'MAESTER':

        if only_genes:
            test_sites = mask_mt_sites(afm.var['pos'])
            afm = afm[:,test_sites].copy()

        # Basic filter as in Weng et al., 2024
        if afm.uns['pp_method'] in ['mito_preprocessing', 'maegatk']:
            test_baseline = (
                (afm.var['mean_cov']>=min_site_cov) & \
                (afm.var['quality']>=min_var_quality) & \
                (afm.var['Variant_CellN']>=min_n_positive) 
            )
            afm = afm[:,test_baseline].copy()
        else:
            logging.info('Baseline filter only exlcudes MT-SNVs in un-targeted sites.')
    
    elif afm.uns['scLT_system'] == 'RedeeM':
        test_baseline = (
            (afm.var['mean_cov']>=min_site_cov) & \
            (afm.var['Variant_CellN']>=min_n_positive) 
        )
        afm = afm[:,test_baseline].copy()

    else:
        raise ValueError(f'Baseline filter not available for scLT_system current scLT_system and pp_method')

    # Exclude sites with more than one alt alleles observed
    var_sites = afm.var_names.map(lambda x: x.split('_')[0])
    test = var_sites.value_counts()[var_sites]==1
    afm = afm[:,afm.var_names[test]].copy()

    # Exclude variants sites not observed in any cells and vice versa
    afm = afm[(afm.X>0).sum(axis=1).A1>0,:].copy()
    afm = afm[:,(afm.X>0).sum(axis=0).A1>0].copy()

    return afm


##


def filter_CV(afm: AnnData, n_top: int = 100) -> AnnData:
    """
    Filter top `n_top` MT-SNVs (MAESTER, redeem), ranked by coefficient of variation (CV).
    """

    scLT_system = afm.uns['scLT_system']
    pp_method = afm.uns['pp_method']

    if scLT_system == 'MAESTER' and pp_method in ['mito_preprocessing', 'maegatk']:
        pass
    else:
        raise ValueError(f'CV filter not available for scLT_system {scLT_system} and pp_method {pp_method}')

    X = afm.X.toarray()
    CV = (np.std(X, axis=0)**2 / np.mean(X, axis=0))
    idx_vars = np.argsort(CV)[::-1][:n_top]
    afm = afm[:,idx_vars].copy()
    del X

    return afm


##


def filter_miller2022(
    afm: AnnData, 
    min_site_cov: float = 100, 
    min_var_quality: float = 30, 
    p1: int = 1, 
    p2: int = 99, 
    perc1: float = 0.01, 
    perc2: float = 0.1
    ) -> AnnData: 
    """
    Filter MT-SNVs (MAESTER only) based on adaptive tresholds adopted in Miller et al., 2022.
    """

    scLT_system = afm.uns['scLT_system']
    pp_method = afm.uns['pp_method']

    if scLT_system == 'MAESTER' and pp_method in ['mito_preprocessing', 'maegatk']:
        pass
    else:
        raise ValueError(f'miller2022 filter not available for scLT_system {scLT_system} and pp_method {pp_method}')

    X = afm.X.toarray()
    test = (
        (afm.var['mean_cov']>=min_site_cov) & \
        (afm.var['quality']>=min_var_quality) & \
        ((np.percentile(X, q=p1, axis=0) < perc1) & \
         (np.percentile(X, q=p2, axis=0) > perc2))
    )
    afm = afm[:,test].copy()

    return afm


##


def fit_MQuad_mixtures(afm, n_top=None, path_=None, ncores=8, minDP=10, minAD=1, with_M=False):
    """
    Filter MT-SNVs (MAESTER, redeem) with the MQuad method (Kwock et al., 2022)
    """

    if n_top is not None:    
        afm = filter_CV(afm, n_top=n_top) # Prefilter again, if still too much MT-SNVs

    # Fit models
    M = Mquad(AD=afm.layers['AD'].T, DP=afm.layers['DP'].T)
    path_ = os.getcwd() if path_ is None else path_
    df = M.fit_deltaBIC(out_dir=path_, nproc=ncores, minDP=minDP, minAD=minAD)
    df.index = afm.var_names
    df['deltaBIC_rank'] = df['deltaBIC'].rank(ascending=False)

    if with_M:
        return df.sort_values('deltaBIC', ascending=False), M
    else:
        return df.sort_values('deltaBIC', ascending=False)
    

##


def filter_MQuad(
    afm: AnnData, 
    ncores: int = 8, 
    minDP: int = 5, 
    minAD: int = 1,
    minCell: int = 2, 
    path_: str = None, 
    n_top: int = None
    ) -> AnnData:
    """
    Filter MT-SNVs (MAESTER, redeem) with the MQuad method (Kwock et al., 2022).

    Parameters
    ----------
    afm : AnnData
        The Allele Frequency Matrix.
    ncores : int, optional
        Number of cores to use for computation. Default is 8.
    minDP : int, optional
        Minimum depth (DP) required. Default is 5.
    minAD : int, optional
        Minimum alternative allele (AD) count required. Default is 1.
    minCell : int, optional
        Minimum number of cells required. Default is 2.
    path_ : str, optional
        Path for saving output or intermediate results. Default is None.
    n_top : int, optional
        Number of top MT-SNVs to select. Default is None.

    Returns
    -------
    afm : AnnData
        Filtered Allele Frequency Matrix.
    """

    scLT_system = afm.uns['scLT_system']
    pp_method = afm.uns['pp_method']

    if scLT_system == 'MAESTER' and pp_method in ['mito_preprocessing', 'maegatk', 'cellsnp-lite']:
        pass
    elif scLT_system == 'redeem':
        pass
    else:
        raise ValueError(f'MQuad filter not available for scLT_system {scLT_system} and pp_method {pp_method}')
    
    _, M = fit_MQuad_mixtures(
        afm, n_top=n_top, path_=path_, ncores=ncores, minDP=minDP, minAD=minAD, with_M=True
    )
    _, _ = M.selectInformativeVariants(
        min_cells=minCell, out_dir=path_, tenx_cutoff=None,
        export_heatmap=False, export_mtx=False
    )
    idx = M.final_df.index.to_list()
    selected = [ afm.var_names[i] for i in idx ]
    afm = afm[:,selected].copy()
    afm.var['deltaBIC'] = M.final_df['deltaBIC']

    os.system(f'rm {os.path.join(path_, "*BIC*")}')

    return afm


##


def filter_weng2024(
    afm: AnnData, 
    min_site_cov: float = 5, 
    min_var_quality: float = 30, 
    min_frac_negative: float = .9,
    min_n_positive: int = 2,
    low_confidence_af: float = .1, 
    high_confidence_af: float = .5, 
    min_prevalence_low_confidence_af: float = .1, 
    min_cells_high_confidence_af: int = 2
    ) -> AnnData:
    """
    Filter MT-SNVs (MAESTER only) as in Weng et al., 2024, and Miller et al. 2022.

    Filters variants using the following criteria:
    - At least `min_site_cov` mean site coverage (across cells)
    - At least `min_var_quality` mean variant allele basecall quality (across cells)
    - At least n cells * `min_frac_negative` negative cells 
    - At least `min_n_positive` (AF > 0) cells
    - At least `min_prevalence_low_confidence_af` prevalence at AF less than `low_confidence_af`
    - At least `min_cells_high_confidence_af` cells with AF greater than `high_confidence_af`

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix.
    min_site_cov : float
        Minimum mean site coverage across cells.
    min_var_quality : float
        Minimum mean variant allele basecall quality across cells.
    min_frac_negative : float
        Fraction of cells that must be negative.
    min_n_positive : int
        Minimum number of cells with AF > 0.
    min_prevalence_low_confidence_af : float
        Minimum prevalence (fraction of cells) with AF less than low_confidence_af.
    low_confidence_af : float
        Threshold for low confidence allele frequency.
    min_cells_high_confidence_af : int
        Minimum number of cells required with AF greater than high_confidence_af.
    high_confidence_af : float
        Threshold for high confidence allele frequency.

    Returns
    -------
    afm : AnnData
        Filtered Allele Frequency Matrix.
    """

    scLT_system = afm.uns['scLT_system']
    pp_method = afm.uns['pp_method']

    if scLT_system == 'MAESTER' and pp_method in ['mito_preprocessing', 'maegatk']:
        pass
    else:
        raise ValueError(f'weng2024 filter not available for scLT_system {scLT_system} and pp_method {pp_method}')

    # Filter Weng et al., 2024

    # vars_filter.tib <- vars.tib %>% filter(mean_cov > 5, quality >= 30, n0 > 0.9*ncol(af.dm),Variant_CellN>=2)

    ## Apply the same filter as in MAESTER
    # IsInfo<-function(x){
    # total<-length(x)
    # if(length(which(x<10))/total>0.1 & length(which(x>50))>10){
    #     return("Variable")
    # }else{
    #     return("Non")
    # }
    # }
    # Variability<-apply(af.dm,1,IsInfo) %>% data.frame(Info=.)
    # vars_filter.tib<-Tomerge_v2(vars_filter.tib,Variability) 
    
    annotate_vars(afm, overwrite=True)
    test = (
        (afm.var['mean_cov']>min_site_cov) & \
        (afm.var['quality']>=min_var_quality) & \
        (afm.var['n0']>min_frac_negative*afm.shape[0]) & \
        (afm.var['Variant_CellN']>=min_n_positive) 
    )
    afm = afm[:,test].copy()

    # Detect "Variable" variants as in MAESTER

    # IsInfo<-function(x){
    # total<-length(x)
    # if(length(
        # which(x<10))/total>0.1        # Test1 : low prevalence of minimal detection.
        # & 
        # length(which(x>50))>10)       # Test2 : enough cells with confident detection.
        # {
    #     return("Variable")            
    # }else{
    #     return("Non")
    # }
    # }
    # Variability<-apply(af.dm,1,IsInfo) %>% data.frame(Info=.)

    t1 = (afm.X<low_confidence_af).sum(axis=0).A1 / afm.shape[0] > min_prevalence_low_confidence_af
    t2 = (afm.X>high_confidence_af).sum(axis=0).A1 > min_cells_high_confidence_af
    test = t1 & t2
    afm = afm[:,test].copy() 

    return afm


##


def filter_MiTo(
    afm: AnnData, 
    min_cov: float = 5,
    min_var_quality: float = 30,
    min_frac_negative: float = 0.2,
    min_n_positive: int = 5,
    af_confident_detection: float = .02,
    min_n_confidently_detected: int = 2,
    min_mean_AD_in_positives: float = 1.25,
    min_mean_DP_in_positives: float = 25
    ) -> AnnData:
    """
    MiTo custom filter. Filter variants with:
    - At least `min_cov` mean site coverage (across cells)
    - At least `min_var_quality` mean variant allele basecall quality (across cells)
    - At least n cells * `min_frac_negative` negative cells 
    - At least `min_n_positive` (AF > 0) cells
    - At least `min_n_confidently_detected` cells in which the variant has been detected with AF greater than `af_confident_detection`
    - At least `min_mean_AD_in_positives` mean AD in positive cells
    - At least `min_mean_DP_in_positives` mean DP in positive cells

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix.
    min_cov : float
        Minimum mean site coverage (across cells). Default is 5.
    min_var_quality : float
        Minimum mean variant allele basecall quality (across cells). Default is 30.
    min_frac_negative : float
        Minimum fraction of negative cells (expressed as a fraction of total cells). Default is 0.2.
    min_n_positive : int
        Minimum number of cells with AF > 0. Default is 5.
    min_n_confidently_detected : int
        Minimum number of cells in which the variant has been detected with an AF greater than
        `af_confident_detection`. Default is 2.
    af_confident_detection : float
        Allele frequency threshold for confident detection. Default is 0.02.
    min_mean_AD_in_positives : float
        Minimum mean alternative allele count (AD) in positive cells. Default is 1.25.
    min_mean_DP_in_positives : float
        Minimum mean total UMI counts (DP) in positive cells. Default is 25.

    Returns
    -------
    afm : AnnData
        Filtered Allele Frequency Matrix.
    """

    scLT_system = afm.uns['scLT_system']
    pp_method = afm.uns['pp_method']

    annotate_vars(afm, overwrite=True)
    afm.var['n_confidently_detected'] = (afm.X>=af_confident_detection).sum(axis=0).A1

    if scLT_system == 'MAESTER':

        if pp_method in ['mito_preprocessing', 'maegatk']:
            test = (
                (afm.var['mean_cov']>=min_cov) & \
                (afm.var['quality']>=min_var_quality) & \
                (afm.var['n0']>=min_frac_negative*afm.shape[0]) & \
                (afm.var['Variant_CellN']>=min_n_positive) & \
                (afm.var['n_confidently_detected']>=min_n_confidently_detected) & \
                (afm.var['mean_AD_in_positives']>=min_mean_AD_in_positives) & \
                (afm.var['mean_DP_in_positives']>=min_mean_DP_in_positives) 
            )
            afm = afm[:,test].copy()
        else:
            raise ValueError(f'MiTo filter not available for pp_method: {pp_method}')
        
    elif scLT_system == 'redeem':
        test = (
            (afm.var['mean_cov']>=min_cov) & \
            (afm.var['n0']>=min_frac_negative*afm.shape[0]) & \
            (afm.var['Variant_CellN']>=min_n_positive) & \
            (afm.var['n_confidently_detected']>=min_n_confidently_detected) & \
            (afm.var['mean_AD_in_positives']>=min_mean_AD_in_positives) & \
            (afm.var['mean_DP_in_positives']>=min_mean_DP_in_positives) 
        )
        afm = afm[:,test].copy()
    
    else:
        raise ValueError(f'MiTo filter not available for scLT_system: {scLT_system}')

    return afm

    

##


def compute_lineage_biases(
    afm: AnnData, 
    lineage_column: str, 
    target_lineage: str, 
    bin_method: str = 'MiTo', 
    binarization_kwargs: Dict[str,Any] = {}, 
    alpha: float = .05
    ) -> pd.DataFrame:
    """
    Compute MT-SNVs enrichment scores for a given lineage category using Fisher's exact test.

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix.
    lineage_column : str
        Field in afm.obs containing the 'lineage' categorical variable.
    target_lineage : str
        The category in afm.obs[lineage_column] to test for MT-SNV enrichment.
    bin_method : str, optional
        Genotyping method. Default is "MiTo".
    binarization_kwargs : dict, optional
        Additional keyword arguments for genotyping. Default is {}.
    alpha : float, optional
        Family-wise error rate for p-value correction. Default is 0.05.

    Returns
    -------
    results : pd.DataFrame
        DataFrame containing computed statistics (e.g., -log10(FDR) from Fisher's exact test).
    """

    if lineage_column not in afm.obs.columns:
        raise ValueError(f'{lineage_column} not present in cell metadata!')
        
    muts = afm.var_names
    prevalences_array = np.zeros(muts.size)
    target_ratio_array = np.zeros(muts.size)
    oddsratio_array = np.zeros(muts.size)
    pvals = np.zeros(muts.size)

    if 'bin' not in afm.layers:
        call_genotypes(afm, bin_method=bin_method, **binarization_kwargs)

    # Here we go
    G = afm.layers['bin'].toarray()
    for i in range(muts.size):

        test_mut = G[:,i] == 1
        test_lineage = afm.obs[lineage_column] == target_lineage
        n_mut_lineage = np.sum(test_mut & test_lineage)
        n_mut_no_lineage = np.sum(test_mut & ~test_lineage)
        n_no_mut_lineage = np.sum(~test_mut & test_lineage)
        n_no_mut_no_lineage = np.sum(~test_mut & ~test_lineage)
        prevalences_array[i] = n_mut_lineage / test_lineage.sum()
        target_ratio_array[i] = n_mut_lineage / test_mut.sum()

        # Fisher
        oddsratio, pvalue = fisher_exact(
            [
                [n_mut_lineage, n_mut_no_lineage],
                [n_no_mut_lineage, n_no_mut_no_lineage], 
            ],
            alternative='greater',
        )
        oddsratio_array[i] = oddsratio
        pvals[i] = pvalue

    # Correct pvals --> FDR
    pvals = multipletests(pvals, alpha=alpha, method="fdr_bh")[1]

    # Results
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


def filter_GT_enriched(
    afm: AnnData, 
    lineage_column: str = None, 
    fdr_treshold: float = .1,
    n_enriched_groups: int = 2, 
    bin_method: str = 'MiTo', 
    binarization_kwargs: Dict[str,Any] = {}
    ) -> AnnData:
    """
    Compute MT-SNVs enrichment scores for a given lineage category using 
    -log10(FDR) from Fisher's exact test.

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix.
    lineage_column : str
        Field in afm.obs that contains the 'lineage' categorical variable.
    target_lineage : str
        The category in afm.obs[lineage_column] to test for MT-SNV enrichment.
    bin_method : str, optional
        Genotyping method. Default is "MiTo".
    binarization_kwargs : dict, optional
        Additional keyword arguments for genotyping. Default is {}.
    alpha : float, optional
        Family-wise error rate for p-value correction. Default is 0.05.

    Returns
    -------
    results : pd.DataFrame
        Computed statistics.
    """

    if lineage_column is not None and lineage_column in afm.obs.columns:
        pass
    else:
        raise ValueError(f'{lineage_column} not available in afm.obs!')
    
    L = []
    lineages = afm.obs[lineage_column].dropna().unique()
    for target_lineage in lineages:
        print(f'Computing variants enrichment for lineage {target_lineage}...')
        res = compute_lineage_biases(afm, lineage_column, target_lineage, 
                                    bin_method=bin_method, binarization_kwargs=binarization_kwargs)
        L.append(res['FDR']<=fdr_treshold)
    
    df_enrich = pd.concat(L, axis=1)
    df_enrich.columns = lineages
    test = df_enrich.apply(lambda x: np.sum(x>0)>0 and np.sum(x>0)<=n_enriched_groups, axis=1)
    vois = df_enrich.loc[test].index.unique()
    id_lineages = df_enrich.loc[test].sum(axis=0).loc[lambda x: x>0].index.to_list()
    cells = afm.obs[lineage_column].loc[lambda x: x.isin(id_lineages)].index
    afm = afm[cells, vois].copy()

    return afm 


##


def moran_I(W, x, num_permutations=100):
    """
    Calculate normalized Moran's I statistics and permutation-based pvalue.
    """

    W = W / W.sum()
    x_stdzd = (x-np.mean(x)) / np.std(x,ddof=0)
    I_obs = x_stdzd.T @ W @ x_stdzd

    # Perform permutation test
    permuted_Is = np.zeros(num_permutations)
    for i in range(num_permutations):
        x_perm = np.random.permutation(x_stdzd)
        permuted_Is[i] = x_perm.T @ W @ x_perm
    p_value = np.sum(permuted_Is >= I_obs) / num_permutations

    return I_obs, p_value


##


def filter_variant_moransI(
    afm: AnnData, 
    num_permutations: int = 100, 
    pval_treshold: float =.01
    ) -> AnnData:

    """
    Filter MT-SNVs if not significantly auto-correlated.
    """
    
    assert 'distances' in afm.obsp
    W = 1-afm.obsp['distances'].toarray()
    X = afm.X.toarray()

    I = []
    P = []
    for i in range(afm.shape[1]):
        i, p = moran_I(W, X[:,i], num_permutations=num_permutations)
        I.append(i)
        P.append(p)
    
    afm.var['Moran I '] = I
    afm.var['Moran I pvalue'] = P

    var_to_retain = afm[:,afm.var['Moran I pvalue']<=pval_treshold].var_names
    afm = afm[:,var_to_retain].copy()

    return afm


##


def filter_dbSNP_common(afm: AnnData):
    """
    Filter Allele Frequency Matrix from "COMMON" MT-SNVs (annotated dbSNP database).
    """

    common = load_common_dbSNP()    
    n_dbSNP = afm.var_names.isin(common).sum()
    logging.info(f'Exclude {n_dbSNP} common SNVs events (dbSNP)')
    variants = afm.var_names[~afm.var_names.isin(common)]
    afm = afm[:,variants].copy() 

    return afm, n_dbSNP


##


def filter_REDIdb_edits(afm: AnnData):
    """
    Filter Allele Frequency Matrix from previously annotated RNA-edits (REDIdb database).
    """
    
    edits = load_edits_REDIdb()
    n_REDIdb = afm.var_names.isin(edits).sum()
    logging.info(f'Exclude {n_REDIdb} common RNA editing events (REDIdb)')
    variants = afm.var_names[~afm.var_names.isin(edits)]
    afm = afm[:,variants].copy()

    return afm, n_REDIdb


##