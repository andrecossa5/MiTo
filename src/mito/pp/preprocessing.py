"""
Pre-process AFMs.
"""

from igraph import Graph
from typing import Dict, Tuple, Iterable, Any
from .filters import *
from .distances import call_genotypes, compute_distances
from .kNN import *
from ..ut.positions import transitions, transversions
from ..tl.phylo import build_tree, AFM_to_seqs


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


def filter_cells(
    afm: AnnData, 
    cell_subset: Iterable[str] = None, 
    cell_filter: str = 'filter1', 
    nmads: int = 5, 
    mean_cov_all: float = 20, 
    median_cov_target: int = 25, 
    min_perc_covered_sites: float = .75
    ) -> AnnData:
    """
    Filter cells from MAESTER/RedeeM Allele Frequency Matrix.

    Args:
        afm (str): Allele Frequency Matrix.
        cell_subset (Iterable[str], optional. Default: None): subset of cells to retain.
        cell_filter (str, optional): cell filtering strategy.
            1. **'filter1'**: Filter cells based on mean MT-genome coverage (all sites).
            2. **'filter2'**: Filter cells based on median target MT-sites coverage and min % of target sites covered (MAESTER only).
        nmads (int, optional): n Minimum Absolute Deviations to filter cells with high MT-library UMI counts. Defaults to 5.
        mean_coverage (int, optional): minimum mean consensus (at least 3-supporting-reads) UMI coverage across MT-genome, per cell. Defaults to 20.
        median_cov_target (int, optional): minimum median UMI coverage at target MT-sites (only for MAESTER data). Defaults to 25.
        min_perc_covered_sites (float, optional): minimum fraction of MT target sites covered (only for MAESTER data). Defaults to .75.

    Returns:
        AnnData: Filetered Allele Frequency Matrix
    """

    if cell_subset is not None: 
        cells = list(set(cell_subset) & set(afm.obs_names))
        logging.info(f'Filter provided cell subset. Valid CBs: {len(cells)}')
        afm = afm[cells,:].copy()

    scLT_system = afm.uns['scLT_system']
    logging.info(f'scLT system: {scLT_system}')

    # Custom cell filters
    if cell_filter == 'filter1':

        if scLT_system == 'MAESTER' or scLT_system == 'RedeeM':
            x = afm.obs['mean_site_coverage']       
            median = np.median(x)
            MAD = np.median(np.abs(x-median))
            test = (x>=mean_cov_all) & (x<=median+nmads*MAD)
            afm = afm[test,:].copy()                                      
            logging.info(f'Filtered cells (i.e., mean MT-genome coverage >={mean_cov_all} and <={median+nmads*MAD:.2f}): {afm.shape[0]}')
            afm.uns['cell_filter'] = {
                'cell_subset':cell_subset,
                'cell_filter':cell_filter,
                'nmads':nmads, 
                'mean_cov_all':mean_cov_all
            }
        else:
            raise ValueError(f'Cell filter {cell_filter} is not available for scLT_system {scLT_system}')

    elif cell_filter == 'filter2':
        
        if scLT_system == 'MAESTER': 
            test1 = afm.obs['median_target_site_coverage'] >= median_cov_target
            test2 = afm.obs['frac_target_site_covered'] >= min_perc_covered_sites
            afm = afm[(test1) & (test2),:].copy()                                  
            logging.info(f'Filtered cells (i.e., median target MT-genome coverage >={median_cov_target} and fraction covered sites >={min_perc_covered_sites}: {afm.shape[0]}')
            afm.uns['cell_filter'] = {
                'cell_subset':cell_subset,
                'cell_filter':cell_filter,
                'median_cov_target':median_cov_target, 
                'min_perc_covered_sites':min_perc_covered_sites
            }
        else:
            raise ValueError(f'Cell filter {cell_filter} is not available for scLT_system {scLT_system}')
    
    else:
        logging.info(f'Skipping cell filters: {cell_filter} not available. Filtered cells: {afm.shape[0]}')

    # Ensure each site has been observed from at least one cell
    test_atleastone = np.sum(afm.X.A>0, axis=0)>0
    afm = afm[:,test_atleastone].copy()

    return afm


##


def compute_metrics_raw(afm):
    """
    Compute raw dataset metrics and update .uns.
    """
    
    # Compute general cell-site coverage metrics
    d = {}
    if afm.uns["scLT_system"] == "MAESTER":

        if afm.uns['pp_method'] in ['mito_preprocessing', 'maegatk']:
            d['median_site_cov'] = afm.obs['median_target_site_coverage'].median()
            d['median_target/untarget_coverage_logratio'] = np.median(
                np.log10(
                    afm.obs['median_target_site_coverage'] / \
                    (afm.obs['median_untarget_site_coverage']+0.000001)
                )
            ).round(2)
        else:
            logging.info(f'Skip general metrics for pp_method {afm.uns["pp_method"]}.')

    elif afm.uns["scLT_system"] == "redeem":
        d['median_site_cov'] = afm.obs['mean_site_coverage'].median()
    
    else:
        logging.info(f'Skip raw metrics (scLT_system: {afm.uns["scLT_system"]}).')

    afm.uns['dataset_metrics'] = d


##


def compute_connectivity_metrics(X):
    """
    Calculate the connectivity metrics presented in Weng et al., 2024.
    """

    # Create connectivity graph
    A = np.dot(X, X.T)
    np.fill_diagonal(A, 0)
    A.diagonal()
    g = Graph.Adjacency((A>0).tolist(), mode='undirected')
    edges = g.get_edgelist()
    weights = [A[i][j] for i, j in edges]
    g.es['weight'] = weights

    # Calculate metrics
    average_degree = sum(g.degree()) / g.vcount()                       # avg_path_length
    if g.is_connected():
        average_path_length = g.average_path_length()
    else:
        largest_component = g.clusters().giant()
        average_path_length = largest_component.average_path_length()   # avg_path_length
    transitivity = g.transitivity_undirected()                          # transitivity
    components = g.clusters()
    largest_component_size = max(components.sizes())
    proportion_largest_component = largest_component_size / g.vcount()  # % cells in largest subgraph

    return average_degree, average_path_length, transitivity, proportion_largest_component


##


def compute_metrics_filtered(afm, spatial_metrics=True, tree_kwargs={}):
    """
    Compute additional metrics on selected MT-SNVs feature space.
    """

    d = {}
    assert 'bin' in afm.layers    
    X_bin = afm.layers['bin'].A.copy()

    # n cells and vars
    d['n_cells'] = X_bin.shape[0]
    d['n_vars'] = X_bin.shape[1]
    # n cells per var and n vars per cell (mean, median, std)
    d['median_n_vars_per_cell'] = np.median((X_bin>0).sum(axis=1))
    d['mean_n_vars_per_cell'] = np.mean((X_bin>0).sum(axis=1))
    d['std_n_vars_per_cell'] = np.std((X_bin>0).sum(axis=1))
    d['mean_n_cells_per_var'] = np.mean((X_bin>0).sum(axis=0))
    d['median_n_cells_per_var'] = np.median((X_bin>0).sum(axis=0))
    d['std_n_cells_per_var'] = np.std((X_bin>0).sum(axis=0))
    # AFM sparseness and genotypes uniqueness
    d['density'] = (X_bin>0).sum() / np.product(X_bin.shape)
    seqs = AFM_to_seqs(afm)
    unique_genomes_occurrences = pd.Series(seqs).value_counts(normalize=True)
    d['genomes_redundancy'] = 1-(unique_genomes_occurrences.size / X_bin.shape[0])
    d['median_genome_prevalence'] = unique_genomes_occurrences.median()
    # Mutational spectra
    class_annot = afm.var_names.map(lambda x: x.split('_')[1]).value_counts().astype('int')
    class_annot.index = class_annot.index.map(lambda x: f'mut_class_{x}')
    n_transitions = class_annot.loc[class_annot.index.str.contains('|'.join(transitions))].sum()
    n_transversions = class_annot.loc[class_annot.index.str.contains('|'.join(transversions))].sum()
    # % lineage-biased mutations
    if afm.var.columns.str.startswith('FDR').any():
        freq_lineage_biased_muts = (afm.var.loc[:,afm.var.columns.str.startswith('FDR')]<=.1).any(axis=1).sum() / afm.shape[1]
    else:
        freq_lineage_biased_muts = np.nan

    # Collect
    d = pd.concat([
        pd.Series(d), 
        class_annot,
        pd.Series({'transitions_vs_transversions_ratio':n_transitions/n_transversions}),
        pd.Series({'freq_lineage_biased_muts':freq_lineage_biased_muts}),
    ])

    # Spatial metrics
    tree = None
    if spatial_metrics:

        # Cell connectedness
        average_degree, average_path_length, transitivity, proportion_largest_component = compute_connectivity_metrics(X_bin)
        d['average_degree'] = average_degree
        d['average_path_length'] = average_path_length
        d['transitivity'] = transitivity
        d['proportion_largest_component'] = proportion_largest_component

        # Baseline tree internal nodes mutations support
        tree = build_tree(afm, precomputed=True, **tree_kwargs)

    # To .uns
    afm.uns['dataset_metrics'].update(d)

    return tree


##


def filter_afm(
    afm: AnnData, 
    lineage_column: str = None, 
    min_cell_number: int = 0, 
    cells: Iterable[str] = None,
    filtering: str = 'MiTo', 
    filtering_kwargs: Dict[str,Any] = {}, 
    filter_moransI: bool = True, 
    max_AD_counts: int = 2, 
    variants: Iterable[str] = None, 
    min_n_var: int = 1, 
    fit_mixtures: bool = False, 
    only_positive_deltaBIC: bool = False, 
    path_dbSNP: str = None, 
    path_REDIdb: str = None, 
    compute_enrichment: bool = False, 
    bin_method: str = 'MiTo', 
    binarization_kwargs: Dict[str,Any] = {}, 
    metric: str = 'weighted_jaccard',
    ncores: int = 8, 
    spatial_metrics: bool = False, 
    tree_kwargs: Dict[str,Any] = {}, 
    return_tree: bool = False
    ):
    """
    Filter an Allele Frequency Matrix for downstream analysis.
    This function implements different strategies to subset the detected cells and MT-SNVs
    to those that exhibit optimal properties for single-cell lineage tracing (scLT). The user
    can tune filtering method defaults via the `filtering_kwargs` argument. Pre-computed sets
    of cells and variants can be selected without relying on any specific method (the function
    ensures integrity of the AFM `AnnData` object after subsetting).

    Args
        afm (AnnData): Allele Frequency Matrix.
        lineage_column (str, optional. Default: None): lineage column of interest.
        min_cell_number (int, optional. Default: 0): minimum number of cells required for groups in afm.obs[`lineage_column`]
        cells (Iterable[str], optional. Default: None): pre-defined list of cells.
        filtering (str, optional. Default: 'MiTo'): MT-SNVs filtering strategy. See mito.pp.filters for available strategies and parameters.
        filtering_kwargs (Dict[str,Any], optional. Default: {}): **kwargs for the selected `filtering` method.
        filter_moransI (bool, optional. Default: True): remove MT-SNVs that are not spatially auto-correlated.
        max_AD_counts (int, optional. Default: 2): retain MT-SNV if at least one cell has `max_AD_counts` alternative allele counts. 
        variants (Iterable[str], optional. Default: None): pre-defined list of variants.
        min_n_var (int, optional. Default: 1): retain cells with at least `min_n_var` MT-SNVs.
        fit_mixtures (bool, optional. Default: False): fit MQuad (Kwock et al., 2022) binomial mixtures.
        only_positive_deltaBIC (bool, optional. Default: False): retain only MT-SNVs with positive deltaBIC (from MQuad)
        path_dbSNP (str, optional. Default: None): path to tab-separated file with "COMMON" MT-SNVs (dbSNP database). See tutorial.
        path_REDIdb (str, optional. Default: None): path to tab-separated file with common MT-RNA edits (REDIdb database). See tutorial.
        compute_enrichment (bool, optional. Default: True): compute MT-SNVs enrichment in `lineage_column`
        bin_method (str, optional. Default: 'MiTo'): genotyping method.
        binarization_kwargs (Dict[str,Any], optional. Default: {}). genotyping **kwargs.
        metric (str, optional. Default: 'weighted_jaccard'): distance metric.
        ncores (int, optional. Default: 1): n cores to use for distance computations and fit_MQuad mixtures, if necessary.
        spatial_metrics (bool, optional. Default: False): compute "spatial" connectivity metrics for filtered MT-SNVs.
        tree_kwargs (Dict[str,Any], optional. Default: {}). tree inference (i.e., `mito.tl.build_tree`) **kwargs.
        return_tree (bool, optional. Default: False): return CassiopeiaTree, if `spatial_metrics` == True

    Returns
        afm (AnnData): Filtered Allelic Frequency Matrix.
    """

    logging.info('Compute general dataset metrics...')
    compute_metrics_raw(afm)

    logging.info('Compute vars_df as in Weng et al., 2024')
    annotate_vars(afm)

    logging.info(f'Filter MT-SNVs...')
    scLT_system = afm.uns['scLT_system']
    pp_method = afm.uns['pp_method'] if 'pp_method' in afm.uns else 'previously pre-processed (public data)'
    logging.info(f'scLT_system: {scLT_system}')
    logging.info(f'pp_method: {pp_method}')
    logging.info(f'Feature selection method: {filtering}')
    logging.info(f'Original afm: n cells={afm.shape[0]}, n features={afm.shape[1]}.')
    
    # Cells from <lineage_column> with at least min_cell_number cells, if necessary
    if min_cell_number>0 and lineage_column not in [None, 'null']:
        afm = filter_cell_clones(afm, column=lineage_column, min_cell_number=min_cell_number)
        annotate_vars(afm, overwrite=True)
       
    # Baseline filter
    afm = filter_baseline(afm)
    logging.info(f'afm after baseline filter: n cells={afm.shape[0]}, n features={afm.shape[1]}.')
    
    # Custom filters
    if filtering in filtering_options:

        if filtering == 'baseline':
            pass
        if filtering == 'CV':
            afm = filter_CV(afm, **filtering_kwargs)
        elif filtering == 'miller2022':
            afm = filter_miller2022(afm, **filtering_kwargs)
        elif filtering == 'weng2024':
            afm = filter_weng2024(afm, **filtering_kwargs)
        elif filtering == 'MQuad':
            afm = filter_MQuad(afm, ncores=ncores, path_=os.getcwd(), **filtering_kwargs)
        elif filtering == 'MiTo':
            afm = filter_MiTo(afm, **filtering_kwargs)
        elif filtering == 'GT_enriched':
            afm = filter_GT_enriched(afm, lineage_column=lineage_column, **filtering_kwargs)

    elif filtering is None:
        
        logging.info(f'Filtering custom sets of cells and variants')
        rows = cells if cells is not None else afm.obs_names 
        cols = variants if variants is not None else afm.var_names 
        afm = afm[
            [x for x in rows if x in afm.obs_names],
            [x for x in cols if x in afm.var_names]
        ].copy()
    
    else:
        raise ValueError(
                f'''The provided filtering method {filtering} is not supported.
                    Choose another one...'''
            )

    # Filter common SNVs and possible RNA-edits
    n_dbSNP = np.nan
    if path_dbSNP not in [None, 'null']:
        if os.path.exists(path_dbSNP):
            common = pd.read_csv(path_dbSNP, index_col=0, sep='\t')
            common = common['pos'].astype('str') + '_' + common['REF'] + '>' + common['ALT'].map(lambda x: x.split('|')[0])
            common = common.to_list()
            n_dbSNP = afm.var_names.isin(common).sum()
            logging.info(f'Exclude {n_dbSNP} common SNVs events (dbSNP)')
            variants = afm.var_names[~afm.var_names.isin(common)]
            afm = afm[:,variants].copy() 

    # Filter possible RNA-edits  
    n_REDIdb = np.nan     
    if path_REDIdb not in [None, 'null']:
        if os.path.exists(path_REDIdb):
            edits = pd.read_csv(path_REDIdb, index_col=0, sep='\t')
            edits = edits.query('nSamples>100')
            edits = edits['Position'].astype('str') + '_' + edits['Ref'] + '>' + edits['Ed']
            edits = edits.to_list()
            n_REDIdb = afm.var_names.isin(edits).sum()
            logging.info(f'Exclude {n_REDIdb} common RNA editing events (REDIdb)')
            variants = afm.var_names[~afm.var_names.isin(edits)]
            afm = afm[:,variants].copy()


    # Genotype cells, and filter the one with less than min_n_var mutations
    call_genotypes(afm, bin_method=bin_method, **binarization_kwargs)
    afm = afm[np.sum(afm.layers['bin'].A>0, axis=1)>=min_n_var,:].copy()
    logging.info(f'Retain cells with at least {min_n_var} MT-SNVs: {afm.shape[0]}')
 
    # Bimodal mixture modelling: deltaBIC (MQuad-like) and max AD in at least one cell (Weng et al., 2024)
    if fit_mixtures:
        afm.var = afm.var.join(fit_MQuad_mixtures(afm, ncores=ncores).dropna()[['deltaBIC']])
        if only_positive_deltaBIC:
            afm = afm[:,afm.var['deltaBIC']>0].copy()
            logging.info(f'Remove MT-SNVs with deltaBIC<0')
    if max_AD_counts>1:
        afm = afm[:,np.max(afm.layers['AD'].A, axis=0)>=max_AD_counts].copy()
        logging.info(f'Remove MT-SNVs with no +cells having at least {max_AD_counts} AD counts')

    # Compute cell-cell distances and filter variants significantly auto-correlated.
    compute_distances(afm, precomputed=True, metric=metric, ncores=ncores)
    if filter_moransI:
        afm = filter_variant_moransI(afm)
    logging.info(f'Filter only MT-SNVs with significant spatial auto-correlation (i.e., Moran I statistics).')
    
    # Final fixes
    afm = afm[np.sum(afm.layers['bin'].A>0, axis=1)>=min_n_var,:].copy()
    annotate_vars(afm, overwrite=True)
    logging.info(f'Retain cells with at least {min_n_var} MT-SNVs: {afm.shape[0]}')
    logging.info(f'Last (optional) filters: filtered afm contains {afm.shape[0]} cells and {afm.shape[1]} MT-SNVs.')
    
    ##

    # Lineage bias
    if lineage_column in afm.obs.columns and compute_enrichment:
        logging.info(f'Compute MT-SNVs enrichment for {lineage_column} categories')
        lineages = afm.obs[lineage_column].dropna().unique()
        for target_lineage in lineages:
            res = compute_lineage_biases(afm, lineage_column, target_lineage, 
                                        bin_method=bin_method, binarization_kwargs=binarization_kwargs)
            afm.var[f'FDR_{target_lineage}'] = res['FDR']
            afm.var[f'odds_ratio_{target_lineage}'] = res['odds_ratio']

    # Compute final metrics
    logging.info(f'Compute last (filtered) statistics.')
    tree = compute_metrics_filtered(
        afm, 
        spatial_metrics=spatial_metrics, 
        tree_kwargs=tree_kwargs
    )

    # Add params to .uns
    afm.uns['char_filter'] = {
        'lineage_column' : lineage_column, 
        'min_cell_number' : min_cell_number,
        'filtering' : filtering if (cells is not None) or (variants is not None) else 'predefined_sets',
        'max_AD_counts' : max_AD_counts,
        'only_positive_deltaBIC' : only_positive_deltaBIC,
        'compute_enrichment' : compute_enrichment,
        'spatial_metrics' : spatial_metrics,
        'n_dbSNP' : n_dbSNP,
        'n_REDIdb' : n_REDIdb,
        'min_n_var' : min_n_var
    }
    afm.uns['char_filter'].update(filtering_kwargs)
    
    if return_tree:
        return afm, tree
    else:
        return afm


##


