"""
Module to create and process AFMs.
"""

import scanpy as sc
from igraph import Graph
from mito_utils.filters import *
from mito_utils.make_afm import *
from mito_utils.distances import *
from mito_utils.phylo import *


##


##------------------------------------------------------------------------------------------------------##
patterns = [ 'A>C', 'T>G', 'A>T', 'A>G', 'G>A', 'C>G', 'C>A', 'T>A', 'G>C', 'G>T', 'N>T', 'C>T', 'T>C' ]
transitions = [pattern for pattern in patterns if pattern in ['A>G', 'G>A', 'C>T', 'T>C']]
transversions = [pattern for pattern in patterns if pattern not in transitions and 'N' not in pattern]
##------------------------------------------------------------------------------------------------------##


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



def filter_cells(afm, cell_subset=None, cell_filter='filter1', nmads=5, 
                mean_cov_all=20, median_cov_target=25, min_perc_covered_sites=.75):
    """

    Read and format a complete Allele Frequency Matrix (follows the logic of Miller et al., 2022). 

    Args:
        afm (str): AnnData object prepared with mito_utils.make_afm.make.afm().
        cell_subset (list-like, optional): desired subsets of cells.
        cell_filter (str, optional): cell filter.
            1. **'filter1'**: Filter cells based on mean MT-genome coverage (all sites).
            2. **'filter2'**: Filter cells based on median target MT-sites coverage and min % of target sites covered.
        nmads (int, optional): n Minimum Absolute Deviations to filter cells with high MT-library UMI counts. Defaults to 5.
        mean_coverage (int, optional): minimum mean consensus (at least 3-supporting-reads) UMI coverage across MT-genome, per cell. Defaults to 20.
        median_cov_target (int, optional): minimum median UMI coverage at target MT-sites. Defaults to 25.
        min_perc_covered_sites (float, optional): minimum fraction of MT target sites covered. Defaults to .75.

    Returns:
        AnnData: Cell-filtered Allelic Frequency Matrix.
    """

    if cell_subset is not None: 
        cells = list(set(cell_subset) & set(afm.obs_names))
        logging.info(f'Filter provided cell subset. Valid CBs: {len(cells)}')
        afm = afm[cells,:].copy()

    if cell_filter == 'filter1':
        x = afm.obs['mean_site_coverage']       
        median = np.median(x)
        MAD = np.median(np.abs(x-median))
        test = (x>=mean_cov_all) & (x<=median+nmads*MAD)
        afm = afm[test,:].copy()                                      
        logging.info(f'Filtered cells (i.e., mean MT-genome coverage >={mean_cov_all} and <={median+nmads*MAD:.2f}): {afm.shape[0]}')

    elif cell_filter == 'filter2':
        test1 = afm.obs['median_target_site_coverage'] >= median_cov_target
        test2 = afm.obs['frac_target_site_covered'] >= min_perc_covered_sites
        afm = afm[(test1) & (test2),:].copy()                                  
        logging.info(f'Filtered cells (i.e., median target MT-genome coverage >={median_cov_target} and fraction covered sites >={min_perc_covered_sites}: {afm.shape[0]}')

    elif cell_filter == 'filter3':
        test = np.mean(afm.layers['DP'].A, axis=1)>=20
        afm = afm[test,:].copy()                                  
        logging.info(f'Filtered cells (i.e., mean DP {mean_cov_all}): {afm.shape[0]}')
    
    elif cell_filter == 'no filter':
        pass

    afm.uns['cell_filter'] = {
        'cell_filter':cell_filter,
        'nmads':nmads, 
        'mean_cov_all':mean_cov_all, 
        'median_cov_target':median_cov_target,
        'min_perc_covered_sites':min_perc_covered_sites
    }

    # Ensure each site has been observed from at least one cell
    test_atleastone = np.sum(afm.X.A>0, axis=0)>0
    afm = afm[:,test_atleastone].copy()

    return afm


##


def compute_metrics_raw(afm):
    """
    Compute raw dataset metrics and update .uns.
    """

    d = {}
    pp_method = afm.uns['pp_method']
    
    if pp_method in ['mito_preprocessing', 'maegatk']:
        d['median_site_cov'] = afm.obs['median_target_site_coverage'].median()
        d['median_target/untarget_coverage_logratio'] = np.median(
            np.log10(
                afm.obs['median_target_site_coverage'] / \
                (afm.obs['median_untarget_site_coverage']+0.000001)
            )
        ).round(2)
    else:
        logging.info(f'Skip general metrics for pp_method {pp_method}.')

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


def compute_metrics_filtered(afm, spatial_metrics=True, 
                            bin_method='MI_TO', binarization_kwargs={}, tree_kwargs={}):
    """
    Compute additional metrics on selected MT-SNVs feature space.
    """

    # Last time, to ensure consistency
    call_genotypes(afm, bin_method=bin_method, **binarization_kwargs)
    
    d = {}
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
        tree = build_tree(afm, bin_method=bin_method, binarization_kwargs=binarization_kwargs, **tree_kwargs)
        # tree_collapsed = tree.copy()
        # tree_collapsed.collapse_mutationless_edges(True)
        # d['frac_supported_nodes'] = len(tree_collapsed.internal_nodes) / len(tree.internal_nodes)

    # To .uns
    afm.uns['dataset_metrics'].update(d)

    return tree


##


def filter_afm(
    afm, lineage_column=None, min_cell_number=0, cells=None,
    filtering='MI_TO', filtering_kwargs={}, max_AD_counts=1, variants=None,
    fit_mixtures=False, only_positive_deltaBIC=False, path_dbSNP=None, path_REDIdb=None, 
    compute_enrichment=False, bin_method='MI_TO', binarization_kwargs={}, ncores=8,
    spatial_metrics=False, tree_kwargs={}, nproc=8, return_tree=False
    ):
    """
    
    Filter an Allele Frequency Matrix for downstream analysis.
    This function implements different strategies to subset the detected cells and MT-SNVs to those that exhibit
    optimal properties for single-cell lineage tracing (scLT). The user can tune filtering method defaults via the `filtering_kwargs` argument. 
    Pre-computed sets of cells and variants can be selected without relying on any specific method (the function ensures integrity of the AFM `AnnData` object after subsetting).

    Parameters
    ----------
    afm : AnnData
        The AFM to subset. AnnData object with slots as for mito_utils.make_afm.make_afm() preprocessing.
    lineage_column : str, optional
        Categorical label used with `min_cell_number` and `compute_enrichment` arguments. Cells from lineages with fewer than `min_cell_number` cells are discarded. Default is `None`.
    min_cell_number : int, optional
        Cell filter. Minimum number of cells required for a certain lineage to be included in the final data. Default is `0`.
    cells : list, optional
        Pre-computed list of cells to subset. Default is `None`.
    filtering : str, optional
        Filtering method to use. Default is `'MI_TO'`.

        The following filtering strategies are implemented (tunable filtering kwargs that can be passed with the `filtering_kwargs` argument are highlighted as `{"kwarg": default_value}`):

        1. **'baseline'**: Baseline filter to retain only MT-SNV candidates showing minimal conditions to not be considered technical artifacts or sequencing errors. Filters MT-SNVs with:
            - Position in MT-gene bodies (`only_genes`: `True`)
            - Mean site coverage ≥ 5 (`min_site_cov`: `5`)
            - Mean MT-SNV consensus UMI base sequencing quality ≥ 30 (`min_var_quality`: `30`)
            - Number of positive cells ≥ 2 (`min_n_positive`: `2`)
            This filter is applied before every other one to exclude "definitely garbage" MT-SNV candidates.

        2. **'CV'**: Filters the top `n_top` MT-SNVs by Coefficient of Variation (CV). (`n_top`: `100`)

        3. **'miller2022'**: Filter adapted from Miller et al., 2022. Filters MT-SNVs with:
            - Mean site coverage ≥ 100 (`min_site_cov`: `100`)
            - Mean MT-SNV consensus UMI base sequencing quality ≥ 30 (`min_var_quality`: `30`)
            - 1st percentile AF value ≤ 0.01 (`perc_1`: `0.01`)
            - 99th percentile AF value ≥ 0.1 (`perc_99`: `0.1`)

        4. **'weng2024'**: Filter adapted from Weng et al., 2024 MAESTER data analysis. Filters MT-SNVs with:
            - Mean site coverage ≥ 5 (`min_site_cov`: `5`)
            - Mean MT-SNV consensus UMI base sequencing quality ≥ 30 (`min_var_quality`: `30`)
            - Fraction of negative cells (i.e., 0 ALT UMIs) ≥ 0.9 (`min_frac_negative`: `0.9`)
            - Number of positive cells ≥ 2 (`min_n_positive`: `2`)
            - Enough prevalence of minimal detection (`low_confidence_af`: `0.1`, `min_prevalence_low_confidence_af`: `0.1`)
            - Enough evidence of high-AF detection events (`high_confidence_af`: `0.5`, `min_cells_high_confidence_af`: `2`)

        5. **'MQuad'**: Filter from Kwock et al., 2022.

        6. **'MI_TO'**: Default filter, integrating aspects of 'miller2022' and 'weng2024'. Filters MT-SNVs with:
            - Mean MT-SNV consensus UMI base sequencing quality ≥ 30 (`min_var_quality`: `30`)
            - Fraction of negative cells >= 0.2 (`min_frac_negative`: `0.2`)
            - Number of positive cells ≥ 2 (`min_n_positive`: `2`)
            - Enough evidence of high-AF detection events (`af_confident_detection`: `0.01`, `min_n_confidently_detected`: `2`)
            - Enough variant allele-supporting molecules across +cells (`min_mean_AD_in_positives` : `1.5`)
            - Enough coverage across +cells (`min_mean_DP_in_positives` : `25`)

        7. **'GT_enriched'**: Filter with availabe lineage cell annotations (i.e., Ground Truth lentiviral clones.). The AF matrix is binarized, and each variant-lineage enrichment is tested. If a variant is significantly enriched in less than `n_enriched_groups` lineages (assumed independent), is retained.
            - afm.obs column for lineage annotation ('lineage_column': None)
            - FDR threshold for Fisher's Exact test ('fdr_treshold' : .1) 
            - Max number of lineages the MT-SNV can be enriched for ('n_enriched_groups' : 2) 
            - Binarization strategy ('bin_method' : 'MI_TO')
            - **kwargs for bianrization ('binarization_kwargs' : {})

    filtering_kwargs : dict, optional
        Keyword arguments for the selected filtering method. Default is `{}`.
    max_AD_counts : int, optional
        Site/variant filter. The minimum number of consensus UMIs supporting the MT-SNV ALT allele required for at least one cell across the dataset. Default is `1` (i.e., no filter).
    af_confident_detection : float, optional
        Cell filter. The minimum AF threshold at which at least one of the filtered MT-SNVs needs to be detected in a cell to retain the cell in the final dataset. It may be passed as a key-value pair in `filtering_kwargs`. Default is `0.01`.
    variants : list, optional
        Pre-computed list of variants to subset. Default is `None`.
    spatial_metrics : bool, optional
        If `True`, compute a list of "spatial" metrics for retained MT-SNVs, including [details missing]. Default is `False`.
    tree_kwargs : dict, optional
        Tree building keyword arguments that can be passed to `mito_utils.phylo.build_tree` when `spatial_metrics=True`. Default is `{}`.
    fit_mixtures : bool, optional
        If `True`, fit MQuad (Kwock et al., 2022) binomial mixtures and calculate each variant's (passing baseline filters) delta BIC. Default is `False`.
    only_positive_deltaBIC : bool, optional
        Site/variant filter. Irrespective of the filtering strategy, retain only variants with positive delta BIC (estimated with MQuad). Default is `False`.  
    ncores: int, optional
        n cores to use for distance computations and fit_MQuad mixtures, if necessary. Default: 8
    path_dbSNP : str, optional
        Path to a `.txt` tab-separated file with MT-SNVs "COMMON" variants from the dbSNP database. Required fields: `'pos'`, `'ALT'`, `'REF'`. All of these variants will be discarded from the final dataset. Can be found at `<path_main from zenodo folder>/data/MI_TO_bench/miscellanea`. Default is `None`.
    path_REDIdb : str, optional
        Path to a `.txt` tab-separated file with common MT-RNA edits from the REDIdb database. Required fields: `'Position'`, `'Ref'`, `'Ed'`, `'nSamples'`. All of these RNA-editing sites will be discarded from the final dataset. Can be found at `<path_main from zenodo folder>/data/MI_TO_bench/miscellanea`. Default is `None`.
    compute_enrichment : bool, optional
        If `True`, compute MT-SNV enrichment in individual lineages from `lineage_column`. Default is `False`.
    bin_method : str, oprional 
        Binarization strategy used for i) compute the final dataset statistics (i.e., mutation number, connectivity ecc) and ii) lineage enrichment. Default is `MI_TO`
    binarization_kwargs : dict, optional
        Binarization strategy **kwargs (see mito_utils.distances.call_genotypes). Default is `{}`
    return_tree : boll, optional
        Wheter to return the tree usen in spatial metrics.

    Returns
    -------
    afm_filtered : AnnData
        Filtered Allelic Frequency Matrix.
    """

    logging.info('Compute general dataset metrics...')
    compute_metrics_raw(afm)

    logging.info('Compute vars_df as in Weng et al., 2024')
    annotate_vars(afm)

    logging.info(f'Filter MT-SNVs...')
    pp_method = afm.uns['pp_method']
    
    if filtering in filtering_options:

        logging.info(f'Feature selection method: {filtering}')
        logging.info(f'Original afm: n cells={afm.shape[0]}, n features={afm.shape[1]}.')
        
        # Cells from clone with at least min_cell_number cells, if necessary
        if min_cell_number>0 and lineage_column is not None:
            afm = filter_cell_clones(afm, column=lineage_column, min_cell_number=min_cell_number)
       
        # Baseline filter
        afm = filter_baseline(afm)
        logging.info(f'afm after baseline filter: n cells={afm.shape[0]}, n features={afm.shape[1]}.')

        test_pp = pp_method in ['maegatk', 'mito_preprocessing']

        if filtering == 'baseline' and test_pp:
            pass
        if filtering == 'CV' and test_pp:
            afm = filter_CV(afm, **filtering_kwargs)
        elif filtering == 'miller2022' and test_pp:
            afm = filter_miller2022(afm, **filtering_kwargs)
        elif filtering == 'weng2024' and test_pp:
            afm = filter_weng2024(afm, **filtering_kwargs)
        elif filtering == 'MQuad' and (test_pp or pp_method == 'cellsnp-lite'):
            afm = filter_MQuad(afm, ncores=ncores, path_=os.getcwd(), **filtering_kwargs)
        elif filtering == 'MI_TO' and test_pp:
            afm = filter_MI_TO(afm, **filtering_kwargs)
        elif filtering == 'GT_enriched' and test_pp:
            afm = filter_GT_enriched(afm, lineage_column=lineage_column, **filtering_kwargs)
        else:
            raise ValueError(
                f'''The provided filtering method {filtering} is not supported for pp_method {pp_method}
                    Choose another one...'''
            )

    elif pp_method in ['samtools', 'freebayes', 'cellsnp-lite'] and filtering is None:
        
        rows = cells if cells is not None else afm.obs_names 
        cols = variants if variants is not None else afm.var_names 
        afm = afm[rows,cols].copy()
        logging.info(f'No further filter applied after baseline.')
    
    elif cells is not None or variants is not None:
        
        logging.info(f'Filtering custom sets of cells and variants')
        rows = cells if cells is not None else afm.obs_names 
        cols = variants if variants is not None else afm.var_names 
        afm = afm[rows,cols].copy()
        logging.info(f'Filtered afm contains {afm.shape[0]} cells and {afm.shape[1]} MT-SNVs.')

    else:
        raise ValueError(
                f'''The provided filtering method {filtering} is not supported.
                    Choose another one...'''
            )

    # Filter common SNVs and possible RNA-edits
    n_dbSNP = np.nan
    if path_dbSNP is not None:
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
    if path_REDIdb is not None:
        if os.path.exists(path_REDIdb):
            edits = pd.read_csv(path_REDIdb, index_col=0, sep='\t')
            edits = edits.query('nSamples>100')
            edits = edits['Position'].astype('str') + '_' + edits['Ref'] + '>' + edits['Ed']
            edits = edits.to_list()
            n_REDIdb = afm.var_names.isin(edits).sum()
            logging.info(f'Exclude {n_REDIdb} common RNA editing events (REDIdb)')
            variants = afm.var_names[~afm.var_names.isin(edits)]
            afm = afm[:,variants].copy()


    # Filter cells with at least one muts above af_confident_detection
    call_genotypes(afm, bin_method=bin_method, **binarization_kwargs)
    afm = afm[np.sum(afm.layers['bin'].A>0, axis=1)>0,:]
 
    # Final dataset and filtered MT-SNVs metrics to evalutate the selected MT-SNVs space quality
    logging.info(f'Filtered afm contains {afm.shape[0]} cells and {afm.shape[1]} MT-SNVs.')
    if afm.shape[1] == 0:
        raise ValueError('No variant selected! Change filtering method!!')
 
    # Bimodal mixture modelling: deltaBIC (MQuad-like)
    if fit_mixtures:
        afm.var = afm.var.join(fit_MQuad_mixtures(afm, ncores=ncores).dropna()[['deltaBIC']])
 
    # Last (optional filters):
    if fit_mixtures and only_positive_deltaBIC:
        afm = afm[:,afm.var['deltaBIC']>0].copy()
    if max_AD_counts>1:
        afm = afm[:,np.max(afm.layers['AD'].A, axis=0)>=max_AD_counts].copy()
 
    # Final fixes
    call_genotypes(afm, bin_method=bin_method, **binarization_kwargs)
    afm = afm[np.sum(afm.layers['bin'].A>0, axis=1)>0,:]
    logging.info(f'Last optional filters: filtered afm contains {afm.shape[0]} cells and {afm.shape[1]} MT-SNVs.')
    
    # Lineage bias
    if lineage_column in afm.obs.columns and compute_enrichment:
        
        logging.info(f'Compute MT-SNVs enrichment for {lineage_column} categories')
        lineages = afm.obs[lineage_column].dropna().unique()
        for target_lineage in lineages:
            res = compute_lineage_biases(afm, lineage_column, target_lineage, 
                                        bin_method=bin_method, binarization_kwargs=binarization_kwargs)
            afm.var[f'FDR_{target_lineage}'] = res['FDR']
            afm.var[f'odds_ratio_{target_lineage}'] = res['odds_ratio']

    # Last dataset stats 
    logging.info(f'Add last metrics')
    tree_kwargs.update({'ncores':ncores})
    tree = compute_metrics_filtered(
        afm, 
        spatial_metrics=spatial_metrics, 
        tree_kwargs=tree_kwargs,
        bin_method=bin_method, 
        binarization_kwargs=binarization_kwargs
    )

    # Add params to .uns
    afm.uns['char_filter'] = {
        'lineage_column' : lineage_column, 
        'min_cell_number' : min_cell_number,
        'filtering' : filtering if (cells is not None) or (variants is not None) else 'predefined_sets',
        'only_positive_deltaBIC' : only_positive_deltaBIC,
        'compute_enrichment' : compute_enrichment,
        'spatial_metrics' : spatial_metrics,
        'n_dbSNP' : n_dbSNP,
        'n_REDIdb' : n_REDIdb,
    }
    afm.uns['char_filter'].update(filtering_kwargs)
    
    if return_tree:
        return afm, tree
    else:
        return afm


##


