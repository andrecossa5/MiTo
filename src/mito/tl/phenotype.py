"""
Tools to map phenotype to lineage structures.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any, Iterable
from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools import score_small_parsimony
from scipy.stats import fisher_exact
from statsmodels.sandbox.stats.multicomp import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf
from ..ut.phylo_utils import get_internal_node_stats


##


def compute_clonal_fate_bias(
    tree: CassiopeiaTree, 
    state_column: str, 
    clone_column: str, 
    target_state: str|Any
    ) -> pd.DataFrame:
    """
    Compute -log10(FDR) Fisher's exact test: clonal fate biases towards some target_state.
    """

    n = len(tree.leaves)
    clones = np.sort(tree.cell_meta[clone_column].unique())

    target_ratio_array = np.zeros(clones.size)
    oddsratio_array = np.zeros(clones.size)
    pvals = np.zeros(clones.size)

    # Here we go
    for i, clone in enumerate(clones):

        test_clone = tree.cell_meta[clone_column] == clone
        test_state = tree.cell_meta[state_column] == target_state

        clone_size = test_clone.sum()
        clone_state_size = (test_clone & test_state).sum()
        target_ratio = clone_state_size / clone_size
        target_ratio_array[i] = target_ratio
        other_clones_state_size = (~test_clone & test_state).sum()

        # Fisher
        oddsratio, pvalue = fisher_exact(
            [
                [clone_state_size, clone_size - clone_state_size],
                [other_clones_state_size, n - other_clones_state_size],
            ],
            alternative='greater',
        )
        oddsratio_array[i] = oddsratio
        pvals[i] = pvalue

    # Correct pvals --> FDR
    pvals = multipletests(pvals, alpha=0.05, method="fdr_bh")[1]

    # Results
    results = pd.DataFrame({
        'perc_in_target_state' : target_ratio_array,
        'odds_ratio' : oddsratio_array,
        'FDR' : pvals,
        'fate_bias' : -np.log10(pvals) 
    }).sort_values('fate_bias', ascending=False)

    return results


##


def compute_scPlasticity(tree: CassiopeiaTree, meta_column: str):
    """
    Compute scPlasticity as in Yang et al., 2022.
    https://www.sc-best-practices.org/trajectories/lineage_tracing.html#
    """

    # Format column of interest
    tree.cell_meta[meta_column] = pd.Categorical(tree.cell_meta[meta_column])
    # parsimony = score_small_parsimony(tree, meta_item=meta_column)

    # compute plasticities for each node in the tree
    for node in tree.depth_first_traverse_nodes():
        effective_plasticity = score_small_parsimony(
            tree, meta_item=meta_column, root=node
        )
        size_of_subtree = len(tree.leaves_in_subtree(node))
        tree.set_attribute(
            node, "effective_plasticity", effective_plasticity / size_of_subtree
        )

    tree.cell_meta["scPlasticity"] = 0
    for leaf in tree.leaves:
        plasticities = []
        parent = tree.parent(leaf)
        while True:
            plasticities.append(tree.get_attribute(parent, "effective_plasticity"))
            if parent == tree.root:
                break
            parent = tree.parent(parent)

        tree.cell_meta.loc[leaf, "scPlasticity"] = np.mean(plasticities)


##


def nb_regression(df: pd.DataFrame, features: Iterable[str], predictor: str) -> pd.DataFrame:
    """
    Negative binomial regression approach to associate 
    clonal-level features to gene expression.

    Parameters
    ----------
    afm : pd.DataFrame (clone/sample x features/covariates)
        Input data table. Contains raw counts for all genes, and covariates of interest.
    features : list, str
        List of variables to test the GLM model coefficients on.
    predictor : str
        Model specification via formula interface. Formula is in the form:
        "gene ~ predictor". Example predictor: "fitness + counts"

    Returns
    -------
    AnnData
        Filtered Allelic Frequency Matrix.
    """

    L = []
    for gene in tqdm(features, total=len(features), desc="Features"):
        try:
            formula = f'{gene} ~ {predictor}'
            family = sm.families.NegativeBinomial()
            model = (
                smf.glm(formula=formula, data=df, family=family)
                .fit()
            )
            df_ = (
                model.params.to_frame('coef')
                .join(model.pvalues.to_frame('pval'))
                .reset_index(names='param')
                .query('param!="Intercept"')
                .assign(gene=gene)
            )
            L.append(df_)
        except:
            pass

    results = pd.concat(L)
    results['-logp10'] = -np.log10(results['pval'])
    results = results[['gene', 'param', 'coef', 'pval', '-logp10']]

    return results


##


def _find_partitions(df, n_cells):

    partitions = []
    i = 0
    while i<=df.shape[0]:
        partitions.append(
            df.iloc[i:min(i+n_cells,df.shape[0]),:].index
        )
        i += n_cells
    
    return partitions


##


def agg_pseudobulk(tree, adata, agg_method='mean', min_n_cells=10, n_cells=None, n_samples=None):
    """
    Aggragete expression data into psudobulk samples.
    """
    meta = tree.cell_meta.copy()
    top_clones = meta['MiTo clone'].value_counts().loc[lambda x: x>=min_n_cells].index.to_list()
    meta_top = meta.loc[meta['MiTo clone'].isin(top_clones)]
    meta_top['MiTo clone'] = meta_top['MiTo clone'].astype('str')
    cells = meta_top.index

    if n_cells is None and n_samples is None:
        agg = (
            pd.DataFrame(
                adata[cells,:].layers['raw'].toarray(), 
                index=cells, columns=adata.var_names
            )
            .join(meta_top[['MiTo clone']])
            .groupby('MiTo clone')
            .agg(agg_method)
            .round()
        )  
    
    elif n_cells is not None and n_samples is not None:

        pseudobulk_samples = []
        for clone in top_clones:
            df_ = meta_top.loc[meta_top['MiTo clone']==clone]
            for i in range(n_samples):
                cells = df_.sample(n_cells).index
                profile = ( 
                    pd.DataFrame(
                        adata[cells,:].layers['raw'].toarray(), 
                        index=cells, columns=adata.var_names
                    )
                    .agg(agg_method, axis=0)
                    .round()
                    .to_frame('counts')
                    .reset_index(names='gene')
                    .assign(sample=f'{clone}_{i}')
                )   
                pseudobulk_samples.append(profile)

        agg = (
            pd.concat(pseudobulk_samples)
            .pivot_table(index='sample', columns='gene', values='counts')
        )
        
    elif n_cells is not None and n_samples is None:
        
        pseudobulk_samples = []
        for clone in top_clones:
            df_ = meta_top.loc[meta_top['MiTo clone']==clone]
            partitions = _find_partitions(df_, n_cells)
            for i,cells in enumerate(partitions):
                profile = ( 
                    pd.DataFrame(
                        adata[cells,:].layers['raw'].toarray(), 
                        index=cells, columns=adata.var_names
                    )
                    .agg(agg_method, axis=0)
                    .round()
                    .to_frame('counts')
                    .reset_index(names='gene')
                    .assign(sample=f'{clone}_{i}')
                )   
                pseudobulk_samples.append(profile)

        agg = (
            pd.concat(pseudobulk_samples)
            .pivot_table(index='sample', columns='gene', values='counts')
        )
    else:
        raise ValueError('Wrong combo of n_cells, n_samples')

    # Filter genes and add counts columns
    if agg_method == 'sum':   
        total_clone_counts = agg.sum(axis=1) 
        agg_norm = agg.apply(lambda x: x/(total_clone_counts+1)*10**6, axis=0)
        norm_mean_expression = agg_norm.mean(axis=0)
        test = (norm_mean_expression >= np.percentile(norm_mean_expression, 10))
        agg = agg.loc[:,test].copy()
        agg['counts'] = total_clone_counts
    
    elif agg_method == 'mean':
        mean_expression = agg.mean(axis=0)
        test = mean_expression > 0
        agg = agg.loc[:,test]
        agg['counts'] = agg.mean(axis=1) 


    # Add clone level covariates, and re-scale them
    clone_features = (
        get_internal_node_stats(tree)
        .loc[lambda x: x['clonal_node']].reset_index(names='lca')
        .merge(tree.cell_meta[['MiTo clone', 'lca', 'n cells']], on='lca')
        .drop_duplicates()
        .loc[lambda x: x['MiTo clone'].isin(top_clones)]
        [['MiTo clone', 'fitness', 'clade_size']]
        .drop_duplicates()
        .reset_index(drop=True)
        .set_index('MiTo clone')
    )
    if n_cells is not None or n_samples is not None:
        agg['MiTo clone'] = agg.index.map(lambda x: x.split('_')[0])
        agg = (
            agg
            .reset_index().set_index('MiTo clone')
            .join(clone_features)
            .reset_index().set_index('sample')
            .drop(columns=['MiTo clone'])
        )
    else:
        agg = agg.join(clone_features)

    # Rescale predictor variables
    rescale = lambda x: (x-x.mean()) / x.std()
    agg['fitness'] = rescale(agg['fitness'])
    agg['clade_size'] = rescale(agg['clade_size'])
    agg['counts'] = rescale(agg['counts'])

    return agg


##