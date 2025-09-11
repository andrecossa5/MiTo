"""
Utils for DE, GSEA analysis.
"""

import numpy as np
import pandas as pd
import gseapy
from typing import List
from anndata import AnnData
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, leaves_list


##


def format_rank_genes_groups(
    adata: AnnData, 
    key: str = 'rank_genes_groups', 
    filter_genes: bool = False,
    rank_by: str = 'log2FC',
    max_pval_adj: float = .05, 
    min_log2FC: float = 1, 
    min_pct_group: float = .5, 
    max_pct_rest: float = .5
    ) -> pd.DataFrame :

    L = []
    cats = adata.uns[key]['names'].dtype.names
    group_col = adata.uns[key]['params']['groupby']
    
    for cat in cats:
        genes = adata.uns[key]['names'][cat]
        df_ = pd.DataFrame({
            'gene' : genes,
            'score' : adata.uns[key]['scores'][cat],
            'log2FC' : adata.uns[key]['logfoldchanges'][cat],
            'pval_adj' : adata.uns[key]['pvals_adj'][cat]
        })
        df_['pct_group'] = adata.uns[key]['pts'][cat].loc[genes].values
        df_['pct_rest'] = adata.uns[key]['pts_rest'][cat].loc[genes].values
        df_['group'] = cat
        cell_group = adata.obs_names[adata.obs[group_col] == cat]
        cell_rest = adata.obs_names[adata.obs[group_col] != cat]
        df_['mean_exp_group'] = adata[cell_group, df_['gene']].X.toarray().mean(axis=0)
        df_['mean_exp_rest'] = adata[cell_rest, df_['gene']].X.toarray().mean(axis=0)
        
        if filter_genes:
            df_ = df_.query('log2FC>=@min_log2FC and pct_group>@min_pct_group and pct_rest<=@max_pct_rest')
            df_ = df_.query('pval_adj<=@max_pval_adj')
            df_ = df_.sort_values(rank_by, ascending=False)

        L.append(df_)

    df_results = pd.concat(L)

    return df_results


##


def order_groups(adata, groupby=None, obsm_key='X_pca', n_dims=15):
    """
    Sort groups in adata.obs[groupby].
    """

    X_pca_agg = (
        pd.DataFrame(adata.obsm[obsm_key][:,:n_dims])
        .assign(group=adata.obs[groupby].astype('str').values)
        .groupby('group').median()
    )
    order_groups = (
        X_pca_agg.index.values
        [leaves_list(linkage(pairwise_distances(X_pca_agg)))]
    )

    return order_groups


##


def get_top_markers(df, groupby='group', sort_by='log2FC', ascending=False, order_groups=None, ntop=3):
    """
    Get top markers for groups in adata.obs[groupby], according to order_groups.
    """

    top_markers = []
    for cat in order_groups:
        markers = (
            df.loc[df[groupby]==cat]
            .sort_values(sort_by, ascending=ascending)
            .head(ntop)
            ['gene'].to_list()
        )
        top_markers.extend(markers)

    return top_markers



##


def run_GSEA(
    ranked_list: pd.Series, 
    collections: str|List[str] = 'MSigDB_Hallmark_2020',
    max_pval_adj: float = .01,
    min_size_set: int = 15,
    max_size_set: int =  1000
    ) -> pd.DataFrame :
    """
    Fast GSEA.
    """

    # names = pd.Series(gseapy.get_library_name())
    # names[names.str.contains('Hall')]

    L = collections if isinstance(collections, list) else [collections]
    results = gseapy.prerank(
        rnk=ranked_list,
        gene_sets=L,
        threads=-1,
        min_size=min_size_set,
        max_size=max_size_set,
        permutation_num=200, 
        outdir=None, 
        seed=1234,
        verbose=True,
    )
    df = (
        results.res2d
        [[ 'Term', 'ES', 'NES', 'FDR q-val', 'Lead_genes' ]]
        .rename(columns={'FDR q-val' : 'pval_adj'})
        .query('pval_adj<=@max_pval_adj')
        .sort_values('NES', ascending=False)
    )

    return results, df


##


def run_ORA(
    gene_list: List[str], 
    collections: str|List[str] = 'MSigDB_Hallmark_2020',
    max_pval_adj: float = .01
    ) -> pd.DataFrame :

    L = collections if isinstance(collections, list) else [collections]
    results = gseapy.enrichr(
        gene_list=gene_list,
        gene_sets=L,
        cutoff=.1,
        no_plot=True,
        outdir=None, 
    ).results
    df = (
        results
        [[ 'Term', 'Overlap', 'Odds Ratio', 'Adjusted P-value', 'Genes' ]]
        .rename(columns={'Adjusted P-value' : 'pval_adj'})
        .query('pval_adj<=@max_pval_adj')
        .sort_values('Odds Ratio', ascending=False)
    )

    return results, df


##