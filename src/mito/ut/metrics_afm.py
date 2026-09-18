"""
Descriptive metrics of an AFM. Reporting only: nothing in the pipeline reads them back,
so they are returned rather than stored.
"""

import numpy as np
import pandas as pd
from anndata import AnnData
from igraph import Graph

from .positions import transitions, transversions

##


def _connectivity_metrics(X):
    """
    Connectivity of the cell-cell graph induced by shared characters (Weng et al., 2024).
    """

    A = np.dot(X, X.T)
    np.fill_diagonal(A, 0)
    g = Graph.Adjacency((A>0).tolist(), mode='undirected')
    g.es['weight'] = [ A[i][j] for i, j in g.get_edgelist() ]

    average_degree = sum(g.degree()) / g.vcount()
    if g.is_connected():
        average_path_length = g.average_path_length()
    else:
        average_path_length = g.clusters().giant().average_path_length()
    transitivity = g.transitivity_undirected()
    proportion_largest_component = max(g.clusters().sizes()) / g.vcount()

    return average_degree, average_path_length, transitivity, proportion_largest_component


##


def dataset_metrics(afm: AnnData, connectivity: bool = False) -> pd.Series:
    """
    Descriptive statistics of a filtered AFM: size, per-cell and per-variant character
    counts, sparsity, genotype redundancy and mutational spectrum.

    Parameters
    ----------
    afm : AnnData
        Filtered AFM, genotyped (i.e. with a "bin" layer).
    connectivity : bool, optional
        Also compute the connectivity metrics of the shared-character graph. They are
        O(n_cells^2) and rarely needed. Default is False.

    Returns
    -------
    pd.Series
        The metrics. Store them yourself if you need them (e.g. in .uns).
    """

    if 'bin' not in afm.layers:
        raise ValueError('dataset_metrics needs genotypes in afm.layers["bin"].')

    B = afm.layers['bin']
    B = (B.toarray() if hasattr(B, 'toarray') else np.asarray(B))>0
    d = {}

    d['n_cells'], d['n_vars'] = B.shape
    d['median_n_vars_per_cell'] = np.median(B.sum(axis=1))
    d['mean_n_vars_per_cell'] = np.mean(B.sum(axis=1))
    d['std_n_vars_per_cell'] = np.std(B.sum(axis=1))
    d['mean_n_cells_per_var'] = np.mean(B.sum(axis=0))
    d['median_n_cells_per_var'] = np.median(B.sum(axis=0))
    d['std_n_cells_per_var'] = np.std(B.sum(axis=0))
    d['density'] = B.sum() / (B.shape[0]*B.shape[1])

    # Genotype redundancy: how often two cells share the exact same character profile
    genomes = pd.Series([ ''.join(row) for row in B.astype(np.int8).astype(str) ])
    occurrences = genomes.value_counts(normalize=True)
    d['genomes_redundancy'] = 1-(occurrences.size / B.shape[0])
    d['median_genome_prevalence'] = occurrences.median()

    # Coverage
    if 'DP' in afm.layers:
        DP = afm.layers['DP']
        d['median_site_cov'] = float(np.median(DP.toarray() if hasattr(DP, 'toarray') else np.asarray(DP)))

    metrics = pd.Series(d)

    # Mutational spectrum (MT-SNVs named <pos>_<ref>><alt>)
    if afm.var_names.str.contains('_').all():
        classes = afm.var_names.map(lambda x: x.split('_')[1]).value_counts().astype(int)
        classes.index = classes.index.map(lambda x: f'mut_class_{x}')
        n_ti = classes.loc[classes.index.str.contains('|'.join(transitions))].sum()
        n_tv = classes.loc[classes.index.str.contains('|'.join(transversions))].sum()
        metrics = pd.concat([
            metrics, classes,
            pd.Series({'transitions_vs_transversions_ratio':n_ti/n_tv if n_tv else np.nan})
        ])

    if connectivity:
        degree, path_length, transitivity_, largest = _connectivity_metrics(B.astype(float))
        metrics = pd.concat([metrics, pd.Series({
            'average_degree':degree, 'average_path_length':path_length,
            'transitivity':transitivity_, 'proportion_largest_component':largest
        })])

    return metrics


##
