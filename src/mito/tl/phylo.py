"""
Phylogenetic inference.
"""

import logging
from typing import Any

import cassiopeia as cs
import numpy as np
import pandas as pd
from anndata import AnnData
from cassiopeia.data import CassiopeiaTree
from scipy.sparse import csr_matrix

from mito.pp.distances import compute_distances

##


solver_d = {
    'UPMGA' : cs.solver.UPGMASolver,
    'NJ' : cs.solver.NeighborJoiningSolver,
    'spectral' : cs.solver.SpectralSolver,
    'greedy' : cs.solver.SpectralGreedySolver,
}

##

_solver_kwargs = {
    'UPMGA' : {},
    'NJ' : {'add_root':True},
    'spectral' : {},
    'greedy' : {},
}


##

def _initialize_CassiopeiaTree_kwargs(afm, distance_key, min_n_positive_cells, max_frac_positive, filter_muts=True):
    """
    Extract afm slots for CassiopeiaTree instantiation.
    """

    assert 'bin' in afm.layers or 'scaled' in afm.layers
    assert distance_key in afm.obsp

    layer = 'bin' if 'bin' in afm.layers else 'scaled'
    D = afm.obsp[distance_key].toarray()
    D[np.isnan(D)] = 0
    D = pd.DataFrame(D, index=afm.obs_names, columns=afm.obs_names)
    M = pd.DataFrame(afm.layers[layer].toarray(), index=afm.obs_names, columns=afm.var_names)
    if afm.X is not None:
        M_raw = pd.DataFrame(afm.X.toarray(), index=afm.obs_names, columns=afm.var_names)
    else:
        M_raw = M.copy()

    # Remove variants from char matrix i) they are called in less than min_n_positive_cells or ii) > max_frac_positive
    # We avoid recomputing distances as their contribution to the average pairwise cell-cell distance is minimal
    if filter_muts:
        test_germline = ((M==1).sum(axis=0) / M.shape[0]) <= max_frac_positive
        test_too_rare = (M==1).sum(axis=0) >= min_n_positive_cells
        test = (test_germline) & (test_too_rare)
        M_raw = M_raw.loc[:,test].copy()
        M = M.loc[:,test].copy()

    return M_raw, M, D


##


def AFM_to_seqs(afm: AnnData) -> dict[str,str]:
    """
    Convert an AFM to a dictionary of sequences.
    """

    # Extract ref and alt character sequences
    L = [ x.split('_')[1].split('>') for x in afm.var_names ]
    ref = ''.join([x[0] for x in L])
    alt = ''.join([x[1] for x in L])

    if 'bin' not in afm.layers:
        raise ValueError('AFM_to_seqs needs genotypes in afm.layers["bin"]: run mito.pp.call_genotypes.')

    # Convert to a dict of strings
    X_bin = afm.layers['bin'].toarray()
    d = {}
    for i, cell in enumerate(afm.obs_names):
        m_ = X_bin[i,:]
        seq = []
        for j, char in enumerate(m_):
            if char == 1:
                seq.append(alt[j])
            elif char == 0:
                seq.append(ref[j])
            else:
                seq.append('N')
        d[cell] = ''.join(seq)

    return d


##


def build_tree(
    afm: AnnData,
    distance_key: str = 'distances',
    metric: str = 'weighted_jaccard',
    solver: str = 'UPMGA',
    ncores: int = 1,
    min_n_positive_cells: int = 2,
    filter_muts: bool = False,
    max_frac_positive: float = .95,
    solver_kwargs: dict[str,Any] = None,
    ) -> CassiopeiaTree:
    """
    Wrapper around cassiopeia lineage solvers. MW Jones et al., 2020.

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix.
    distance_key : str, optional
        Key in afm.obsp where distances are stored. Default is "distances".
    metric : str, optional
        Distance metric, if distances need to be computed. Default is "weighted_jaccard".
    solver : str, optional
        Lineage solver to use. Default is "UPMGA".
    ncores : int, optional
        Number of cores to use for computation. Default is 1.
    min_n_positive_cells : int, optional
        Minimum number of positive cells required. Default is 2.
    filter_muts : bool, optional
        Whether to filter mutations. Default is False.
    max_frac_positive : float, optional
        Maximum fraction of positive cells allowed. Default is 0.95.
    solver_kwargs : dict, optional
        Additional keyword arguments for the solver. Default is {}.

    Returns
    -------
    CassiopeiaTree
        Solved single-cell phylogeny.
    """

    # Cell-cell distances: reuse them if `mito.pp.filter_afm` (or the user) already
    # computed them, otherwise compute them here from the genotypes in the bin layer.
    if solver_kwargs is None:
        solver_kwargs = {}
    if distance_key in afm.obsp:
        metric = afm.uns['distances'][distance_key]['metric']
        layer = afm.uns['distances'][distance_key]['layer']
        logging.info(f'Use precomputed distances: metric={metric}, layer={layer}')
    else:
        compute_distances(afm, distance_key=distance_key, metric=metric, ncores=ncores)

    # Init
    M_raw, M, D = _initialize_CassiopeiaTree_kwargs(
        afm, distance_key, min_n_positive_cells, max_frac_positive, filter_muts=filter_muts
    )

    # Solve cell phylogeny
    metric = afm.uns['distances'][distance_key]['metric']
    logging.info(f'Build tree: metric={metric}, solver={solver}')
    np.random.seed(1234)
    tree = cs.data.CassiopeiaTree(character_matrix=M, dissimilarity_map=D, cell_meta=afm.obs)
    _solver = solver_d[solver]
    kwargs = _solver_kwargs[solver]
    kwargs.update(solver_kwargs)
    solver = _solver(**kwargs)
    solver.solve(tree)

    # Add layers to CassiopeiaTree
    tree.layers['raw'] = M_raw
    tree.layers['transformed'] = M

    return tree


##


def coarse_grained_tree(tree: CassiopeiaTree, groupby: str) -> CassiopeiaTree:
    """
    Take a full cell phylogeny and coarse-grained it into a clone or
    "groupby" phylogeny.
    """

    meta = tree.cell_meta.copy()
    X_raw = tree.layers['raw'].copy()
    X_raw_agg = X_raw.join(meta[[groupby]]).groupby(groupby).median()
    X_bin_agg = (X_raw_agg>0).astype(int)
    muts = X_bin_agg.sum(axis=0).loc[lambda x: x>0].index
    X_raw_agg = X_raw_agg[muts].copy()
    X_bin_agg = X_bin_agg[muts].copy()

    afm_agg = AnnData(
        X=csr_matrix(X_raw_agg.values),
        obs=pd.DataFrame(index=X_raw_agg.index),
        layers={'bin':csr_matrix(X_bin_agg.values)},
    )
    compute_distances(afm_agg)
    tree_agg = build_tree(afm_agg)

    return tree_agg


##


def _get_leaves_order(tree):
    order = []
    for node in tree.depth_first_traverse_nodes():
        if node in tree.leaves:
            order.append(node)
    return order


##


def _get_muts_order(tree: CassiopeiaTree) -> list:
    """
    Diagonal order of the characters, for plotting: each character is placed at the clade
    it marks best (highest prevalence inside against outside), and the clades are visited
    depth-first. Characters that mark no clade are appended, least prevalent first.
    """

    M = tree.layers['transformed']
    cells = list(M.index)
    pos = {c:i for i, c in enumerate(cells)}
    B = (M.values>0)
    n, m = B.shape

    best_node = np.full(m, None, dtype=object)
    best_score = np.full(m, -np.inf)
    for node in tree.depth_first_traverse_nodes():
        lv = [pos[c] for c in tree.leaves_in_subtree(node) if c in pos]
        if len(lv) < 2 or len(lv) == n:
            continue
        mask = np.zeros(n, bool)
        mask[lv] = True
        score = B[mask].mean(0) - B[~mask].mean(0)
        test = score > best_score
        best_score[test] = score[test]
        best_node[test] = node

    order = []
    placed = np.zeros(m, bool)
    for node in tree.depth_first_traverse_nodes():
        idx = np.flatnonzero((best_node == node) & (best_score > 0) & ~placed)
        for j in idx[np.argsort(-best_score[idx])]:
            order.append(M.columns[j])
            placed[j] = True

    rest = np.flatnonzero(~placed)
    order += list(M.columns[rest[np.argsort(B[:,rest].sum(0))]])

    return order


##
