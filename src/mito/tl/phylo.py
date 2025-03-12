"""
Phylogenetic inference.
"""

import logging
import numpy as np
import pandas as pd
import cassiopeia as cs
from ..pp.distances import call_genotypes, compute_distances


##


solver_d = {
    'UPMGA' : cs.solver.UPGMASolver,
    'NJ' : cs.solver.NeighborJoiningSolver,
    'spectral' : cs.solver.SpectralSolver,
    'shared_muts' : cs.solver.SharedMutationJoiningSolver,
    'greedy' : cs.solver.SpectralGreedySolver,
    'max_cut' : cs.solver.MaxCutGreedySolver,
}
    
##

_solver_kwargs = {
    'UPMGA' : {},
    'NJ' : {'add_root':True},
    'spectral' : {},
    'shared_muts' : {},
    'greedy' : {},
    'max_cut' : {}
}


##

def _initialize_CassiopeiaTree_kwargs(afm, distance_key, min_n_positive_cells, max_frac_positive, filter_muts=True):
    """
    Extract afm slots for CassiopeiaTree instantiation.
    """

    assert 'bin' in afm.layers or 'scaled' in afm.layers
    assert distance_key in afm.obsp

    layer = 'bin' if 'bin' in afm.layers else 'scaled'
    D = afm.obsp[distance_key].A.copy()
    D[np.isnan(D)] = 0
    D = pd.DataFrame(D, index=afm.obs_names, columns=afm.obs_names)
    M = pd.DataFrame(afm.layers[layer].A, index=afm.obs_names, columns=afm.var_names)
    if afm.X is not None:
        M_raw = pd.DataFrame(afm.X.A, index=afm.obs_names, columns=afm.var_names)
    else:
        M_raw = M.copy()

    # Remove variants from char matrix i) they are called in less than min_n_positive_cells or ii) > max_frac_positive 
    # We avoid recomputing distances as their contribution to the average pairwise cell-cell distance is minimal
    if filter_muts and afm.uns['scLT_system'] != 'Cas9':
        test_germline = ((M==1).sum(axis=0) / M.shape[0]) <= max_frac_positive
        test_too_rare = (M==1).sum(axis=0) >= min_n_positive_cells
        test = (test_germline) & (test_too_rare)
        M_raw = M_raw.loc[:,test].copy()
        M = M.loc[:,test].copy()

    return M_raw, M, D


##


def get_clades(tree, with_root=True, with_singletons=False):
    """
    Find all clades in a tree, from top to bottom
    """
    clades = { x : frozenset(tree.leaves_in_subtree(x)) for x in tree.internal_nodes }

    if not with_root:
        if 'root' in clades:
            del clades['root']

    if with_singletons:
        for x in tree.leaves:
            clades[x] = frozenset([x])

    return clades


##


def AFM_to_seqs(afm, bin_method='MiTo', binarization_kwargs={}):
    """
    Convert an AFM to a dictionary of sequences.
    """

    # Extract ref and alt character sequences
    L = [ x.split('_')[1].split('>') for x in afm.var_names ]
    ref = ''.join([x[0] for x in L])
    alt = ''.join([x[1] for x in L])

    if 'bin' not in afm.layers:
        call_genotypes(afm, bin_method=bin_method, **binarization_kwargs)

    # Convert to a dict of strings
    X_bin = afm.layers['bin'].A.copy()
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


def get_supports(tree):

    L = []
    for node in tree.internal_nodes:
        try:
            s = tree.get_attribute(node, 'support')
            s = s if s is not None else np.nan
            L.append(s)
        except:
            pass

    return np.array(L)


##


def get_internal_node_stats(tree):
    """
    Get internal nodes supports, time, mut status and clade size
    """

    clades = get_clades(tree)
    df = pd.DataFrame({
        'support' : get_supports(tree), 
        'time' : [ tree.get_time(node) for node in tree.internal_nodes ],
        'clade_size' : [ len(clades[node]) for node in tree.internal_nodes ]
        }, 
        index=tree.internal_nodes
    )
    if 'lca' in tree.cell_meta:
        clades = tree.cell_meta['lca'].loc[lambda x: ~x.isna()].unique()
        df['mut_clade'] = [ True if node in clades else False for node in tree.internal_nodes ]
    
    return df 


##


def build_tree(
    afm, precomputed=False, distance_key='distances', metric='jaccard', 
    bin_method='MiTo', solver='UPMGA', ncores=1, min_n_positive_cells=2, filter_muts=False,
    max_frac_positive=.95, binarization_kwargs={}, solver_kwargs={},
    ):
    """
    Wrapper around CassiopeiaTree distance-based and maximum parsimony solvers.
    """
    
    # Compute (if necessary, cell-cell distances, and retrieve necessary afm .slots)
    if precomputed:
        if distance_key in afm.obsp and precomputed:
            metric = afm.uns['distance_calculations'][distance_key]['metric']
            layer = afm.uns['distance_calculations'][distance_key]['layer']
            logging.info(f'Use precomputed distances: metric={metric}, layer={layer}')
            if layer == 'bin':
                bin_method = afm.uns['genotyping']['bin_method']
                binarization_kwargs = afm.uns['genotyping']['binarization_kwargs']
                logging.info(f'Precomputed bin layer: bin_method={bin_method} and binarization_kwargs={binarization_kwargs}')
    else:
        compute_distances(
            afm, distance_key=distance_key, metric=metric, 
            bin_method=bin_method, ncores=ncores, binarization_kwargs=binarization_kwargs
        )
    M_raw, M, D = _initialize_CassiopeiaTree_kwargs(afm, distance_key, min_n_positive_cells, max_frac_positive, filter_muts=filter_muts)
 
    # Solve cell phylogeny
    metric = afm.uns['distance_calculations'][distance_key]['metric']
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