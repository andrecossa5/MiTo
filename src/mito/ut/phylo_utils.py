"""
Utils for mito.tl.phylo functions.
"""

import numpy as np
import pandas as pd
from cassiopeia.data import CassiopeiaTree


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


def get_internal_node_feature(tree: CassiopeiaTree, feature: str) -> np.array:
    """
    Extract internal node feature `feature`.
    """

    L = []
    for node in tree.internal_nodes:
        try:
            s = tree.get_attribute(node, feature)
            s = s if s is not None else np.nan
            L.append(s)
        except:
            L.append(np.nan)

    return np.array(L)


##


def get_internal_node_stats(tree: CassiopeiaTree):
    """
    Get internal nodes stats (i.e, time, clade_size, support, expansion_pvalue, 
    fitness scores and average cell similarity).
    """

    clades = get_clades(tree)
    df = pd.DataFrame({ 
            'time' : [ tree.get_time(node) for node in tree.internal_nodes ],
            'clade_size' : [ len(clades[node]) for node in tree.internal_nodes ],
            'support' : get_internal_node_feature(tree, 'support'),
            'expansion_pvalue' : get_internal_node_feature(tree, 'expansion_pvalue'),
            'fitness' : get_internal_node_feature(tree, 'fitness'),
            'similarity' : get_internal_node_feature(tree, 'similarity'),
        }, 
        index=tree.internal_nodes
    )
    if 'lca' in tree.cell_meta:
        clades = tree.cell_meta['lca'].loc[lambda x: ~x.isna()].unique()
        df['clonal_node'] = [ True if node in clades else False for node in tree.internal_nodes ]
    
    return df 


##