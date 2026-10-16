"""
The clone-level tree: every clone of a phylogeny as one tip.
"""

import networkx as nx
import numpy as np
import pandas as pd
from cassiopeia.data import CassiopeiaTree

##


def collapse_clones(
    tree: CassiopeiaTree,
    key: str = 'MiTo_clone',
    show_unassigned: bool = False,
    support_key: str = 'support',
    ) -> CassiopeiaTree:
    """
    Collapse a phylogeny to one tip per clone.

    Every clade of `tree` whose cells all carry one label of `key` (other than "unassigned")
    becomes a single tip, attached where the clade is and at the depth of its cells, so
    every branch keeps its true length (and an ultrametric tree stays ultrametric).
    Unassigned cells are dropped, or kept as tips with `show_unassigned`. Clades left with a
    single child are merged into the branch above. A label whose cells are not one clade of the
    tree (several pieces) gives one tip per piece, named "<label>.1", "<label>.2", ...

    Parameters
    ----------
    tree : CassiopeiaTree
        Cell phylogeny, with the clone labels in tree.cell_meta[`key`] (e.g. after
        `mito.tl.annotate_clones`).
    key : str, optional
        Column of tree.cell_meta with the clone labels. Default is "MiTo_clone".
    show_unassigned : bool, optional
        Keep the unassigned cells as tips. Default is False.
    support_key : str, optional
        Node attribute with the clade support (`mito.tl.clade_support`). The support of a
        clone's clade is carried onto its tip, and that of the other nodes onto themselves.
        Default is "support".

    Returns
    -------
    CassiopeiaTree
        The clone-level tree. Its cell_meta has one row per tip, with the columns "clone"
        (the label), "n_cells", "support" (of the clone's clade, NaN if the tree has none)
        and "cells" (a tuple of the cells in the tip).
    """

    if key not in tree.cell_meta.columns:
        raise KeyError(f'"{key}" is not a column of tree.cell_meta: run mito.tl.annotate_clones first.')
    labels = tree.cell_meta[key].astype(str)

    label = {}
    for node in tree.depth_first_traverse_nodes(postorder=True):
        if tree.is_leaf(node):
            label[node] = None if labels.loc[node] in ('unassigned', 'nan') else labels.loc[node]
        else:
            kids = {label[c] for c in tree.children(node)}
            label[node] = kids.pop() if len(kids) == 1 else None
    depth = max(tree.get_time(leaf) for leaf in tree.leaves)

    graph, length, tip = nx.DiGraph(), {}, {}
    stack = [tree.root]
    while stack:
        node = stack.pop()
        for child in tree.children(node):
            if label[child] is not None:                      # a clone: one tip, at the depth of its cells
                graph.add_edge(node, child)
                length[(node, child)] = tree.get_branch_length(node, child) + depth - tree.get_time(child)
                tip[child] = (label[child], tuple(tree.leaves_in_subtree(child)))
            elif tree.is_leaf(child):                         # an unassigned cell
                if show_unassigned:
                    graph.add_edge(node, child)
                    length[(node, child)] = tree.get_branch_length(node, child)
                    tip[child] = (None, (child,))
            else:
                graph.add_edge(node, child)
                length[(node, child)] = tree.get_branch_length(node, child)
                stack.append(child)

    for node in [n for n in nx.dfs_postorder_nodes(graph, tree.root) if graph.out_degree(n) == 0 and n not in tip]:
        graph.remove_node(node)                               # clades left without tips
    for node in [n for n in nx.dfs_postorder_nodes(graph, tree.root)
                 if n in graph and graph.out_degree(n) == 1 and n != tree.root]:
        (child,) = graph.successors(node)
        (parent,) = graph.predecessors(node)
        graph.add_edge(parent, child)
        length[(parent, child)] = length[(parent, node)] + length[(node, child)]
        graph.remove_node(node)

    # tip names: the clone label, ".k" for the pieces of a label that is not one clade
    names, seen = {}, {}
    for t, (lab, cells) in tip.items():
        if lab is None:
            names[t] = t
        else:
            seen[lab] = seen.get(lab, 0) + 1
    count = dict(seen)
    used = dict.fromkeys(seen, 0)
    for t, (lab, cells) in tip.items():
        if lab is not None:
            used[lab] += 1
            names[t] = lab if count[lab] == 1 else f'{lab}.{used[lab]}'
    graph = nx.relabel_nodes(graph, names)

    def support_of(node):
        try:
            return float(tree.get_attribute(node, support_key))
        except Exception:  # noqa: BLE001
            return np.nan

    rows = {names[t]: dict(clone=lab, n_cells=len(cells), support=support_of(t) if len(cells) > 1 else np.nan,
                           cells=cells) for t, (lab, cells) in tip.items()}
    meta = pd.DataFrame.from_dict(rows, orient='index')
    flat = CassiopeiaTree(tree=graph, cell_meta=meta)
    for (u, v), l in length.items():
        a, b = names.get(u, u), names.get(v, v)
        if graph.has_edge(a, b):
            flat.set_branch_length(a, b, l)
    for node in flat.internal_nodes:
        s = support_of(node)
        if not np.isnan(s):
            flat.set_attribute(node, support_key, s)
    for t, (lab, cells) in tip.items():
        flat.set_attribute(names[t], 'n_cells', len(cells))
        if lab is not None and len(cells) > 1 and not np.isnan(support_of(t)):
            flat.set_attribute(names[t], support_key, support_of(t))

    return flat
