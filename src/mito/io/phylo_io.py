"""
I/O functions to read/write CassiopeiaTrees from annotated (supports) .newick strigs.
"""

import pandas as pd
from cassiopeia.data import CassiopeiaTree
from Bio.Phylo.NewickIO import Parser
from io import StringIO
import networkx as nx


##


def _add_edges(G, clade, parent=None, counter=[1]):
    """
    Update the binary graph recursively.
    """
    if clade.is_terminal():
        node_name = clade.name
    else:
        node_name = f"internal_{counter[0]}"
        counter[0] += 1

    G.add_node(node_name, support=clade.confidence)
    if parent:
        branch_length = clade.branch_length if clade.branch_length is not None else 1
        G.add_edge(parent, node_name, length=branch_length)
    for child in clade.clades:
        _add_edges(G, child, node_name, counter)


##


def read_newick(
    path, X_raw: pd.DataFrame = None, X_bin: pd.DataFrame = None, 
    D: pd.DataFrame = None, meta: pd.DataFrame = None
    ) -> CassiopeiaTree:
    """
    Read a newick string as CassiopeiaTree object.

    Parameters
    ----------
    path : str
        Path to newick string.
    X_raw : pd.DataFrame, optional
        Raw allelic frequency table. Cell x variants. Default is None.
    X_bin : pd.DataFrame, optional
        Binary (1,0) cell genotypes. Cell x variants. Default is None.
    D : pd.DataFrame, optional
        Cell x cell distance matrix. Default is None.
    meta : pd.DataFrame, optional
        Cell metadata. Cell x covariates. Default is None.

    Returns
    -------
    afm : AnnData
        The assembled Allele Frequency Matrix (AFM).
    """

    with open(path, 'r') as f:
        newick = f.read().strip()

    parser = Parser(StringIO(newick))
    original_tree = list(parser.parse())[0]

    G = nx.DiGraph()
    _add_edges(G, original_tree.root, counter=[1])

    edge_list = []
    for u, v, data in G.edges(data=True):
        length = data['length'] if 'length' in data else 0.0
        edge_list.append((u, v, length))

    cells = [ x for x in G.nodes if not x.startswith('internal') ]
    cassiopeia_tree = CassiopeiaTree(
        tree=G, 
        character_matrix=X_bin.loc[cells,:] if X_bin is not None else None, 
        dissimilarity_map=D.loc[cells,cells] if D is not None else None, 
        cell_meta=meta.loc[cells,:] if meta is not None else None
    )
    if X_raw is not None and X_bin is not None:
        cassiopeia_tree.layers['raw'] = X_raw.loc[cells,:]
        cassiopeia_tree.layers['transformed'] = X_bin.loc[cells,:]

    for u, v, length in edge_list:
        cassiopeia_tree.set_branch_length(u, v, length)
    for node in G.nodes:
        if 'support' in G.nodes[node]:
            support = G.nodes[node]['support']
            cassiopeia_tree.set_attribute(node, 'support', support)

    return cassiopeia_tree


##


def to_DiGraph(tree: CassiopeiaTree) -> nx.DiGraph:
    """
    Create a nx.DiGraph from annotated (i.e., support, for now) CassiopeiaTree.
    """
    G = nx.DiGraph()
    for node in tree.nodes:
        try:
            G.add_node(node, support=tree.get_attribute(node, 'support'))
        except:
            pass
    for u, v, in tree.edges:
        G.add_edge(u, v, branch_length=tree.get_branch_length(u, v))

    return G
    

##


def _to_newick_str(g, node):

    is_leaf = g.out_degree(node) == 0
    branch_length_str = ""
    support_str = ""

    if g.in_degree(node) > 0:
        parent = list(g.predecessors(node))[0]
        branch_length_str = ":" + str(g[parent][node]["branch_length"])

    if 'support' in g.nodes[node] and g.nodes[node]['support'] is not None:
        try:
            support_str = str(int(g.nodes[node]['support']))
        except:
            support_str = "0"

    _name = str(node)
    return (
        "%s" % (_name,) + branch_length_str
        if is_leaf
        else (
            "("
            + ",".join(
                _to_newick_str(g, child) for child in g.successors(node)
            )
            + ")" + (support_str if support_str else "") + branch_length_str
        )
    )


##


def write_newick(tree: CassiopeiaTree, path: str):
    """
    Write a CassiopeiaTree as a newick string.

    Parameters
    ----------
    tree : CassiopeiaTree
        Tree to write.
    path : str
        Path to newick string.
    """
    
    G = to_DiGraph(tree)
    root = [ node for node in G if G.in_degree(node) == 0 ][0]
    newick = f'{_to_newick_str(G, root)};'
    with open(path, 'w') as f:
        f.write(newick)


##