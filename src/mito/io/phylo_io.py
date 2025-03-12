"""
I/O functions to read/write CassiopeiaTrees from annotated (supports) .newick strigs.
"""

import anndata
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


def read_newick(path, X_raw=None, X_bin=None, D=None, meta=None) -> CassiopeiaTree:
    """
    Read an newick string as a CassiopeiaTree object.
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


def write_newick(tree: CassiopeiaTree, path=None):
    """
    Write a cassiopeia tree as newick.
    """
    G = to_DiGraph(tree)
    root = [ node for node in G if G.in_degree(node) == 0 ][0]
    newick = f'{_to_newick_str(G, root)};'
    with open(path, 'w') as f:
        f.write(newick)


##


def create_from_annot(nodes: pd.DataFrame, edges: pd.DataFrame, afm: anndata.AnnData) -> CassiopeiaTree:
    """
    Helper function to reate a CassiopeiaTree from two nodes and edges dfs. 
    Set nodes and branch attributes from scratch.
    The nodes df needs to have columns:
    - node
    - cell
    - support
    - clonal_ancestor
    - clone
    - assigned_var
    - p
    - af1
    - af0
    - p1
    - p0
    Repeated nodes (i.e., multiple assigned MT-SNVs) and NaNs are allowed.
    """

    G = nx.DiGraph()

    # Fix nodes names
    map_leaves = { k:v for k,v in zip(nodes['node'], nodes['cell']) if isinstance(v, str) }
    mapping = lambda x: map_leaves[x] if x in map_leaves else x
    edges['u'] = edges['u'].map(mapping)
    edges['v'] = edges['v'].map(mapping)
    nodes['node'] = nodes['node'].map(mapping)

    for node in nodes['node'].unique():
        attributes = {
            **nodes.loc[nodes['node']==node].iloc[:,:5].drop_duplicates().iloc[0,:].to_dict(),
            **{ 'vars' : nodes.loc[nodes['node']==node].iloc[:,5:].reset_index(drop=True).set_index('assigned_var')}
        }
        G.add_node(node, **attributes)

    for edge in edges.index:
        u,v = edges.loc[edge,['u','v']]
        attributes = edges.drop(columns=['u','v'], inplace=False).loc[edge].to_dict()
        G.add_edge(u, v, **attributes)
    
    cells = afm.obs_names[afm.obs_names.isin(G.nodes)]
    char_X = pd.DataFrame(afm.X, columns=afm.var_names, index=afm.obs_names).loc[cells]
    tree = CassiopeiaTree(character_matrix=char_X, tree=G, cell_meta=afm.obs.loc[cells])

    return tree


##