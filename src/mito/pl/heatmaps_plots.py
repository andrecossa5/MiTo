"""
Utils and plotting functions to visualize (clustered and annotated) cells x vars AFM matrices
or cells x cells distances/affinity matrices.
"""

import logging
import matplotlib
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from typing import Dict, Any
from anndata import AnnData
from cassiopeia.data import CassiopeiaTree
from .plotting_base import format_ax, add_cbar, plot_heatmap
from .colors import create_palette
from ..tl.phylo import build_tree
from ..tl.annotate import MiToTreeAnnotator


##


def _get_leaves_order(tree):
    order = []
    for node in tree.depth_first_traverse_nodes():
        if node in tree.leaves:
            order.append(node)
    return order


##


def _get_muts_order(tree):

    tree_ = tree.copy()
    model = MiToTreeAnnotator(tree_)
    model.get_T()
    model.get_M()
    model.extract_mut_order()

    return model.ordered_muts


##


def heatmap_distances(
    afm: AnnData, 
    tree: CassiopeiaTree = None, 
    vmin: float = .25, vmax: float = .95, 
    cmap: str = 'Spectral', 
    ax: matplotlib.axes.Axes = None
    ) -> matplotlib.axes.Axes:
    """
    Heatmap cell/cell pairwise distances.

    Args:
        afm (AnnData): Allele Frequency Matrix
        tree (CassiopeiaTree, optional. Default: None): Tree from which cell ordering can be retrieved. 
        vmin (float, optional. Default: .25): Min value for colobar.
        vmax (float, optional. Default: .95): Max value for colobar.
        cmap (str, optional. Default: 'Spectral'): cmap for cell-cell distances.
        ax (matplotlib.axes.Axes, optional. Default: False): ax object to draw on.

    Returns:
        ax (matplotlib.axes.Axes): ax object.
    """

    if 'distances' not in afm.obsp:
        raise ValueError('Compute distances first!')

    if tree is None:
        logging.info('Compute tree from precomputed cell-cell distances...')
        tree = build_tree(afm, precomputed=True)

    order = _get_leaves_order(tree)
    ax.imshow(afm[order].obsp['distances'].A, cmap=cmap)
    format_ax(
        ax=ax, xlabel='Cells', ylabel='Cells', xticks=[], yticks=[],
        xlabel_size=10, ylabel_size=10
    )
    add_cbar(
        afm.obsp['distances'].A.flatten(), ax=ax, palette=cmap, 
        label='Distance', layout='outside', label_size=10, ticks_size=10,
        vmin=vmin, vmax=vmax
    )

    return ax


##


def heatmap_variants(
    afm: AnnData, 
    tree: CassiopeiaTree = None,  
    label: str = 'Allelic Frequency', 
    annot: str = None, 
    annot_cmap: Dict[str,Any] = None, 
    layer: str = None, 
    ax: matplotlib.axes.Axes = None, 
    cmap: str = 'mako', 
    vmin: float = 0, 
    vmax: float = .1
    ) -> matplotlib.axes.Axes:
    """
    Heatmap cell x variants.

    Args:
        afm (AnnData): Allele Frequency Matrix
        tree (CassiopeiaTree, optional. Default: None): Tree from which cell ordering can be retrieved. 
        label (str, optional. Default: 'Allelic Frequency'): Label for layer colorbar.
        annot (str, optional. Default: None): afm.obs columns to annotate.
        annot_cmap (Dict[str,Any], optional. Default: None): color mapping for afm.obs[annot].
        layer (str, optional. Default: None): layer to plot.
        ax (matplotlib.axes.Axes, optional. Default: False): ax object to draw on.
        cmap (str, optional. Default: 'mako'): cmap for layer.
        vmin (float, optional. Default: .25): Min value for colobar.
        vmax (float, optional. Default: .95): Max value for colobar.

    Returns:
        ax (matplotlib.axes.Axes): ax object.
    """

    # Order cells and columns
    if 'distances' not in afm.obsp:
        raise ValueError('Compute distances first!')

    if tree is None:
        logging.info('Compute tree from precomputed cell-cell distances...')
        tree = build_tree(afm, precomputed=True)

    cell_order = _get_leaves_order(tree)
    mut_order = _get_muts_order(tree)

    if layer is None:
        X = afm.X.A
    elif layer in afm.layers:
        X = afm.layers[layer]
    else:
        raise KeyError(f'Layer {layer} not present in afm.layers')
    
    # Prep ordered df
    df_ = (
        pd.DataFrame(X, index=afm.obs_names, columns=afm.var_names)
        .loc[cell_order, mut_order]
    )

    # Plot annot, if necessary
    if annot is None:
        pass
        
    elif annot in afm.obs.columns:

        annot_cmap_ = sc.pl.palettes.vega_10_scanpy if annot_cmap is None else annot_cmap
        palette = create_palette(afm.obs, annot, annot_cmap_)
        colors = (
            afm.obs.loc[df_.index, annot]
            .astype('str')
            .map(palette)
            .to_list()
        )
        orientation = 'vertical'
        pos = (-.06, 0, 0.05, 1)
        axins = ax.inset_axes(pos) 
        annot_cmap = matplotlib.colors.ListedColormap(colors)
        cb = plt.colorbar(
            matplotlib.cm.ScalarMappable(cmap=annot_cmap), 
            cax=axins, orientation=orientation
        )
        cb.ax.yaxis.set_label_position("left")
        cb.set_label(annot, rotation=90, labelpad=0, fontsize=10)
        cb.ax.set(xticks=[], yticks=[])

    else:
        raise KeyError(f'{annot} not in afm.obs. Check annotation...')
    
    # Plot heatmap
    plot_heatmap(df_, ax=ax, vmin=vmin, vmax=vmax, 
                linewidths=0, y_names=False, label=label, palette=cmap)

    return ax


##





