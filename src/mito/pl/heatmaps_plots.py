"""
Utils and plotting functions to visualize (clustered and annotated) cells x vars AFM matrices
or cells x cells distances/affinity matrices.
"""

import logging
import matplotlib
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import plotting_utils as plu
from typing import Dict, Any
from anndata import AnnData
from cassiopeia.data import CassiopeiaTree
from ..tl.phylo import build_tree, _get_leaves_order
from ..tl.annotate import _get_muts_order


##


def heatmap_distances(
    afm: AnnData, 
    distance_key: str = 'distances',
    tree: CassiopeiaTree = None, 
    vmin: float = .25, vmax: float = .95, 
    cmap: str = 'Spectral', 
    ax: matplotlib.axes.Axes = None
    ) -> matplotlib.axes.Axes:
    """
    Heatmap cell/cell pairwise distances.

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix.
    distance_key : str,
        Distence key in afm.obsp. Default is distances
    tree : CassiopeiaTree, optional
        Tree from which cell ordering can be retrieved. Default is None.
    vmin : float, optional
        Minimum value for the colorbar. Default is 0.25.
    vmax : float, optional
        Maximum value for the colorbar. Default is 0.95.
    cmap : str, optional
        Color map for cell-cell distances. Default is "Spectral".
    ax : matplotlib.axes.Axes, optional
        Axes object to draw on. Default is False.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object.
    """

    if distance_key not in afm.obsp:
        raise ValueError('Compute distances first!')

    if tree is None:
        logging.info('Compute tree from precomputed cell-cell distances...')
        tree = build_tree(afm, precomputed=True)

    order = _get_leaves_order(tree)
    D = afm[order].obsp[distance_key].toarray()
    ax.imshow(D, cmap=cmap)
    plu.format_ax(
        ax=ax, xlabel='Cells', ylabel='Cells', xticks=[], yticks=[],
    )
    plu.add_cbar(
        D.flatten(), ax=ax, palette=cmap, 
        label='Distance', layout='outside',
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
    vmax: float = .1,
    kwargs: Dict[str, Any] = {} 
    ) -> matplotlib.axes.Axes:
    """
    Heatmap cell x variants.

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix.
    tree : CassiopeiaTree, optional
        Tree from which cell ordering can be retrieved. Default is None.
    label : str, optional
        Label for layer colorbar. Default is "Allelic Frequency".
    annot : str, optional
        afm.obs column to annotate. Default is None.
    annot_cmap : dict, optional
        Color mapping for afm.obs[annot]. Default is None.
    layer : str, optional
        Layer to plot. Default is None.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw on. Default is False.
    cmap : str, optional
        Color map for layer. Default is "mako".
    vmin : float, optional
        Minimum value for the colorbar. Default is 0.25.
    vmax : float, optional
        Maximum value for the colorbar. Default is 0.95.
    kwargs: dict, optional
        Optional kwargs to plu.plot_heatmap. Default is {}.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object.
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
        X = afm.X.toarray()
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
        palette = plu.create_palette(afm.obs, annot, annot_cmap_)
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
        cb.set_label(annot, rotation=90, labelpad=0)
        cb.ax.set(xticks=[], yticks=[])

    else:
        raise KeyError(f'{annot} not in afm.obs. Check annotation...')
    
    # Plot heatmap
    plu.plot_heatmap(
        df_, ax=ax, vmin=vmin, vmax=vmax, 
        ylabel='Cells',
        linewidths=0, y_names=False, label=label, palette=cmap,
        **kwargs
    )

    return ax


##





