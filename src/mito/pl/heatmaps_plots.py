"""
Utils and plotting functions to visualize (clustered and annotated) cells x vars AFM matrices
or cells x cells distances/affinity matrices.
"""

import logging
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotting_utils as plu
import scanpy as sc
from anndata import AnnData
from cassiopeia.data import CassiopeiaTree

from mito.tl.phylo import _get_leaves_order, _get_muts_order, build_tree

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
        raise ValueError(
            f'No distances in afm.obsp["{distance_key}"]: run mito.pp.compute_distances '
            f'(or mito.pp.filter_afm) first.'
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    if tree is None:
        logging.info('Compute tree from precomputed cell-cell distances...')
        tree = build_tree(afm)

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
    label: str = None,
    annot: str = None,
    annot_cmap: dict[str,Any] = None,
    layer: str = None,
    ax: matplotlib.axes.Axes = None,
    cmap: str = 'afmhot_r',
    vmin: float = 0,
    vmax: float = None,
    kwargs: dict[str, Any] = None
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
        Label for layer colorbar. Default is None: "Allelic Frequency" for the allele frequencies
        in afm.X, "Genotype" for the "bin" layer, and the name of any other layer.
    annot : str, optional
        afm.obs column to annotate. Default is None.
    annot_cmap : dict, optional
        Color mapping for afm.obs[annot]. Default is None.
    layer : str, optional
        Layer to plot. Default is None: the allele frequencies in afm.X.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw on. Default is False.
    cmap : str, optional
        Color map for layer. Default is "afmhot_r".
    vmin : float, optional
        Minimum value for the colorbar. Default is 0.
    vmax : float, optional
        Maximum value for the colorbar. Default is None: 1 for the genotypes, 0.1 otherwise.
    kwargs: dict, optional
        Optional kwargs to plu.plot_heatmap. Default is {}. The variant labels are sized to fit
        the width of the axes, and left out when they would be smaller than 2.5 points, unless
        `x_names` or `x_names_size` are given.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object.
    """

    # Order cells and columns
    if kwargs is None:
        kwargs = {}
    if 'distances' not in afm.obsp:
        raise ValueError(
            'No distances in afm.obsp["distances"]: run mito.pp.compute_distances '
            '(or mito.pp.filter_afm) first.'
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    if tree is None:
        logging.info('Compute tree from precomputed cell-cell distances...')
        tree = build_tree(afm)

    cell_order = _get_leaves_order(tree)
    mut_order = _get_muts_order(tree)

    if layer is None:
        X = afm.X.toarray()
    elif layer in afm.layers:
        X = afm.layers[layer]
        X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)   # AD/bin sparse, DP dense
    else:
        raise KeyError(f'Layer {layer} not present in afm.layers')

    if label is None:
        label = 'Allelic Frequency' if layer is None else ('Genotype' if layer == 'bin' else layer)
    if vmax is None:
        vmax = 1. if layer == 'bin' else .1

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

    # Variant labels: as large as the columns allow, none if unreadable
    if 'x_names' not in kwargs and 'x_names_size' not in kwargs:
        width = ax.get_position().width * ax.figure.get_figwidth() * 72        # points
        size = min(7., .9 * width / df_.shape[1])
        kwargs = {**kwargs, 'x_names': bool(size >= 2.5), 'x_names_size': float(size)}

    # Plot heatmap
    plu.plot_heatmap(
        df_, ax=ax, vmin=vmin, vmax=vmax,
        ylabel='Cells',
        linewidths=0, y_names=False, label=label, palette=cmap,
        **kwargs
    )

    return ax


##





