"""
Custom plotting function for embeddings.
"""

import scanpy as sc
import matplotlib
from typing import Iterable, Dict, Any, Tuple
from anndata import AnnData
import plotting_utils as plu


##


def draw_embedding(
    afm: AnnData, 
    basis: str = 'X_umap', 
    feature: str = None,
    ax: matplotlib.axes.Axes = None,
    categorical_cmap: str|Dict[str,Any] = sc.pl.palettes.vega_20_scanpy,
    continuous_cmap: str = 'viridis',
    size: float = None,
    frameon: bool = False,
    outline: bool = False,
    legend: bool = False,
    loc: str = 'center left',
    bbox_to_anchor: Tuple[float, float] = (1,.5),
    artists_size: float = 10,
    label_size: float = 10,
    ticks_size: float = 10,
    kwargs: Dict[str,Any] = {}
    ) -> matplotlib.axes.Axes:
    """
    sc.pl.embedding, with some defaults and a custom legend.

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix with some basis to plot in afm.obsm.
    basis : str, optional
        Key in afm.obsm. Default is "X_umap".
    feature : Iterable[str], optional
        Features to plot. Default is an empty list.
    ax : matplotlib.axes.Axes, optional
        Axes object to populate. Default is None.
    categorical_cmap : str or dict, optional
        Color palette for categoricals. Default is sc.pl.palettes.vega_20_scanpy.
    continuous_cmap : str, optional
        Color palette for continuous data. Default is "viridis".
    size : float, optional
        Point size. Default is None.
    frameon : bool, optional
        Whether to draw a frame around the axes. Default is False.
    outline : bool, optional
        Whether to draw a fancy outline around dots. Default is False.
    legend : bool, optional
        Whether to automatically draw a legend. Default is False.
    loc : str, optional
        Which corner of the legend to anchor. Default is "center left".
    bbox_to_anchor : tuple of float, optional
        Anchor 'loc' legend corner to ax.transformed coordinates. Default is (1, 0.5).
    artists_size : float, optional
        Size of legend artists. Default is 10.
    label_size : float, optional
        Size of legend labels. Default is 10.
    ticks_size : float, optional
        Size of legend ticks. Default is 10.
    kwargs: dict, optional
        Kwargs to sc.pl.embedding. Default is {}

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object.
    """

    if isinstance(categorical_cmap, str) and feature in afm.obs.columns:
        _cmap = plu.create_palette(afm.obs, feature, _cmap)
    elif isinstance(categorical_cmap, dict) and feature in afm.obs.columns:
        assert all(x in categorical_cmap for x in afm.obs[feature].unique())
        _cmap = categorical_cmap
    else:
        _cmap = None

    ax = sc.pl.embedding(
        afm, 
        basis=basis, 
        ax=ax, 
        color=feature, 
        palette=_cmap,
        color_map=continuous_cmap, 
        legend_loc=None,
        size=size, 
        frameon=frameon, 
        add_outline=outline,
        show=False,
        **kwargs
    )

    if legend:
        plu.add_legend(
            ax=ax, 
            label=feature, 
            colors=categorical_cmap,
            loc=loc, 
            bbox_to_anchor=bbox_to_anchor,
            artists_size=artists_size, 
            label_size=label_size, 
            ticks_size=ticks_size
        )

    ax.set(title=None)

    return ax


##