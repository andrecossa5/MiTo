"""
Custom plotting function for embeddings.
"""

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import plotting_utils as plu
import scanpy as sc
from anndata import AnnData

##


def _is_continuous(x):
    """A numeric, non-categorical column is drawn with a continuous colormap."""
    import pandas as pd
    return not isinstance(x.dtype, pd.CategoricalDtype) and pd.api.types.is_numeric_dtype(x)


##


def draw_embedding(
    afm: AnnData,
    basis: str = 'X_umap',
    feature: str = None,
    ax: matplotlib.axes.Axes = None,
    categorical_cmap: str|dict[str,Any] = sc.pl.palettes.vega_20_scanpy,
    continuous_cmap: str = 'viridis',
    size: float = None,
    frameon: bool = False,
    outline: bool = False,
    legend: bool = False,
    loc: str = 'center left',
    bbox_to_anchor: tuple[float, float] = (1,.5),
    artists_size: float = 10,
    label_size: float = 10,
    ticks_size: float = 10,
    kwargs: dict[str,Any] = None
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

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    if kwargs is None:
        kwargs = {}

    # Colors of a categorical feature, as a {category: color} mapping. A list of colors
    # (the default palette) or the name of a seaborn palette are both turned into one:
    # the legend needs the mapping, not the palette it came from.
    _cmap = None
    if feature is not None and feature in afm.obs.columns and not _is_continuous(afm.obs[feature]):
        if isinstance(categorical_cmap, dict):
            missing = set(afm.obs[feature].astype(str).unique()) - set(categorical_cmap)
            if missing:
                raise ValueError(f'No color for {sorted(missing)} in categorical_cmap.')
            _cmap = categorical_cmap
        elif isinstance(categorical_cmap, str):
            _cmap = plu.create_palette(afm.obs, feature, palette=categorical_cmap)
        else:
            _cmap = plu.create_palette(afm.obs, feature, col_list=list(categorical_cmap))

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

    if legend and _cmap is not None:
        plu.add_legend(
            ax=ax,
            label=feature,
            colors=_cmap,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            artists_size=artists_size,
            label_size=label_size,
            ticks_size=ticks_size
        )

    ax.set(title=None)

    return ax


##
