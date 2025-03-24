"""
Custom plotting function for embeddings.
"""

import scanpy as sc
import matplotlib
from typing import Iterable, Dict, Any, Tuple
from anndata import AnnData
from .colors import create_palette
from .plotting_base import add_legend


##


def draw_embedding(
    afm: AnnData, 
    basis: str = 'X_umap', 
    feature: Iterable[str] = [],
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
    ticks_size: float = 10
    ) -> matplotlib.axes.Axes:
    """
    sc.pl.embedding, with some defaults and a custom legend.

    Args:
        afm (AnnData): Allele Frequency Matrix with some basis to plot in afm.obsm.
        basis (str, optional. Default: X_umap): key in afm.obsm.
        feature (Iterable[str], optional. Default: []): features to plot.
        ax (matplotlib.axes.Axes, optional. Default: None): ax object to populate.
        categorical_cmap (str|Dict[str,Any], optional. Default: sc.pl.palettes.vega_20_scanpy): color palette for categoricals.
        continuous_cmap (str, optional. Defaults>: 'viridis'): color palette for continuous data.
        size (float, optional. Default: None): point size.
        frameon (bool, optional. Default: False): draw frame araund ax, or not.
        outline (bool, optional. Default: False): Fancy outline around dots.
        legend (bool, optional. Default: False): draw authomatically a legend.
        loc (str, optional. Default: 'center left'): which corner of the legend to anchor.
        bbox_to_anchor (Tuple[float, float], optional. Default: (1,.5)): anchor 'loc' legend corner to ax.transformed coordinates.
        artists_size (float, optional. Default: 10): size of legend artists.
        label_size (float, optional. Default: 10): size of legend label.
        ticks_size (float, optional. Default: 10): size of legend ticks.

    Returns:
        ax (matplotlib.axes.Axes): ax object.
    """

    if not isinstance(categorical_cmap, dict):
        categorical_cmap = create_palette(afm.obs, feature, categorical_cmap)
    else:
        pass

    ax = sc.pl.embedding(
        afm, 
        basis=basis, 
        ax=ax, 
        color=feature, 
        palette=categorical_cmap,
        color_map=continuous_cmap, 
        legend_loc=None,
        size=size, 
        frameon=frameon, 
        add_outline=outline,
        show=False
    )

    if legend:
        add_legend(
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