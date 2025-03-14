"""
Stores plotting functions for embeddings.
"""

import scanpy as sc
import matplotlib.pyplot as plt
from .colors import create_palette
from .plotting_base import add_legend


##


def draw_embedding(
    afm, 
    basis='X_umap', 
    feature=[],
    ax=None,
    categorical_cmap=sc.pl.palettes.vega_20_scanpy,
    continuous_cmap='viridis',
    size=None,
    frameon=False,
    outline=False,
    legend=False,
    loc='center left',
    bbox_to_anchor=(1,.5),
    artists_size=10,
    label_size=10,
    ticks_size=10
    ):
    """
    sc.pl.embedding, with some defaults and a custom legend.
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