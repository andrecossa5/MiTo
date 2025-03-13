"""
Utils and plotting functions to visualize (clustered and annotated) cells x vars AFM matrices
or cells x cells distances/affinity matrices.
"""

from .plotting_base import *
from .colors import *


##


def heatmap_distances(afm, tree, vmin=.25, vmax=.95, cmap='Spectral', ax=None):
    """
    Heatmap cell/cell distances, ordered as in tree.
    """

    order = []
    for node in tree.depth_first_traverse_nodes():
        if node in tree.leaves:
            order.append(node)

    ax.imshow(afm[order].obsp['distances'].A, cmap='Spectral')
    format_ax(ax=ax, xlabel='Cells', ylabel='Cells', xticks=[], yticks=[])
    add_cbar(
        afm.obsp['distances'].A.flatten(), ax=ax, palette='Spectral', 
        label='Distance', layout='outside', label_size=8, ticks_size=8,
        vmin=vmin, vmax=vmax
    )

    return ax














