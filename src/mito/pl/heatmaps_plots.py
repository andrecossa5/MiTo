"""
Utils and plotting functions to visualize (clustered and annotated) cells x vars AFM matrices
or cells x cells distances/affinity matrices.
"""

from .plotting_base import *
from .colors import *
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


def heatmap_distances(afm, tree=None, vmin=.25, vmax=.95, cmap='Spectral', ax=None):
    """
    Heatmap cell/cell distances.
    """

    if 'distances' not in afm.obsp:
        raise ValueError('Compute distances first!')

    if tree is None:
        logging.info('Compute tree from precomputed cell-cell distances...')
        tree = build_tree(afm, precomputed=True)

    order = _get_leaves_order(tree)
    ax.imshow(afm[order].obsp['distances'].A, cmap='Spectral')
    format_ax(
        ax=ax, xlabel='Cells', ylabel='Cells', xticks=[], yticks=[],
        xlabel_size=10, ylabel_size=10
    )
    add_cbar(
        afm.obsp['distances'].A.flatten(), ax=ax, palette='Spectral', 
        label='Distance', layout='outside', label_size=10, ticks_size=10,
        vmin=vmin, vmax=vmax
    )

    return ax


##


def heatmap_variants(afm, tree=None, label='Allelic Frequency', annot=None, 
                     annot_cmap=None, layer=None, ax=None, cmap='mako', vmin=0, vmax=.1):
    """
    Heatmap cell x variants.
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





