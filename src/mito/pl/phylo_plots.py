"""
Tree plotting utils.
"""

import logging
import pandas as pd
import scanpy as sc
import plotting_utils as plu
from typing import Iterable, Dict, Any
from cassiopeia.data import CassiopeiaTree
from cassiopeia.plotting.local import utilities as ut
from cassiopeia.plotting.local import *
from matplotlib.patches import Polygon
from .other_plots import *


##


_categorical_cmaps = [sc.pl.palettes.vega_20_scanpy, sc.pl.palettes.default_20, plu.ten_godisnot, 'set1', 'dark']
_continuous_cmaps = ['viridis', 'inferno', 'magma']
_cont_character_cmap = 'mako'
_bin_character_cmap = { 1 : 'r', 0 : 'b', -1 : 'lightgrey', np.nan : 'lightgrey' }
 

##


def _to_polar_coords(d):

    new_d = {}
    for k in d:
        x, y = d[k]
        if not isinstance(x, list):
            x = [x]
            y = [y]
            x, y = ut.polars_to_cartesians(x, y)
            new_d[k] = x[0], y[0]
        else:
            x, y = ut.polars_to_cartesians(x, y)
            new_d[k] = x, y
    
    return new_d


##


def _to_polar_colorstrips(L):

    new_L = []
    for d in L:
        new_d = {}
        for k in d:
            x, y, a, b = d[k]
            x, y = ut.polars_to_cartesians(x, y)
            new_d[k] = x, y, a, b
        new_L.append(new_d)
    
    return new_L


##


def _place_tree_and_annotations(
    tree, 
    features=None, 
    characters=None,
    orient=90, 
    extend_branches=True, 
    angled_branches=True, 
    add_root=True, 
    continuous_cmaps=None, 
    cont_character_cmap=None,
    categorical_cmaps=None,
    bin_character_cmap=None,
    layer='raw', 
    colorstrip_width=None, 
    colorstrip_spacing=None,
    vmin=None,
    vmax=None,
    vmin_characters=None,
    vmax_characters=None
    ):
    """
    Util to set tree elements.
    """

    is_polar = isinstance(orient, (float, int))
    loc = "polar" if is_polar else orient
    
    # Node and branch coords
    node_coords, branch_coords = ut.place_tree(
        tree,
        orient=orient,
        extend_branches=extend_branches,
        angled_branches=angled_branches,
        add_root=add_root
    )

    # Colorstrips
    anchor_coords = { k:node_coords[k] for k in node_coords if tree.is_leaf(k) }
    tight_width, tight_height = compute_colorstrip_size(node_coords, anchor_coords, loc)
    width = colorstrip_width or tight_width
    spacing = colorstrip_spacing or tight_width / 2
    colorstrips = []
    features = features if features is not None else []
    characters = characters if characters is not None else []
    covariates = features + characters
    n_cat = 0

    # Here we go
    for cov in covariates:

        # Feature
        if cov in features:
            if cov in tree.cell_meta.columns:
                x = tree.cell_meta[cov].copy()
            else:
                raise KeyError(f'{cov} not in tree.cell_meta!')
        
        # Character
        is_bin_layer = all(x in [1,0,-1] for x in tree.layers[layer].iloc[:,0].unique())
        if cov in characters:
            if cov in tree.layers[layer].columns:
                if is_bin_layer:
                    x = tree.layers[layer][cov].copy()
                    x = x.astype('category')
                else:
                    x = tree.layers[layer][cov].copy()
            else:
                raise KeyError(f'{cov} not in tree.layers[{layer}].')

        # Colorstrip specification
        if pd.api.types.is_numeric_dtype(x):

            if cov in features:
                vmin_annot = np.percentile(x, 10) if vmin is None else vmin 
                vmax_annot = np.percentile(x, 90) if vmax is None else vmax 
                if continuous_cmaps is None:
                    continuous_cmap = _continuous_cmaps[0]
                elif cov in continuous_cmaps:
                    continuous_cmap = continuous_cmaps[cov]
                else:
                    raise KeyError(f'{cov} not in continuous_cmaps.')
            
            elif cov in characters:
                vmin_annot = vmin_characters
                vmax_annot = vmax_characters
                continuous_cmap = cont_character_cmap if cont_character_cmap is not None else _cont_character_cmap

            colorstrip, anchor_coords = create_continuous_colorstrip(
                x.to_dict(), 
                anchor_coords,
                width, 
                tight_height,
                spacing, 
                loc, 
                continuous_cmap,
                vmin_annot, 
                vmax_annot
            )

        elif pd.api.types.is_string_dtype(x) or pd.api.types.is_categorical_dtype(x):

            x = x.astype('category')
            if cov in features:
                if categorical_cmaps is None or cov not in categorical_cmaps:
                    categorical_cmap = plu.create_palette(tree.cell_meta, cov, _categorical_cmaps[n_cat], add_na=True)
                elif cov in categorical_cmaps:
                    _cmap = categorical_cmaps[cov]
                    if isinstance(_cmap, str) or isinstance(_cmap, list):
                        categorical_cmap = plu.create_palette(tree.cell_meta, cov, _cmap, add_na=True)
                    elif isinstance(_cmap, dict):
                        categorical_cmap = _cmap
                        categorical_cmap[np.nan] = 'lightgrey'
                    else:
                        raise ValueError(f'''Adjust categorical_cmaps. {cov}: 
                                         categorical_cmaps is nor a str, a list or a dict...''')
            
            elif cov in characters:
                categorical_cmap = bin_character_cmap if bin_character_cmap is not None else _bin_character_cmap
        
            if not all([ cat in categorical_cmap.keys() for cat in x.unique() ]):

                cats = x.unique()
                missing_cats = cats[[ cat not in categorical_cmap.keys() for cat in cats ]]
                logging.info(f'Missing cats in cmap for meta feat {cov}: {missing_cats}. Adding new colors...')

                for i,missing in enumerate(missing_cats):
                    categorical_cmap[missing] = sc.pl.palettes.godsnot_102[i]

            assert (all([ cat in categorical_cmap.keys() for cat in x.unique() ]))
            assert categorical_cmap[np.nan] == 'lightgrey'

            # Place
            boxes, anchor_coords = ut.place_colorstrip(
                anchor_coords, width, tight_height, spacing, loc
            )
            
            colorstrip = {}
            for leaf,value in zip(x.index, x.values):
                colorstrip[leaf] = boxes[leaf] + (categorical_cmap[value], f"{leaf}\n{value}")

            n_cat += 1

        else:
            raise ValueError(f'{cov} has {x.dtype} dtype. Check meta and layers...')
        
        colorstrips.append(colorstrip)

    # To polar, if necessary
    if is_polar:
        branch_coords = _to_polar_coords(branch_coords)
        node_coords = _to_polar_coords(node_coords)    
        colorstrips = _to_polar_colorstrips(colorstrips) 
    
    # Add feature names as colorstrips labels
    colorstrips = [ (c,name) for c,name in zip(colorstrips, covariates) ]
    
    return node_coords, branch_coords, colorstrips


##


def _set_colors(d, meta=None, cov=None, cmap=None, kwargs=None, vmin=None, vmax=None):
    """
    Create a dictionary of elements colors.
    """

    if meta is not None and cov is not None:
        if cov in meta.columns:
            x = meta[cov]
            if isinstance(cmap, str):
                if pd.api.types.is_numeric_dtype(x):
                    cmap = matplotlib.colormaps[cmap]
                    cmap = matplotlib.cm.get_cmap(cmap)
                    if vmin is None or vmax is None:
                        vmin = np.percentile(x.values, 10)
                        vmax = np.percentile(x.values, 90)
                    normalize = plt.Normalize(vmin=vmin, vmax=vmax)
                    colors = [ cmap(normalize(value)) for value in x ]
                    colors = { k:v for k, v in zip(x.index, colors)}
                elif pd.api.types.is_string_dtype(x):
                    colors = (
                        meta[cov]
                        .map(plu.create_palette(meta, cov, cmap))
                        .to_dict()
                    )
            elif isinstance(cmap, dict):
                print('User-provided colors dictionary...')
                colors = meta[cov].astype('str').map(cmap).to_dict()
            else:
                raise KeyError(f'{cov} You can either specify a string cmap or an element:color dictionary.')
        else:
            raise KeyError(f'{cov} not present in cell_meta.')
    else:
        colors = { k : kwargs['c'] for k in d }

    return colors


##


def plot_tree(
    tree: CassiopeiaTree, 
    ax: matplotlib.axes.Axes = None, 
    orient: float|str = 90, 
    extend_branches: bool = True, 
    angled_branches: bool = True, 
    add_root: bool = False, 
    features: Iterable[str] = None, 
    categorical_cmaps: Dict[str, str|Dict[str,Any]] = None, 
    continuous_cmaps: Dict[str, str|Dict[str,Any]] = None, 
    characters: Iterable[str] = None,  
    cont_character_cmap: str = 'mako', 
    bin_character_cmap: Dict[str,Any] = None, 
    layer: str ='raw', 
    vmin_characters: float = 0, 
    vmax_characters: float =.05,
    colorstrip_spacing: float =.25, 
    colorstrip_width: float = 1.5, 
    labels: bool = True, 
    label_size: float = 10, 
    label_offset: float = 2,
    meta_branches: pd.DataFrame = None, 
    cov_branches: str = None, 
    cmap_branches: str|Dict[str,Any] = 'Spectral_r',
    cov_leaves: str = None, 
    cmap_leaves: str|Dict[str,Any] = 'tab20', 
    feature_internal_nodes: str = None, 
    cmap_internal_nodes: str|Dict[str,Any] ='Spectral_r', 
    vmin: float = None, 
    vmax: float = None,
    vmin_internal_nodes: float = .2, 
    vmax_internal_nodes: float = .8,
    vmin_leaves: float = None, 
    vmax_leaves: float = None,
    internal_node_labels: bool = False, 
    internal_node_subset: Iterable[str] = None, 
    internal_node_label_size: float = 7, 
    show_internal: bool = False, 
    leaves_labels: bool = False, 
    leaf_label_size: float = 5, 
    colorstrip_kwargs: Dict[str,Any] = {}, 
    leaf_kwargs: Dict[str,Any] = {}, 
    internal_node_kwargs: Dict[str,Any] = {}, 
    branch_kwargs: Dict[str,Any] = {}, 
    x_space: float = 1.5
    ) -> matplotlib.axes.Axes:
    """
    Plotting function that extends capabilities in cs.plotting.local.plot_matplotlib from
    Cassiopeia, MW Jones et al, 2020.

    Parameters
    ----------
    tree : CassiopeiaTree
        Tree to plot.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw on. Default is None.
    orient : float or str, optional
        Tree layout in polar (90) or cartesian coordinates (e.g., "down"). Default is 90.
    extend_branches : bool, optional
        Equal length branch from leaf to root. Default is True.
    angled_branches : bool, optional
        Make branches angled, not round. Default is True.
    add_root : bool, optional
        Add root to tree. Default is False.
    features : Iterable[str], optional
        Features in tree.cell_meta to plot. Default is None.
    categorical_cmaps : dict of {str: str or dict}, optional
        Dictionary of colors for categorical features. Default is None.
    continuous_cmaps : dict of {str: str or dict}, optional
        Dictionary of colors for continuous features. Default is None.
    characters : Iterable[str], optional
        List of characters to plot. Default is None.
    cont_character_cmap : str, optional
        Color map for characters ("raw" layer). Default is "mako".
    bin_character_cmap : dict, optional
        Colors for binary character states ("transformed" layer). Default is None.
    layer : str, optional
        Layer in tree.layers to plot, if characters is not None. Default is "raw".
    vmin_characters : float, optional
        Minimum value for character colorbar. Default is 0.
    vmax_characters : float, optional
        Maximum value for character colorbar. Default is 0.05.
    colorstrip_spacing : float, optional
        Relative amount of spacing between colorstrips. Default is 0.25.
    colorstrip_width : float, optional
        Relative colorstrip width. Default is 1.5.
    labels : bool, optional
        Draw labels for features and characters. Default is True.
    label_size : float, optional
        Features and character label size. Default is 10.
    label_offset : float, optional
        Features and character label offset. Default is 2.
    meta_branches : pd.DataFrame, optional
        Annotation table for branches. Default is None.
    cov_branches : str, optional
        Branch feature to plot. Default is None.
    cmap_branches : str or dict, optional
        Color map for branch feature. Default is "Spectral_r".
    cov_leaves : str, optional
        Leaf feature to plot. Default is None.
    cmap_leaves : str or dict, optional
        Color map for leaves feature. Default is "tab20".
    vmin_leaves : float, optional
        Min value for leaves cmap.
    vmax_leaves : float, optional
        Max value for leaves cmap.
    feature_internal_nodes : str, optional
        Internal node feature to plot. Default is None.
    cmap_internal_nodes : str or dict, optional
        Color map for internal nodes feature. Default is "Spectral_r".
    vmin_internal_nodes : float, optional
        Minimum value for internal node feature colorbar. Default is 0.2.
    vmax_internal_nodes : float, optional
        Maximum value for internal node feature colorbar. Default is 0.8.
    internal_node_labels : bool, optional
        Draw internal node names on location. Default is False.
    internal_node_subset : Iterable[str], optional
        Subset of internal nodes to plot. Default is None.
    internal_node_label_size : float, optional
        Internal node name/label size. Default is 7.
    show_internal : bool, optional
        Show internal nodes. Default is False.
    leaves_labels : bool, optional
        Plot leaves names. Default is False.
    leaf_label_size : float, optional
        Leaf name/label size. Default is 5.
    colorstrip_kwargs : dict, optional
        Additional colorstrip keyword arguments. Default is {}.
    leaf_kwargs : dict, optional
        Additional leaves keyword arguments. Default is {}.
    internal_node_kwargs : dict, optional
        Additional internal nodes keyword arguments. Default is {}.
    branch_kwargs : dict, optional
        Additional branch keyword arguments. Default is {}.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object.
    """
    
    # Set coord and axis
    ax.axis('off')

    # Set graphic elements
    (
        node_coords,
        branch_coords,
        colorstrips,
    ) = _place_tree_and_annotations(
        tree, 
        features=features, 
        characters=characters,
        orient=orient, 
        extend_branches=extend_branches, 
        angled_branches=angled_branches, 
        add_root=add_root, 
        continuous_cmaps=continuous_cmaps, 
        cont_character_cmap=cont_character_cmap,
        categorical_cmaps=categorical_cmaps,
        bin_character_cmap=bin_character_cmap,
        layer=layer, 
        colorstrip_width=colorstrip_width, 
        colorstrip_spacing=colorstrip_spacing,
        vmin=vmin,
        vmax=vmax,
        vmin_characters=vmin_characters,
        vmax_characters=vmax_characters
    )

    ##

    # Branches
    _branch_kwargs = {'linewidth':1, 'c':'k'}
    _branch_kwargs.update(branch_kwargs or {})
    colors = _set_colors(
        branch_coords, meta=meta_branches, cov=cov_branches, 
        cmap=cmap_branches, kwargs=_branch_kwargs
    )
    for branch, (xs, ys) in branch_coords.items():
        c = colors[branch] if branch in colors else _branch_kwargs['c']
        _dict = _branch_kwargs.copy()
        _dict.update({'c':c})
        ax.plot(xs, ys, **_dict)
    
    ##
    
    # Colorstrips
    _colorstrip_kwargs = {'linewidth':0, 'alpha':1}
    _colorstrip_kwargs.update(colorstrip_kwargs or {})
    for colorstrip, feat in colorstrips:
        y_positions = []
        x_positions = []
        for xs, ys, c, _ in colorstrip.values():
            _dict = _colorstrip_kwargs.copy()
            _dict["facecolor"] = c
            polygon = Polygon(xy=list(zip(xs, ys)), closed=True, **_dict)
            polygon.set_rasterized(True) 
            ax.add_patch(polygon)
            y_positions.extend(ys)
            x_positions.extend(xs)
        if orient == 'down' and labels:
            y_min = min(y_positions)
            y_max = max(y_positions)
            y_mid = (y_min + y_max) / 2
            x_min = min(x_positions)
            x_offset = label_offset
            ax.text(
                x_min - x_offset, y_mid, feat, ha='right', va='center', fontsize=label_size
            )
     
    ##
 
    # Leaves 
    leave_size = 2 if cov_leaves is not None else 0
    _leaf_kwargs = {'markersize':leave_size, 'c':'k', 'marker':'o'}
    _leaf_kwargs.update(leaf_kwargs or {})
    leaves = { node : node_coords[node] for node in node_coords if tree.is_leaf(node) }
    colors = _set_colors(
        leaves, meta=tree.cell_meta, cov=cov_leaves, 
        cmap=cmap_leaves, kwargs=_leaf_kwargs, 
        vmin=vmin_leaves, vmax=vmax_leaves
    )     
    for node in leaves:
        _dict = _leaf_kwargs.copy()
        x = leaves[node][0]
        y = leaves[node][1]
        c = colors[node] if node in colors else _leaf_kwargs['c']
        _dict.update({'c':c})
        ax.plot(x, y, **_dict)
        if leaves_labels:
            if orient == 'right':
                ax.text(
                    x+x_space, y, str(node), ha='center', va='center', 
                    fontsize=leaf_label_size
                )
            else:
                raise ValueError(
                    'Correct placement of labels at leaves implemented only for the right orient.'
                    )
 
    ##
 
    # Internal nodes
    _internal_node_kwargs = {
        'markersize': 0 if internal_node_labels else 2, 
        'c':'white', 'marker':'o', 'alpha':1, 
        'markeredgecolor':'k', 'markeredgewidth':1, 'zorder':10
    }
    _internal_node_kwargs.update(internal_node_kwargs or {})
    internal_nodes = { 
        node : node_coords[node] for node in node_coords \
        if tree.is_internal_node(node) and node != 'root'
    }
 
    # Subset nodes if necessary
    if internal_node_subset is not None:
        internal_node_subset = [ x for x in internal_node_subset if x in tree.internal_nodes ]
        internal_nodes = { node : internal_nodes[node] for node in internal_nodes if node in internal_node_subset }
 
    if feature_internal_nodes is not None:
        s = pd.Series({ node : tree.get_attribute(node, feature_internal_nodes) for node in internal_nodes })
        s.loc[lambda x: x.isna()] = 0 # Set missing values to 0
        colors = _set_colors(
            internal_nodes, meta=s.to_frame(feature_internal_nodes), cov=feature_internal_nodes, 
            cmap=cmap_internal_nodes, kwargs=_internal_node_kwargs,
            vmin=vmin_internal_nodes, vmax=vmax_internal_nodes
        )
    # else:
    #     if feature_internal_nodes is None and internal_node_subset is not None:
    #         for node in tree.internal_nodes:
    #             colors = 
    #     else:
    #         raise ValueError('')
        
    for node in internal_nodes:
        _dict = _internal_node_kwargs.copy()
        x = internal_nodes[node][0]
        y = internal_nodes[node][1]
        c = colors[node] if node in colors else _internal_node_kwargs['c']
        s = _internal_node_kwargs['markersize'] if (node in colors or show_internal) else 0
        _dict.update({'c':c, 'markersize':s})
        ax.plot(x, y, **_dict)
 
        if internal_node_labels:
            if node in colors:
                v = tree.get_attribute(node, feature_internal_nodes)
                if isinstance(v, float):
                    v = round(v, 2)
                ax.text(
                    x+.3, y-.1, str(v), ha='center', va='bottom', 
                    bbox=dict(boxstyle='round', alpha=0, pad=10),
                    fontsize=internal_node_label_size,
                )
 
    return ax


##