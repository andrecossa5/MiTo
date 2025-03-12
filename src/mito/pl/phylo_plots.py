"""
Tree plotting utils.
"""

from cassiopeia.plotting.local import utilities as ut
from cassiopeia.plotting.local import *
from .colors import *
from .plotting_base import *


##


_categorical_cmaps = [ten_godisnot, 'tab10', 'set1', 'dark']
_continuous_cmaps = ['mako', 'viridis', 'inferno', 'magma']
_bin_character_cmap = { 1 : 'r', 0 : 'b', -1 : 'lightgrey' }
 

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
    features=[], 
    depth_key=None, 
    orient=90, 
    extend_branches=True, 
    angled_branches=True, 
    add_root=True, 
    continuous_cmaps=None,
    bin_character_cmap=None,
    layer='raw', 
    categorical_cmaps=None, 
    vmin_annot=None, vmax_annot=None, 
    colorstrip_width=None, 
    colorstrip_spacing=None
    ):
    """
    Util to set tree elements.
    """

    is_polar = isinstance(orient, (float, int))
    loc = "polar" if is_polar else orient
    
    # Node and branch coords
    node_coords, branch_coords = ut.place_tree(
        tree,
        depth_key=depth_key,
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
    features = features or []
    n_cat = 0

    for feat in features:

        is_bin_layer = all(x in [1,0,-1] for x in tree.layers[layer].iloc[:,0].unique())
        if feat in tree.cell_meta.columns:
            x = tree.cell_meta[feat]
        elif feat in tree.layers[layer].columns and layer in tree.layers:
            if is_bin_layer:
                x = tree.layers[layer][feat].astype('category')
                if categorical_cmaps is None:
                    categorical_cmaps = {}
                categorical_cmaps[feat] = _bin_character_cmap if bin_character_cmap is None else bin_character_cmap
            else:
                x = tree.layers[layer][feat]
        else:
            raise KeyError(f'{feat} not in tree.cell_meta nor tree.layers[{layer}].')

        if pd.api.types.is_numeric_dtype(x):

            if continuous_cmaps is None:
                continuous_cmap = _continuous_cmaps[0]
                vmin_annot = np.percentile(x, 25)
                vmax_annot = np.percentile(x, 75)
            else:
                if feat in continuous_cmaps:
                    continuous_cmap, vmin_annot, vmax_annot = continuous_cmaps[feat]
                else:
                    raise KeyError(f'{feat} not in continuous_cmaps.')
                
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

        elif pd.api.types.is_string_dtype(x) or x.dtype == 'category':

            if categorical_cmaps is None:
                categorical_cmap = create_palette(tree.cell_meta, feat, _categorical_cmaps[n_cat])
            else:
                if feat in categorical_cmaps:
                    if isinstance(categorical_cmaps[feat], str) or isinstance(categorical_cmaps[feat], list):
                        categorical_cmap = create_palette(tree.cell_meta, feat, categorical_cmaps[feat])
                    elif isinstance(categorical_cmaps[feat], dict):
                        categorical_cmap = categorical_cmaps[feat]
                    else:
                        raise ValueError(f'''Adjust categorical_cmaps. {feat}: 
                                         categorical_cmaps is nor a str, a list or a dict...''')
                else:
                    raise KeyError(f'{feat} not present in meta. Adjust categorical_cmaps and meta params...')
            
            if not all([ cat in categorical_cmap.keys() for cat in x.unique() ]):

                cats = x.unique().copy()
                missing_cats = cats[[ cat not in categorical_cmap.keys() for cat in cats ]]
                print(f'Missing cats in cmap for meta feat {feat}: {missing_cats}. Adding new colors...')

                for i,missing in enumerate(missing_cats):
                    categorical_cmap[missing] = _categorical_cmaps[0][i]

            assert(all([ cat in categorical_cmap.keys() for cat in x.unique() ]))

            x[x.isna()] = 'unassigned'
            categorical_cmap.update({'unassigned':'grey'})

            boxes, anchor_coords = ut.place_colorstrip(
                anchor_coords, width, tight_height, spacing, loc
            )
            
            colorstrip = {}
            for leaf in x.index:
                cat = x.loc[leaf]
                colorstrip[leaf] = boxes[leaf] + (categorical_cmap[cat], f"{leaf}\n{cat}")

            n_cat += 1

        else:
            raise ValueError(f'{feat} in meta has {x.dtype} dtype. Check meta...')
        
        colorstrips.append(colorstrip)

    # To polar, if necessary
    if is_polar:
        branch_coords = _to_polar_coords(branch_coords)
        node_coords = _to_polar_coords(node_coords)    
        colorstrips = _to_polar_colorstrips(colorstrips) 
    
    # Add feature names as colorstrips labels
    colorstrips = [ (c,name) for c,name in zip(colorstrips, features) ]
    
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
                        .map(create_palette(meta, cov, cmap))
                        .to_dict()
                    )
            elif isinstance(cmap, dict):
                print('User-provided colors dictionary...')
                colors = meta[cov].map(cmap).to_dict()
            else:
                raise KeyError(f'{cov} You can either specify a string cmap or an element:color dictionary.')
        else:
            raise KeyError(f'{cov} not present in cell_meta.')
    else:
        colors = { k : kwargs['c'] for k in d }

    return colors


##


def plot_tree(
    tree, ax=None, depth_key=None, orient=90, extend_branches=True, angled_branches=True, add_root=False, 
    features=None, categorical_cmaps=None, continuous_cmaps=None, bin_character_cmap=None, layer='raw', 
    colorstrip_spacing=.25, colorstrip_width=1.5, feature_labels=True, feature_label_size=10, feature_label_offset=2,
    meta_branches=None, cov_branches=None, cmap_branches='Spectral_r',
    cov_leaves=None, cmap_leaves='tab20', 
    feature_internal_nodes=None, cmap_internal_nodes=('Spectral_r', .2, .8),
    internal_node_labels=False, internal_node_subset=None, internal_node_label_size=7, show_internal=False, 
    leaves_labels=False, leaf_label_size=5, 
    colorstrip_kwargs={}, leaf_kwargs={}, internal_node_kwargs={}, branch_kwargs={}, 
    x_space=1.5
    ):
    """
    Plotting function that exends capabilities in cs.plotting.local.plot_matplotlib from
    Cassiopeia, MW Jones.
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
        depth_key=depth_key, 
        orient=orient, 
        extend_branches=extend_branches, 
        angled_branches=angled_branches, 
        add_root=add_root, 
        continuous_cmaps=continuous_cmaps, 
        bin_character_cmap=bin_character_cmap,
        layer=layer,
        categorical_cmaps=categorical_cmaps, 
        colorstrip_width=colorstrip_width, 
        colorstrip_spacing=colorstrip_spacing
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
    _colorstrip_kwargs = {'linewidth':0}
    _colorstrip_kwargs.update(colorstrip_kwargs or {})
    for colorstrip, feat in colorstrips:
        y_positions = []
        x_positions = []
        for xs, ys, c, _ in colorstrip.values():
            _dict = _colorstrip_kwargs.copy()
            _dict["c"] = c
            ax.fill(xs, ys, **_dict)
            y_positions.extend(ys)
            x_positions.extend(xs)
        if orient == 'down' and feature_labels:
            y_min = min(y_positions)
            y_max = max(y_positions)
            y_mid = (y_min + y_max) / 2
            x_min = min(x_positions)
            x_offset = feature_label_offset
            ax.text(
                x_min - x_offset, y_mid, feat, ha='right', va='center', fontsize=feature_label_size
            )
     
    ##
 
    # Leaves 
    leave_size = 2 if cov_leaves is not None else 0
    _leaf_kwargs = {'markersize':leave_size, 'c':'k', 'marker':'o'}
    _leaf_kwargs.update(leaf_kwargs or {})
    leaves = { node : node_coords[node] for node in node_coords if tree.is_leaf(node) }
    colors = _set_colors(
        leaves, meta=tree.cell_meta, cov=cov_leaves, 
        cmap=cmap_leaves, kwargs=_leaf_kwargs
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
        cmap_internal_nodes, vmin_annot, vmax_annot = cmap_internal_nodes
        colors = _set_colors(
            internal_nodes, meta=s.to_frame(feature_internal_nodes), cov=feature_internal_nodes, 
            cmap=cmap_internal_nodes, kwargs=_internal_node_kwargs,
            vmin=vmin_annot, vmax=vmax_annot
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