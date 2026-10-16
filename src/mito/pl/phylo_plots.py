"""
Tree plotting utils.
"""

import logging
from collections.abc import Iterable
from difflib import get_close_matches
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotting_utils as plu
import scanpy as sc
from cassiopeia.data import CassiopeiaTree
from cassiopeia.plotting.local import compute_colorstrip_size, create_continuous_colorstrip
from cassiopeia.plotting.local import utilities as ut
from matplotlib.patches import Polygon

from mito.ut.phylo_utils import get_internal_node_feature

##


_categorical_cmaps = [sc.pl.palettes.vega_20_scanpy, sc.pl.palettes.default_20, plu.ten_godisnot, 'set1', 'dark']
_continuous_cmaps = ['viridis', 'inferno', 'magma']
_cont_character_cmap = 'mako'
_bin_character_cmap = { 1 : 'r', 0 : 'b', -1 : 'lightgrey', np.nan : 'lightgrey' }


##

_NA_COLOR = 'lightgrey'


##


def _is_continuous(x: pd.Series) -> bool:
    """
    Continuous features get a colormap, everything else a palette. A numeric column
    stored as categorical (e.g. binary characters) counts as categorical.
    """

    return not isinstance(x.dtype, pd.CategoricalDtype) and pd.api.types.is_numeric_dtype(x)


##


def _categorical_palette(x, cmap=None, default=None):
    """
    {category: color} for a categorical feature.

    `cmap` is an explicit mapping, the name of a seaborn palette, or a list of colors;
    None falls back to `default` (a name or a list). Categories the mapping does not
    cover are given colors rather than raising, and missing values are always grey - a
    tree with an unlabelled cell should still draw.
    """

    frame = x.astype('category').to_frame('x')
    spec = cmap if cmap is not None else default

    if isinstance(spec, dict):
        palette = dict(spec)
    elif isinstance(spec, str):
        palette = plu.create_palette(frame, 'x', palette=spec, add_na=True)
    elif spec is None:
        palette = plu.create_palette(frame, 'x', add_na=True)
    else:
        palette = plu.create_palette(frame, 'x', col_list=list(spec), add_na=True)

    missing = [ cat for cat in x.dropna().unique() if cat not in palette ]
    if missing:
        logging.info(f'No color for {missing}: assigning defaults.')
        for i, cat in enumerate(missing):
            palette[cat] = sc.pl.palettes.godsnot_102[i % len(sc.pl.palettes.godsnot_102)]
    palette.setdefault(np.nan, _NA_COLOR)

    return palette


##


def _resolve_colors(x, cmap=None, default=None, vmin=None, vmax=None):
    """
    Colors of one tree annotation, whatever it annotates (leaves, branches, internal
    nodes, colorstrips) and however the user specified the colors.

    Returns (colors by index, spec, kind), where `spec` is the {category: color} palette
    for a categorical feature and a (colormap, vmin, vmax) triple for a continuous one -
    which is what the colorstrip and legend code need downstream.
    """

    if _is_continuous(x):
        cmap = cmap if cmap is not None else (default or _continuous_cmaps[0])
        colormap = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
        finite = x.values[np.isfinite(x.values)]
        vmin = (np.percentile(finite, 10) if finite.size else 0) if vmin is None else vmin
        vmax = (np.percentile(finite, 90) if finite.size else 1) if vmax is None else vmax
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        colors = {
            i:(colormap(norm(v)) if np.isfinite(v) else _NA_COLOR) for i, v in x.items()
        }
        return colors, (cmap, vmin, vmax), 'continuous'

    palette = _categorical_palette(x, cmap=cmap, default=default)
    colors = { i:palette.get(v, _NA_COLOR) for i, v in x.items() }

    return colors, palette, 'categorical'


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


def _split_spec(spec, defaults, element):
    """
    Split one element's specification into what it means and how it looks.

    Keys in `defaults` say WHAT to draw (which feature, which colors, whether to label);
    anything else is passed to matplotlib as style (linewidth, marker, zorder, ...). A
    key that is nearly one of the semantic names is a typo, and is refused rather than
    silently forwarded to matplotlib.
    """

    spec = dict(spec or {})
    values = dict(defaults)
    for key in list(spec):
        if key in defaults:
            values[key] = spec.pop(key)
    for key in spec:
        close = get_close_matches(key, defaults, n=1, cutoff=.8)
        if close:
            raise ValueError(f'Unknown key "{key}" for {element}: did you mean "{close[0]}"?')

    return values, spec


##


def _annotation_values(tree, name, layer):
    """
    The values of one annotation, from the cell metadata or from a tree layer, and
    whether it is a character (which has its own default colors).
    """

    if name in tree.cell_meta.columns:
        return tree.cell_meta[name].copy(), False

    if layer in tree.layers and name in tree.layers[layer].columns:
        x = tree.layers[layer][name].copy()
        # 0/1/-1 characters are states, not magnitudes, so they get a palette
        states = pd.unique(x.dropna().values)
        if all(v in [1, 0, -1] for v in states):
            x = x.astype('category')
        return x, True

    raise KeyError(
        f'"{name}" is neither a column of tree.cell_meta nor a character in '
        f'tree.layers["{layer}"].'
    )


##


def _place_tree(tree, annot, cmaps, limits, layer, orient, extend_branches,
                angled_branches, add_root, width=None, spacing=None, leaf_length=0.):
    """
    Coordinates of nodes and branches, plus one colorstrip per annotation.
    """

    is_polar = isinstance(orient, (float, int))
    loc = 'polar' if is_polar else orient

    # cassiopeia places nodes by distance from the root, so a zero-length branch (a cell
    # identical to its node, in a tree with collapsed ties) would tie a child with its
    # parent: give such branches a negligible length while placing, then restore them
    original = {e: tree.get_branch_length(*e) for e in tree.edges}
    eps = 1e-6 * max(original.values())
    for e, length in original.items():
        if tree.is_leaf(e[1]):
            length += leaf_length
        tree.set_branch_length(*e, length if length > 0 else eps)
    try:
        node_coords, branch_coords = ut.place_tree(
            tree, orient=orient, extend_branches=extend_branches,
            angled_branches=angled_branches, add_root=add_root
        )
    finally:
        for e, length in original.items():
            tree.set_branch_length(*e, length)

    anchor_coords = { k:node_coords[k] for k in node_coords if tree.is_leaf(k) }
    tight_width, tight_height = compute_colorstrip_size(node_coords, anchor_coords, loc)
    if not extend_branches and width is None:
        # true lengths: scale the strips to the tree, whose depth is small and unit-free
        depth = max(tree.get_time(leaf) for leaf in tree.leaves)
        width, spacing = .16*depth, (.025*depth if spacing is None else spacing)
    width = width or tight_width
    spacing = spacing if spacing is not None else tight_width/2

    colorstrips = []
    n_cat = 0
    for name in annot:

        x, is_character = _annotation_values(tree, name, layer)
        if _is_continuous(x):
            default = _cont_character_cmap if is_character else _continuous_cmaps[0]
        else:
            default = _bin_character_cmap if is_character else _categorical_cmaps[n_cat % len(_categorical_cmaps)]
        lo, hi = (limits or {}).get(name, (None, None))

        colors, spec, kind = _resolve_colors(
            x, cmap=(cmaps or {}).get(name), default=default, vmin=lo, vmax=hi
        )

        if kind == 'continuous':
            cmap, lo, hi = spec
            colorstrip, anchor_coords = create_continuous_colorstrip(
                x.to_dict(), anchor_coords, width, tight_height, spacing, loc, cmap, lo, hi
            )
        else:
            boxes, anchor_coords = ut.place_colorstrip(
                anchor_coords, width, tight_height, spacing, loc
            )
            colorstrip = {
                leaf:boxes[leaf] + (colors[leaf], f'{leaf}\n{x.loc[leaf]}') for leaf in x.index
            }
            n_cat += 1

        colorstrips.append(colorstrip)

    if is_polar:
        branch_coords = _to_polar_coords(branch_coords)
        node_coords = _to_polar_coords(node_coords)
        colorstrips = _to_polar_colorstrips(colorstrips)

    return node_coords, branch_coords, list(zip(colorstrips, annot, strict=True))


##


def _element_colors(elements, values, cmap, limits):
    """
    Colors of a set of tree elements, from a Series of values over them, or flat.
    """

    if values is None:
        return {}
    lo, hi = limits if limits is not None else (None, None)
    colors, _, _ = _resolve_colors(values, cmap=cmap, vmin=lo, vmax=hi)

    return { el:colors[el] for el in elements if el in colors }


##


def _collapse_clones(tree, name, layer):
    """
    Copy of the tree in which each clade with a single label (other than "unassigned") is a
    polytomy: its cells are attached to the clade's MRCA, at the distance they had from it.
    """

    x, _ = _annotation_values(tree, name, layer)
    label = {}
    for node in tree.depth_first_traverse_nodes(postorder=True):
        if tree.is_leaf(node):
            label[node] = x.loc[node] if pd.notna(x.loc[node]) else None
        else:
            kids = {label[c] for c in tree.children(node)}
            label[node] = kids.pop() if len(kids) == 1 else None

    graph = nx.DiGraph()
    lengths = {}
    stack = [tree.root]
    while stack:
        node = stack.pop()
        for child in tree.children(node):
            if label[child] not in (None, 'unassigned') and not tree.is_leaf(child):
                graph.add_edge(node, child)
                lengths[(node, child)] = tree.get_branch_length(node, child)
                for leaf in tree.leaves_in_subtree(child):
                    graph.add_edge(child, leaf)
                    lengths[(child, leaf)] = tree.get_time(leaf) - tree.get_time(child)
            else:
                graph.add_edge(node, child)
                lengths[(node, child)] = tree.get_branch_length(node, child)
                stack.append(child)

    collapsed = CassiopeiaTree(tree=graph, cell_meta=tree.cell_meta)
    for edge, length in lengths.items():
        collapsed.set_branch_length(*edge, length)
    for key in tree.layers:
        collapsed.layers[key] = tree.layers[key]

    return collapsed


##


def _clade_branch_colors(tree, name, annot, layer, cmaps, default, leaves_only=True):
    """
    {branch: color}: the color of annotation `name` for the branches above cells that all
    share one label (the same color as in its colorstrip), `default` for the others.
    """

    x, _ = _annotation_values(tree, name, layer)
    if _is_continuous(x):
        raise ValueError(f'branches={{"annot": "{name}"}} needs a categorical annotation.')
    names = [annot] if isinstance(annot, str) else list(annot or [])
    n_cat = sum(1 for a in names[:names.index(name)] if not _is_continuous(_annotation_values(tree, a, layer)[0])) \
        if name in names else 0
    colors, _, _ = _resolve_colors(
        x, cmap=(cmaps or {}).get(name), default=_categorical_cmaps[n_cat % len(_categorical_cmaps)]
    )

    label = {}
    for node in tree.depth_first_traverse_nodes(postorder=True):
        if tree.is_leaf(node):
            label[node] = x.loc[node] if pd.notna(x.loc[node]) else None
        else:
            kids = {label[c] for c in tree.children(node)}
            label[node] = kids.pop() if len(kids) == 1 else None

    out = {}
    for parent, child in tree.edges:
        lab = label[child]
        if lab is None or lab == 'unassigned' or (leaves_only and not tree.is_leaf(child)):
            out[(parent, child)] = default
        else:
            out[(parent, child)] = colors[next(iter(tree.leaves_in_subtree(child)))]
    return out


def _draw_branches(ax, branch_coords, spec, style):
    """Tree branches, optionally coloured by a per-branch covariate."""

    values = None
    if spec['feature'] is not None:
        meta = spec['meta']
        if meta is None or spec['feature'] not in meta.columns:
            raise KeyError(
                f'To colour branches by "{spec["feature"]}", pass branches={{"feature": ..., '
                f'"meta": <DataFrame with that column>}}.'
            )
        values = meta[spec['feature']]
    colors = spec.get('colors') or _element_colors(branch_coords, values, spec['cmap'], spec['limits'])

    for branch, (xs, ys) in branch_coords.items():
        ax.plot(xs, ys, **{**style, 'c':colors.get(branch, spec['color'])})


##


def _draw_colorstrips(ax, colorstrips, orient, spec, style):
    """Annotation colorstrips next to the leaves, and their labels."""

    for colorstrip, name in colorstrips:
        xs_all, ys_all = [], []
        for xs, ys, color, _ in colorstrip.values():
            patch = Polygon(xy=list(zip(xs, ys, strict=True)), closed=True,
                            **{**style, 'facecolor':color})
            patch.set_rasterized(True)
            ax.add_patch(patch)
            xs_all.extend(xs)
            ys_all.extend(ys)
        if orient == 'down' and spec['labels']:
            ax.text(
                min(xs_all)-spec['label_offset'], (min(ys_all)+max(ys_all))/2, name,
                ha='right', va='center', fontsize=spec['label_size']
            )


##


def _draw_leaves(ax, tree, node_coords, spec, style, orient):
    """Leaves, optionally coloured by a cell covariate and labelled with their name."""

    leaves = { node:node_coords[node] for node in node_coords if tree.is_leaf(node) }
    values = tree.cell_meta[spec['feature']] if spec['feature'] is not None else None
    if spec['feature'] is not None and spec['feature'] not in tree.cell_meta.columns:
        raise KeyError(f'"{spec["feature"]}" is not a column of tree.cell_meta.')
    colors = _element_colors(leaves, values, spec['cmap'], spec['limits'])
    if spec['labels'] and orient != 'right':
        raise ValueError('Leaf labels are placed correctly only with orient="right".')

    for node, (x, y) in leaves.items():
        ax.plot(x, y, **{**style, 'c':colors.get(node, spec['color'])})
        if spec['labels']:
            ax.text(x+spec['label_offset'], y, str(node), ha='center', va='center',
                    fontsize=spec['label_size'])


##


def _draw_internal_nodes(ax, tree, node_coords, spec, style):
    """Internal nodes, optionally coloured and labelled by one of their attributes."""

    # NB: with add_root=True the layout adds a node that is not part of the tree, so
    # membership is checked before is_internal_node, which raises for unknown nodes.
    nodes = { node:node_coords[node] for node in node_coords
              if node in set(tree.nodes) and tree.is_internal_node(node) }
    if spec['subset'] is not None:
        nodes = { node:xy for node, xy in nodes.items() if node in set(spec['subset']) }

    values = None
    if spec['feature'] is not None:
        attr = pd.Series(
            dict(zip(tree.internal_nodes,
                     get_internal_node_feature(tree, spec['feature']), strict=True))
        )
        if attr.isna().all():
            raise ValueError(
                f'No internal node carries the "{spec["feature"]}" attribute. Set it first: '
                f'mito.tl.compute_fitness (fitness), mito.tl.compute_expansions '
                f'(expansion_pvalue), or tree.set_attribute for your own.'
            )
        values = attr.reindex(list(nodes)).fillna(0)
    colors = _element_colors(nodes, values, spec['cmap'], spec['limits'])

    for node, (x, y) in nodes.items():
        size = style['markersize'] if (node in colors or spec['show']) else 0
        ax.plot(x, y, **{**style, 'c':colors.get(node, spec['color']), 'markersize':size})
        if spec['labels'] and values is not None and node in colors:
            v = values.get(node)
            ax.text(x+.3, y-.1, f'{v:.2f}' if isinstance(v, float) else str(v),
                    ha='center', va='bottom', fontsize=spec['label_size'])


##


_LEGACY_ARGS = {
    'features':'annot', 'characters':'annot',
    'categorical_cmaps':'cmaps', 'continuous_cmaps':'cmaps',
    'cont_character_cmap':'cmaps', 'bin_character_cmap':'cmaps',
    'vmin':'limits', 'vmax':'limits',
    'vmin_characters':'limits', 'vmax_characters':'limits',
    'colorstrip_width':'colorstrips={"width": ...}',
    'colorstrip_spacing':'colorstrips={"spacing": ...}',
    'colorstrip_kwargs':'colorstrips={...}',
    'labels':'colorstrips={"labels": ...}',
    'label_size':'colorstrips={"label_size": ...}',
    'label_offset':'colorstrips={"label_offset": ...}',
    'meta_branches':'branches={"meta": ...}',
    'cov_branches':'branches={"feature": ...}',
    'cmap_branches':'branches={"cmap": ...}',
    'branch_kwargs':'branches={...}',
    'cov_leaves':'leaves={"feature": ...}',
    'cmap_leaves':'leaves={"cmap": ...}',
    'vmin_leaves':'leaves={"limits": (vmin, vmax)}',
    'vmax_leaves':'leaves={"limits": (vmin, vmax)}',
    'leaves_labels':'leaves={"labels": ...}',
    'leaf_label_size':'leaves={"label_size": ...}',
    'leaf_kwargs':'leaves={...}',
    'x_space':'leaves={"label_offset": ...}',
    'feature_internal_nodes':'internal_nodes={"feature": ...}',
    'cmap_internal_nodes':'internal_nodes={"cmap": ...}',
    'vmin_internal_nodes':'internal_nodes={"limits": (vmin, vmax)}',
    'vmax_internal_nodes':'internal_nodes={"limits": (vmin, vmax)}',
    'internal_node_labels':'internal_nodes={"labels": ...}',
    'internal_node_label_size':'internal_nodes={"label_size": ...}',
    'internal_node_subset':'internal_nodes={"subset": ...}',
    'internal_node_kwargs':'internal_nodes={...}',
    'show_internal':'internal_nodes={"show": ...}',
}


def plot_tree(
    tree: CassiopeiaTree,
    annot: str|Iterable[str] = None,
    ax: matplotlib.axes.Axes = None,
    orient: float|str = 90,
    cmaps: dict[str, Any] = None,
    limits: dict[str, tuple[float,float]] = None,
    layer: str = 'raw',
    colorstrips: dict[str, Any] = None,
    branches: dict[str, Any] = None,
    leaves: dict[str, Any] = None,
    internal_nodes: dict[str, Any] = None,
    extend_branches: bool = None,
    leaf_length: float = None,
    collapse_clones: str = None,
    angled_branches: bool = True,
    add_root: bool = False,
    **legacy
    ) -> matplotlib.axes.Axes:
    """
    Plot a cell phylogeny, with annotations. Extends
    `cassiopeia.plotting.local.plot_matplotlib` (MW Jones et al., 2020).

    Annotations are named once, in `annot`, wherever they live: a column of
    tree.cell_meta or a character in tree.layers[`layer`]. Their colors and ranges are
    given by name in `cmaps` and `limits`, so the same two arguments cover categorical
    features, continuous features and characters:

    >>> mt.pl.plot_tree(tree, annot=['MiTo_clone', 'clone_support'],
    ...                 cmaps={'MiTo_clone': {'MT-1': 'r'}, 'clone_support': 'viridis'},
    ...                 limits={'clone_support': (0, 1)})

    The tree's own elements are configured one dictionary each - `colorstrips`,
    `branches`, `leaves`, `internal_nodes`. Every one of them takes the same semantic
    keys where they apply ("feature", "cmap", "limits", "labels", "label_size") plus any
    matplotlib style argument, which is forwarded:

    >>> mt.pl.plot_tree(tree, orient='right',
    ...                 leaves={'feature': 'MiTo_clone', 'labels': True, 'markersize': 3},
    ...                 internal_nodes={'feature': 'fitness', 'show': True})

    Parameters
    ----------
    tree : CassiopeiaTree
        Tree to plot.
    annot : str or Iterable[str], optional
        Annotations to draw as colorstrips, from tree.cell_meta or tree.layers[`layer`].
        Default is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; one is created if not given. Default is None.
    orient : float or str, optional
        Polar layout (a number, the starting angle) or cartesian ("up", "down", "left",
        "right"). Default is 90.
    cmaps : dict, optional
        Colors per annotation: {name: palette name | list of colors | {category: color} |
        colormap}. Default is None (per-annotation defaults).
    limits : dict, optional
        {name: (vmin, vmax)} for continuous annotations. Default is None (10th-90th
        percentile).
    layer : str, optional
        Layer characters are read from. Default is "raw".
    colorstrips : dict, optional
        {"width", "spacing", "labels", "label_size", "label_offset"} plus Polygon style.
    branches : dict, optional
        {"feature", "meta", "cmap", "limits", "color", "annot", "where"} plus Line2D style. "meta" is the
        DataFrame holding the per-branch covariate. "annot" names one of the categorical
        annotations (e.g. "MiTo_clone"): a branch takes the color of that annotation in the
        colorstrip if all the cells below it share one label, and "color" (black) otherwise,
        including for the label "unassigned". With "where" = "leaves" (default) only the leaf
        branches are colored, so that the clones read as fans over a black tree; "all" colors every
        branch above a single-label clade.
    leaves : dict, optional
        {"feature", "cmap", "limits", "labels", "label_size", "label_offset", "color"}
        plus marker style.
    internal_nodes : dict, optional
        {"feature", "cmap", "limits", "labels", "label_size", "subset", "show", "color"}
        plus marker style. "feature" is a node attribute (see `mito.tl.compute_fitness`,
        `mito.tl.compute_expansions`).
    extend_branches : bool, optional
        True draws equal-length branches from leaves to root (topology only). Default is None:
        the true branch lengths for an ultrametric tree with varying lengths (UPGMA trees from
        `mito.tl.build_tree`), and equal-length branches otherwise (unit lengths, or trees whose
        leaves are not equidistant from the root, such as neighbor-joining).
    leaf_length : float, optional
        Length added to every leaf branch, so that a polytomy is drawn as a fan instead of as a
        point. Default is None: 4% of the tree depth when drawing true lengths, 0 otherwise.
    collapse_clones : str, optional
        Name of a categorical annotation (e.g. "MiTo_clone"). Every clade whose cells all share one
        label, other than "unassigned", is drawn as a single fan: its cells hang from the clade's MRCA
        by equal-length branches, which centers the MRCA on them, and an outgroup cell or subclade of
        the clone no longer looks like a separate lineage. The tree itself is not modified.
        Default is None.
    angled_branches : bool, optional
        Angled, rather than rounded, branches. Default is True.
    add_root : bool, optional
        Draw a root branch. Default is False.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes drawn on.
    """

    if legacy:
        renamed = { k:_LEGACY_ARGS[k] for k in legacy if k in _LEGACY_ARGS }
        unknown = [ k for k in legacy if k not in _LEGACY_ARGS ]
        msg = 'plot_tree got unexpected arguments.'
        if renamed:
            msg += ' These moved: ' + ', '.join(f'{k} -> {v}' for k, v in renamed.items()) + '.'
        if unknown:
            msg += f' Unknown: {unknown}.'
        raise TypeError(msg)

    annot = [] if annot is None else ([annot] if isinstance(annot, str) else list(annot))
    if collapse_clones is not None:
        tree = _collapse_clones(tree, collapse_clones, layer)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.axis('off')

    if extend_branches is None:
        depths = [tree.get_time(leaf) for leaf in tree.leaves]
        extend_branches = not (np.ptp(depths) <= 1e-9*max(depths) and max(depths) > 0 and
                               len({round(tree.get_branch_length(*e), 9) for e in tree.edges}) > 1)

    if leaf_length is None:
        leaf_length = 0. if extend_branches else .04*max(tree.get_time(leaf) for leaf in tree.leaves)

    strip_spec, strip_style = _split_spec(
        colorstrips,
        {'width':None, 'spacing':None, 'labels':True, 'label_size':10, 'label_offset':2},
        'colorstrips'
    )
    branch_spec, branch_style = _split_spec(
        branches,
        {'feature':None, 'meta':None, 'cmap':'Spectral_r', 'limits':None, 'color':'k', 'annot':None, 'where':'leaves'},
        'branches'
    )
    leaf_spec, leaf_style = _split_spec(
        leaves,
        {'feature':None, 'cmap':'tab20', 'limits':None, 'labels':False, 'label_size':5,
         'label_offset':1.5, 'color':'k'},
        'leaves'
    )
    node_spec, node_style = _split_spec(
        internal_nodes,
        {'feature':None, 'cmap':'Spectral_r', 'limits':(.2, .8), 'labels':False,
         'label_size':7, 'subset':None, 'show':False, 'color':'white'},
        'internal_nodes'
    )

    node_coords, branch_coords, strips = _place_tree(
        tree, annot, cmaps, limits, layer, orient, extend_branches, angled_branches,
        add_root, width=strip_spec['width'], spacing=strip_spec['spacing'], leaf_length=leaf_length
    )
    if branch_spec['annot'] is not None:
        branch_spec['colors'] = _clade_branch_colors(
            tree, branch_spec['annot'], annot, layer, cmaps, branch_spec['color'],
            leaves_only=branch_spec['where'] == 'leaves'
        )

    _draw_branches(ax, branch_coords, branch_spec, {'linewidth':1, **branch_style})
    _draw_colorstrips(ax, strips, orient, strip_spec, {'linewidth':0, 'alpha':1, **strip_style})
    _draw_leaves(
        ax, tree, node_coords, leaf_spec,
        {'markersize':2 if leaf_spec['feature'] is not None else 0, 'marker':'o', **leaf_style},
        orient
    )
    _draw_internal_nodes(
        ax, tree, node_coords, node_spec,
        {'markersize':2, 'marker':'o', 'alpha':1, 'markeredgecolor':'k',
         'markeredgewidth':1, 'zorder':10, **node_style}
    )

    return ax


##
