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


def _open_gaps(tree, node_coords, branch_coords, name, layer, gap, polar, orient):
    """
    Space between neighbouring groups of cells: `gap` leaf widths wherever the label of
    annotation `name` changes along the leaves (a run of unlabelled or "unassigned" cells is
    one group). The leaf axis is stretched piecewise, so nodes and branches follow.
    """

    x, _ = _annotation_values(tree, name, layer)
    group = x.astype(object).where(x.notna(), 'unassigned')
    leaves = sorted((n for n in node_coords if n in tree.nodes and tree.is_leaf(n)), key=lambda n: node_coords[n][0])
    old = np.array([node_coords[n][0] for n in leaves], dtype=float)
    change = np.array([0] + [int(group.loc[a] != group.loc[b]) for a, b in zip(leaves[:-1], leaves[1:], strict=True)])
    shift = np.cumsum(change) * gap
    step = np.min(np.diff(old)) if len(old) > 1 else 1.
    new = old + shift * step
    if polar:
        # keep the full circle: rescale about the first leaf
        total = (new[-1] - new[0]) + step
        new = old[0] + (new - new[0]) * (old[-1] - old[0] + step) / total

    def move(v):
        return np.interp(v, old, new)

    node_coords = {k: (float(move(c[0])), c[1]) for k, c in node_coords.items()}
    branch_coords = {k: ([float(v) for v in move(np.asarray(xs))], ys) for k, (xs, ys) in branch_coords.items()}

    return node_coords, branch_coords


##


def _place_tree(tree, annot, cmaps, limits, layer, orient, extend_branches,
                angled_branches, add_root, width=None, spacing=None, leaf_length=0., gap_by=None, gap=0.):
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

    if gap_by is not None and gap > 0:
        node_coords, branch_coords = _open_gaps(tree, node_coords, branch_coords, gap_by, layer, gap, is_polar, orient)

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

    raw_nodes = dict(node_coords)
    if is_polar:
        branch_coords = _to_polar_coords(branch_coords)
        node_coords = _to_polar_coords(node_coords)
        colorstrips = _to_polar_colorstrips(colorstrips)

    return node_coords, branch_coords, list(zip(colorstrips, annot, strict=True)), raw_nodes


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
    clone_nodes = {}
    stack = [tree.root]
    while stack:
        node = stack.pop()
        for child in tree.children(node):
            if label[child] not in (None, 'unassigned') and not tree.is_leaf(child):
                graph.add_edge(node, child)
                lengths[(node, child)] = tree.get_branch_length(node, child)
                clone_nodes[child] = list(tree.leaves_in_subtree(child))
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

    return collapsed, clone_nodes


##


def _label_colors(tree, name, annot, layer, cmaps):
    """{cell: color} of a categorical annotation, the colors of its colorstrip."""

    x, _ = _annotation_values(tree, name, layer)
    if _is_continuous(x):
        raise ValueError(f'"{name}" must be a categorical annotation.')
    names = [annot] if isinstance(annot, str) else list(annot or [])
    n_cat = sum(1 for a in names[:names.index(name)] if not _is_continuous(_annotation_values(tree, a, layer)[0])) \
        if name in names else 0
    colors, _, _ = _resolve_colors(
        x, cmap=(cmaps or {}).get(name), default=_categorical_cmaps[n_cat % len(_categorical_cmaps)]
    )
    return colors


def _draw_wedges(ax, tree, clone_nodes, raw_nodes, colors, orient, height, style, filled, colored):
    """
    Collapsed clones as short fans or wedges: a stem from the clone's MRCA to where, over the
    last `height` of the tree depth, an arc and one spoke per cell (or a filled triangle) as
    wide as its cells.
    """

    leaves = [n for n in raw_nodes if n in tree.nodes and tree.is_leaf(n)]
    polar = isinstance(orient, (float, int))
    if polar:
        rim = max(raw_nodes[n][1] for n in leaves)
        step = float(np.min(np.diff(sorted(raw_nodes[n][0] for n in leaves)))) if len(leaves) > 1 else 1.
    else:
        rim = max(abs(raw_nodes[n][1]) for n in leaves)
        step = float(np.min(np.diff(sorted(raw_nodes[n][0] for n in leaves)))) if len(leaves) > 1 else 1.
    reach = height * rim
    order = {n: i for i, n in enumerate(tree.depth_first_traverse_nodes())}

    for node, cells in clone_nodes.items():
        cells = sorted(cells, key=order.get)
        pos_lo, pos_hi = raw_nodes[cells[0]][0] - step/2, raw_nodes[cells[-1]][0] + step/2
        pos, depth = raw_nodes[node]
        depth = abs(depth) if not polar else depth
        apex = max(depth, rim - reach)
        color = colors[cells[0]] if colored else 'k'
        if polar:
            thetas = list(np.linspace(pos_lo, pos_hi, max(int(abs(pos_hi - pos_lo)), 2)))
            sx, sy = ut.polars_to_cartesians([pos, pos], [depth, apex])
        else:
            sx, sy = [pos, pos], [-depth, -apex]
        ax.plot(sx, sy, **{**style, 'c':'k'})
        if filled:
            if polar:
                xs, ys = ut.polars_to_cartesians([pos] + thetas, [apex] + [rim]*len(thetas))
            else:
                xs, ys = [pos, pos_lo, pos_hi], [-apex, -rim, -rim]
            ax.add_patch(Polygon(xy=list(zip(xs, ys, strict=True)), closed=True,
                                 **{'facecolor':colors[cells[0]], 'edgecolor':'k', 'linewidth':.3, 'alpha':.9}))
            continue
        pos_cells = [raw_nodes[c][0] for c in cells]
        if polar:
            arc = np.linspace(min(pos_cells), max(pos_cells), max(int(abs(max(pos_cells) - min(pos_cells))), 2))
            ax.plot(*ut.polars_to_cartesians(list(arc), [apex]*len(arc)), **{**style, 'c':color})
            for theta, cell in zip(pos_cells, cells, strict=True):
                ax.plot(*ut.polars_to_cartesians([theta, theta], [apex, raw_nodes[cell][1]]), **{**style, 'c':color})
        else:
            ax.plot([min(pos_cells), max(pos_cells)], [-apex, -apex], **{**style, 'c':color})
            for theta, cell in zip(pos_cells, cells, strict=True):
                ax.plot([theta, theta], [-apex, raw_nodes[cell][1]], **{**style, 'c':color})


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
    clone_style: str = 'fan',
    gap_by: str = None,
    gap: float = 3.,
    wedge_height: float = .15,
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
    clone_style : str, optional
        How a collapsed clone is drawn. "fan" (default): a stem from the MRCA to a short fan, an
        arc and one spoke per cell over the last `wedge_height` of the depth, black or in the
        clone's color if `branches` has an "annot". "wedge": the same extent as a filled triangle.
        Both are for the polar and `orient="down"` layouts. "full": every cell on its own line
        from the MRCA, which is dense when the MRCAs are close to the root.
    gap_by : str, optional
        Name of a categorical annotation (e.g. "MiTo_clone"): neighbouring groups of cells with a
        different label are separated by `gap`. Default is None (no gaps).
    gap : float, optional
        Space between groups, in cell widths. Default is 3.
    wedge_height : float, optional
        Fraction of the tree depth the short fans and wedges span. Default is 0.15.
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
    clone_nodes = {}
    if collapse_clones is not None:
        tree, clone_nodes = _collapse_clones(tree, collapse_clones, layer)
    wedges = bool(clone_nodes) and clone_style in ('fan', 'wedge') and (isinstance(orient, (float, int)) or orient == 'down')
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

    node_coords, branch_coords, strips, raw_nodes = _place_tree(
        tree, annot, cmaps, limits, layer, orient, extend_branches, angled_branches,
        add_root, width=strip_spec['width'], spacing=strip_spec['spacing'], leaf_length=leaf_length,
        gap_by=gap_by, gap=gap
    )
    if branch_spec['annot'] is not None:
        branch_spec['colors'] = _clade_branch_colors(
            tree, branch_spec['annot'], annot, layer, cmaps, branch_spec['color'],
            leaves_only=branch_spec['where'] == 'leaves'
        )

    if wedges:
        branch_coords = {b: c for b, c in branch_coords.items() if b[0] not in clone_nodes}
        _draw_wedges(
            ax, tree, clone_nodes, raw_nodes, _label_colors(tree, collapse_clones, annot, layer, cmaps),
            orient, wedge_height, {'linewidth':1, **branch_style}, clone_style == 'wedge',
            branch_spec['annot'] is not None
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


##


def plot_clone_tree(
    tree: CassiopeiaTree,
    annot: str = 'MiTo_clone',
    color_by: str|pd.Series = None,
    cmap: Any = None,
    limits: tuple[float,float] = None,
    show_unassigned: bool = False,
    orient: float = 90,
    pad: float = .12,
    linewidth: float = None,
    alpha: float = .9,
    max_radius: float = .09,
    ax: matplotlib.axes.Axes = None,
    ) -> matplotlib.axes.Axes:
    """
    Clone-level phylogeny: one tip per clone, a circle whose area is its number of cells.

    Every clade of the tree whose cells share one label of `annot` (e.g. "MiTo_clone") is
    replaced by a single tip at the same depth as its cells, so the branches keep their
    true lengths; the clades above are drawn as in `plot_tree`. The angular space of a tip
    grows with its circle, so circles never overlap.

    Parameters
    ----------
    tree : CassiopeiaTree
        Phylogeny (ultrametric, e.g. from `mito.tl.build_tree`), with the clone labels in
        tree.cell_meta[`annot`].
    annot : str, optional
        Categorical annotation naming the clones. Default is "MiTo_clone".
    color_by : str or pd.Series, optional
        Column of tree.cell_meta (mean per clone if numeric, most common value otherwise) or
        a Series of values per clone, indexed by clone label. Default is None: the colors of
        `annot` in `plot_tree`.
    cmap : str or dict, optional
        Colormap name for a numeric `color_by`, or {category: color}. Default is "Spectral_r".
    limits : tuple[float, float], optional
        Color range of a numeric `color_by`. Default is its 5th to 95th percentile.
    show_unassigned : bool, optional
        Draw each unassigned cell as a small grey tip. Default is False: they are left out.
    orient : float, optional
        Angle offset of the circular layout. Default is 90.
    pad : float, optional
        Free angular space around a tip, relative to the largest circle. Default is 0.12.
    linewidth : float, optional
        Branch width. Default is None: 0.8, or 0.3 when there are more than 60 tips.
    alpha : float, optional
        Opacity of the circles. Default is 0.9.
    max_radius : float, optional
        Largest circle radius, as a fraction of the radius of the tree. Default is 0.09.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Default is None (new figure).

    Returns
    -------
    matplotlib.axes.Axes
    """

    x, _ = _annotation_values(tree, annot, 'raw')
    label = {}
    for node in tree.depth_first_traverse_nodes(postorder=True):
        if tree.is_leaf(node):
            lab = x.loc[node] if pd.notna(x.loc[node]) else 'unassigned'
            label[node] = None if lab == 'unassigned' else lab
        else:
            kids = {label[c] for c in tree.children(node)}
            label[node] = kids.pop() if len(kids) == 1 else None
    depth = max(tree.get_time(leaf) for leaf in tree.leaves)

    # clone-level tree: a clone's clade is one tip; unassigned cells are tips or dropped
    graph, length, tip = nx.DiGraph(), {}, {}
    stack = [tree.root]
    while stack:
        node = stack.pop()
        for child in tree.children(node):
            if label[child] is not None:                  # a clone: one tip, at the cells' depth
                graph.add_edge(node, child)
                length[(node, child)] = tree.get_branch_length(node, child) + depth - tree.get_time(child)
                tip[child] = (label[child], list(tree.leaves_in_subtree(child)) if not tree.is_leaf(child) else [child])
            elif tree.is_leaf(child):                     # an unassigned cell
                if show_unassigned:
                    graph.add_edge(node, child)
                    length[(node, child)] = tree.get_branch_length(node, child)
                    tip[child] = (None, [child])
            else:
                graph.add_edge(node, child)
                length[(node, child)] = tree.get_branch_length(node, child)
                stack.append(child)
    # drop clades left without tips, then merge chains with a single child
    for node in [n for n in nx.dfs_postorder_nodes(graph, tree.root) if graph.out_degree(n) == 0 and n not in tip]:
        graph.remove_node(node)
    for node in [n for n in nx.dfs_postorder_nodes(graph, tree.root) if n in graph and graph.out_degree(n) == 1 and n != tree.root]:
        (child,) = graph.successors(node)
        (parent,) = graph.predecessors(node)
        graph.add_edge(parent, child)
        length[(parent, child)] = length[(parent, node)] + length[(node, child)]
        graph.remove_node(node)
    flat = CassiopeiaTree(tree=graph)
    for edge in graph.edges:
        flat.set_branch_length(*edge, length.get(edge, 0.) if edge in length else 0.)

    zero = [e for e in flat.edges if flat.get_branch_length(*e) <= 0]
    for e in zero:
        flat.set_branch_length(*e, 1e-6 * depth)
    node_coords, branch_coords = ut.place_tree(flat, orient=orient, extend_branches=False, angled_branches=True)

    # angular slots proportional to the circle sizes
    n_cells = {t: len(tip[t][1]) for t in tip}
    top = max(n_cells.values())
    size = {t: np.sqrt(n_cells[t] / top) for t in tip}
    if show_unassigned:
        size = {t: (size[t] if tip[t][0] is not None else .15 * 0 + .04) for t in tip}
    order = sorted(tip, key=lambda t: node_coords[t][0])
    widths = np.array([size[t] + pad for t in order])
    old = np.array([node_coords[t][0] for t in order], dtype=float)
    span = old[-1] - old[0] + (old[1] - old[0] if len(old) > 1 else 1.)
    edges = np.concatenate([[0], np.cumsum(widths)])
    new = old[0] + span * (edges[:-1] + widths / 2) / edges[-1]
    node_coords = {k: (float(np.interp(c[0], old, new)), c[1]) for k, c in node_coords.items()}
    branch_coords = {k: ([float(v) for v in np.interp(np.asarray(xs), old, new)], ys) for k, (xs, ys) in branch_coords.items()}
    rim = max(node_coords[t][1] for t in tip)
    unit = 2 * np.pi * rim * span / 360 / edges[-1]          # arc length of one unit of size
    unit = min(unit, 2 * max_radius * rim)                 # circles never outgrow their slot or the tree
    radius = {t: size[t] * unit / 2 for t in tip}
    linewidth = linewidth if linewidth is not None else (.8 if len(tip) <= 60 else .3)

    node_xy = _to_polar_coords(node_coords)
    branch_xy = _to_polar_coords(branch_coords)

    # colors
    per_clone = isinstance(color_by, pd.Series)
    clone_of = {t: tip[t][0] for t in tip}
    if color_by is None:
        palette = _label_colors(tree, annot, [annot], 'raw', None)
        face = {t: palette[tip[t][1][0]] if clone_of[t] is not None else '#bdbdbd' for t in tip}
        norm = colormap = None
    else:
        values = color_by if per_clone else tree.cell_meta[color_by]
        numeric = pd.api.types.is_numeric_dtype(values) and not isinstance(values.dtype, pd.CategoricalDtype)
        if numeric:
            stat = {t: (values.get(clone_of[t], np.nan) if per_clone else values.loc[tip[t][1]].mean())
                    for t in tip if clone_of[t] is not None}
            vals = pd.Series(stat)
            lo, hi = limits if limits is not None else (np.nanpercentile(vals, 5), np.nanpercentile(vals, 95))
            colormap = matplotlib.colormaps[cmap if isinstance(cmap, str) else 'Spectral_r']
            norm = plt.Normalize(lo, hi)
            face = {t: colormap(norm(stat[t])) if t in stat else '#bdbdbd' for t in tip}
        else:
            mode = {t: (values.get(clone_of[t]) if per_clone else values.loc[tip[t][1]].mode().iloc[0])
                    for t in tip if clone_of[t] is not None}
            palette = cmap if isinstance(cmap, dict) else _categorical_palette(values, default=_categorical_cmaps[0])
            face = {t: palette.get(mode[t], _NA_COLOR) if t in mode else '#bdbdbd' for t in tip}
            norm = colormap = None

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    for xs, ys in branch_xy.values():
        ax.plot(xs, ys, color='k', linewidth=linewidth, solid_capstyle='round')
    for t in tip:
        cx, cy = node_xy[t]
        ax.add_patch(plt.Circle((cx, cy), radius[t], facecolor=face[t], edgecolor='k', linewidth=.3, alpha=alpha, zorder=5))

    # size legend: circles of round numbers of cells
    refs = [r for r in (10, 100, 1000, 10000) if r <= top] or [top]
    refs = refs[-3:]
    lim = (rim + max(radius.values())) * 1.08
    x0, y0 = lim * .62, -lim * .92
    for i, r in enumerate(refs):
        rr = np.sqrt(r / top) * unit / 2
        ax.add_patch(plt.Circle((x0 + i * lim * .14, y0 + rr), rr, facecolor='none', edgecolor='k', linewidth=.5))
        ax.text(x0 + i * lim * .14, y0 - lim * .01, str(r), ha='center', va='top', fontsize=7)
    ax.text(x0 - lim * .06, y0 + lim * .02, 'cells', ha='right', va='center', fontsize=7)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal'); ax.axis('off')
    if norm is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
        ax.figure.colorbar(sm, ax=ax, shrink=.35, label=color_by.name if per_clone else color_by)

    return ax
