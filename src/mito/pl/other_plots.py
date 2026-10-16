"""
Other plots (i.e., packed_circle_plot)
"""

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotting_utils as plu
from cassiopeia.data import CassiopeiaTree
from circlify import Circle, circlify

from .phylo_plots import _annotation_values, _categorical_cmaps, _categorical_palette, _label_colors, _NA_COLOR

##


def packed_circle_plot(
    df: pd.DataFrame,
    ax: matplotlib.axes.Axes = None,
    covariate: str = None,
    color: str = None,
    cmap: dict[str,Any] = None,
    color_by: str = None,
    alpha: float = .5, linewidth: float = 1.2,
    t_cov: float = .01, annotate: bool = False,
    fontsize: float = 6, ascending: bool = False,
    fontcolor: Any = 'white',
    fontweight: str ='normal'
    ) -> matplotlib.axes.Axes:
    """
    Circle plot. Packed.

    `df` is either a DataFrame with the sizes in column `covariate`, or a Series of
    sizes (e.g. `afm.obs["MiTo_clone"].value_counts()`).
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    if isinstance(df, pd.Series):
        covariate = covariate or (df.name or 'size')
        df = df.to_frame(covariate)
    elif covariate is None:
        if df.shape[1] != 1:
            raise ValueError('With a DataFrame of several columns, name the one to plot in `covariate`.')
        covariate = df.columns[0]

    df = df.sort_values(covariate, ascending=False)
    circles = circlify(
        df[covariate].to_list(),
        show_enclosure=True,
        target_enclosure=Circle(x=0, y=0, r=1)
    )
    lim = max(
        max(
            abs(c.x) + c.r,
            abs(c.y) + c.r,
        )
        for c in circles
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # Colors
    if color is None and cmap is None:
        colors = dict.fromkeys(df.index, 'grey')        # plain default
    elif isinstance(color, str) and color not in df.columns:
        colors = dict.fromkeys(df.index, color)
    elif isinstance(cmap, str) and color_by in df.columns:
        colors = plu.create_palette(df, color_by, cmap, order=df.index[::-1]) # Assumes index contains IDs
    elif isinstance(cmap, dict):
        colors = cmap
    else:
        raise TypeError(
            'Provide either a single "color", a "cmap" name together with a '
            '"color_by" column, or a {category: color} "cmap" dictionary.'
        )

    # Plot circles
    for name, circle in zip(df.index[::-1], circles, strict=False): # Don't know why, but it reverses...
        x, y, r = circle
        ax.add_patch(
            plt.Circle((x, y), r*0.95, alpha=alpha, linewidth=linewidth,
                fill=True, edgecolor=colors[name], facecolor=colors[name])
        )
        if annotate:
            cov = df.loc[name, covariate]
            if cov > t_cov:
                n = name if len(name)<=5 else name[:5]
                ax.annotate(
                    f'{n}: {df.loc[name, covariate]:.2f}',
                    (x,y),
                    va='center', ha='center',
                    fontweight=fontweight, fontsize=fontsize, color=fontcolor,
                )

    ax.axis('off')

    return ax


##


def clone_packing(
    tree: CassiopeiaTree,
    annot: str = 'MiTo_clone',
    color_by: str|pd.Series = None,
    cmap: Any = None,
    limits: tuple[float,float] = None,
    max_levels: int = 3,
    show_unassigned: bool = False,
    dominant: float = .85,
    alpha: float = .85,
    ax: matplotlib.axes.Axes = None,
    ) -> matplotlib.axes.Axes:
    """
    Clones as circles packed inside their clades: a clone-level view of the phylogeny.

    Every clone of `annot` (e.g. "MiTo_clone") is a filled circle whose area is its number of
    cells. The clades of the tree that contain several clones are the open circles around
    them, so neighbouring clones are siblings in the tree. Chains of clades with a single
    clone below are skipped, and clades more than `max_levels` deep are merged into the
    last enclosing circle, to keep hundreds of clones readable.

    Parameters
    ----------
    tree : CassiopeiaTree
        Phylogeny, with the clone labels in tree.cell_meta[`annot`].
    annot : str, optional
        Categorical annotation naming the clones. Default is "MiTo_clone".
    color_by : str or pd.Series, optional
        Column of tree.cell_meta to color the clones by: its mean per clone if numeric, its
        most common value if categorical. Or a Series of values per clone, indexed by clone label. Default is None: the colors of `annot` in `plot_tree`.
    cmap : str or dict, optional
        Colormap name for a numeric `color_by`, or {category: color} for a categorical one.
        Default is "Spectral_r".
    limits : tuple[float, float], optional
        Color range of a numeric `color_by`. Default is its 5th to 95th percentile.
    max_levels : int, optional
        Number of nested levels of enclosing clades. Default is 3.
    show_unassigned : bool, optional
        Draw the unassigned cells, grey, as one circle each. Default is False.
    dominant : float, optional
        A clade holding at least this fraction of the cells of its parent clade is not drawn as
        a separate circle, which would only repeat the parent. Default is 0.85.
    alpha : float, optional
        Opacity of the clone circles. Default is 0.85.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Default is None (new figure).

    Returns
    -------
    matplotlib.axes.Axes
    """

    x, _ = _annotation_values(tree, annot, 'raw')
    label = {}
    items = {}
    for node in tree.depth_first_traverse_nodes(postorder=True):
        if tree.is_leaf(node):
            lab = x.loc[node] if pd.notna(x.loc[node]) else 'unassigned'
            label[node] = None if lab == 'unassigned' else lab
            items[node] = ([{'id': node, 'datum': 1, 'cells': [node], 'clone': label[node]}]
                           if label[node] is not None or show_unassigned else [])
            continue
        kids = {label[c] for c in tree.children(node)}
        label[node] = kids.pop() if len(kids) == 1 else None
        merged = [it for ch in tree.children(node) for it in items.pop(ch)]
        if label[node] is not None:                      # a single clone: its clade is one circle
            cells = [c for it in merged for c in it['cells']]
            items[node] = [{'id': node, 'datum': len(cells), 'cells': cells, 'clone': label[node]}]
        elif len(merged) > 1:                            # a clade holding several clones/cells
            total = sum(it['datum'] for it in merged)
            lifted = True
            while lifted:                                # a dominant sub-clade is not a separate circle
                lifted = False
                for i, it in enumerate(merged):
                    if 'children' in it and it['datum'] >= dominant * total:
                        merged = merged[:i] + it['children'] + merged[i+1:]
                        lifted = True
                        break
            items[node] = [{'id': node, 'datum': total, 'children': merged, 'cells': [], 'clone': None}]
        else:                                            # a chain with one thing below it: skipped
            items[node] = merged

    top = _merge_pieces(items[tree.root])
    if len(top) == 1 and 'children' in top[0]:
        top = top[0]['children']
    data = [d for d in _limit_levels(top, max_levels) if d['datum'] > 0]

    meta = tree.cell_meta
    per_clone = isinstance(color_by, pd.Series)
    if color_by is not None:
        values = color_by if per_clone else meta[color_by]
        numeric = pd.api.types.is_numeric_dtype(values) and not isinstance(values.dtype, pd.CategoricalDtype)
    leaves = {}
    _collect_leaves(data, leaves)
    if color_by is None:
        colors = _label_colors(tree, annot, [annot], 'raw', None)
        facecolor = {k: colors[v['cells'][0]] if v['clone'] is not None else '#bdbdbd' for k, v in leaves.items()}
    elif numeric:
        stat = {k: (values.get(v['clone'], np.nan) if per_clone else values.loc[v['cells']].mean())
                for k, v in leaves.items() if v['clone'] is not None}
        vals = pd.Series(stat)
        lo, hi = limits if limits is not None else (np.nanpercentile(vals, 5), np.nanpercentile(vals, 95))
        colormap = matplotlib.colormaps[cmap if isinstance(cmap, str) else 'Spectral_r']
        norm = plt.Normalize(lo, hi)
        facecolor = {k: colormap(norm(stat[k])) if k in stat else '#bdbdbd' for k in leaves}
    else:
        mode = {k: (values.get(v['clone']) if per_clone else values.loc[v['cells']].mode().iloc[0])
                for k, v in leaves.items() if v['clone'] is not None}
        palette = cmap if isinstance(cmap, dict) else _categorical_palette(values, default=_categorical_cmaps[0])
        facecolor = {k: palette.get(mode[k], _NA_COLOR) if k in mode else '#bdbdbd' for k in leaves}

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    circles = circlify(_clean(data), show_enclosure=False, target_enclosure=Circle(x=0, y=0, r=1))
    for c in circles:
        ex = c.ex
        if 'children' in ex:
            ax.add_patch(plt.Circle((c.x, c.y), c.r, fill=False, edgecolor='#555555', linewidth=.5))
        else:
            ax.add_patch(plt.Circle((c.x, c.y), c.r*.94, facecolor=facecolor[ex['id']], alpha=alpha,
                                    edgecolor='#333333', linewidth=.3))
    ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02); ax.set_aspect('equal'); ax.axis('off')
    if color_by is not None and numeric:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
        ax.figure.colorbar(sm, ax=ax, shrink=.35, label=color_by.name if per_clone else color_by)

    return ax


def _merge_pieces(items):
    """
    One circle per clone: a label whose cells are not a single clade of the tree (several
    pieces) is drawn once, where its largest piece is, with all its cells.
    """

    pieces = {}
    stack = list(items)
    while stack:
        it = stack.pop()
        if 'children' in it:
            stack.extend(it['children'])
        elif it['clone'] is not None:
            pieces.setdefault(it['clone'], []).append(it)
    keep = {}
    for clone, group in pieces.items():
        group.sort(key=lambda it: -it['datum'])
        main = group[0]
        main['cells'] = [c for it in group for c in it['cells']]
        main['datum'] = len(main['cells'])
        keep[clone] = id(main)

    def prune(its):
        out = []
        for it in its:
            if 'children' in it:
                children = prune(it['children'])
                if children:
                    out.append({**it, 'children': children, 'datum': sum(c['datum'] for c in children)})
            elif it['clone'] is None or keep[it['clone']] == id(it):
                out.append(it)
        return out

    return prune(items)


def _limit_levels(items, max_levels, level=1):
    """Items with at most `max_levels` nested levels of children; deeper clones are flattened."""

    out = []
    for it in items:
        if 'children' not in it:
            out.append(it)
        elif level >= max_levels:
            flat = []
            _collect_leaf_items(it['children'], flat)
            out.append({**it, 'children': flat})
        else:
            out.append({**it, 'children': _limit_levels(it['children'], max_levels, level + 1)})
    return out


def _collect_leaf_items(items, out):
    stack = list(items)
    while stack:
        it = stack.pop()
        if 'children' in it:
            stack.extend(it['children'])
        else:
            out.append(it)


def _collect_leaves(items, out):
    stack = list(items)
    while stack:
        it = stack.pop()
        if 'children' in it:
            stack.extend(it['children'])
        else:
            out[it['id']] = it


def _clean(items):
    """The items circlify needs: id, datum and children."""

    return [{'id': it['id'], 'datum': it['datum'], **({'children': _clean(it['children'])} if 'children' in it else {})}
            for it in items]
