"""
plotting_base.py stores plotting utilities and 'base plots', i.e., 
simple plots returning an Axes object.
"""

import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Iterable, Any
from circlify import circlify, Circle
from plotting_utils._plotting_base import create_handles, add_wilcox
from .colors import create_palette
from ..ut.utils import update_params


##


# Params
axins_pos = {

    'v2' : ( (.95,.75,.01,.22), 'left', 'vertical' ),
    'v3' : ( (.95,.05,.01,.22), 'left','vertical' ),
    'v1' : ( (.05,.75,.01,.22), 'right', 'vertical' ),
    'v4' : ( (.05,.05,.01,.22), 'right', 'vertical' ),

    'h2' : ( (1-.27,.95,.22,.01), 'bottom', 'horizontal' ),
    'h3' : ( (1-.27,.05,.22,.01), 'top', 'horizontal' ),
    'h1' : ( (0.05,.95,.22,.01), 'bottom', 'horizontal' ),
    'h4' : ( (0.05,.05,.22,.01), 'top', 'horizontal' ),

    'outside' : ( (1.05,.25,.03,.5), 'right', 'vertical' )
}


##


def set_rcParams():
    """
    Applies Nature Methods journal-style settings for matplotlib figures.
    """
    plt.rcParams.update({

        # Figure dimensions and DPI
        # 'figure.figsize': (7, 3.5),  # Recommended size for 1-row, 2-column figure
        # 'figure.dpi': 300,           # High DPI for print quality

        # Font settings
        # 'font.size': 7,                # Base font size
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],  # Preferred font for Nature figures

        # Axes properties
        # 'axes.titlesize': 8,           # Title font size
        # 'axes.labelsize': 7,           # Label font size
        # 'axes.linewidth': 0.5,         # Minimum line width for axes

        # Tick properties
        # 'xtick.labelsize': 6,
        # 'ytick.labelsize': 6,
        # 'xtick.direction': 'in',
        # 'ytick.direction': 'in',
        # 'xtick.major.size': 3,         # Major tick length
        # 'ytick.major.size': 3,
        # 'xtick.minor.size': 1.5,       # Minor tick length
        # 'ytick.minor.size': 1.5,
        # 'xtick.major.width': 0.5,      # Tick width
        # 'ytick.major.width': 0.5,

        # Legend properties
        # 'legend.fontsize': 6, 
        # # Line properties
        # 'lines.linewidth': 1,          # Line width for main data elements
        # 'lines.markersize': 4,         # Marker size
    })


##


def add_cbar(
    x: Iterable[int|float], palette: str = 'viridis', 
    ax: matplotlib.axes.Axes = None, 
    label_size: int = 10, ticks_size: int = 10, 
    vmin: int|float = None, vmax: int|float = None, 
    label: str = None, 
    layout: str|Dict[str,Tuple[Tuple[float,float,float,float],str,str]]='h1'
    ):
    """
    Draw a colorbar on the provided ax=matplotlib.axes.Axes object inset.
    Example layout: 'h1', or ( (0.05,.95,.22,.01), 'bottom', 'horizontal' )
    """

    if layout in axins_pos:
        pos, xticks_position, orientation = axins_pos[layout]
    else:
        pos, xticks_position, orientation= layout
        
    cmap = matplotlib.colormaps[palette]
    if vmin is None and vmax is None:
        norm = matplotlib.colors.Normalize(
            vmin=np.percentile(x, q=25), vmax=np.percentile(x, q=75))
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    axins = ax.inset_axes(pos) 
    
    cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), 
        cax=axins, orientation=orientation, ticklocation=xticks_position
    )
    cb.set_label(label=label, size=label_size, loc='center')
    if orientation == 'vertical':
        cb.ax.tick_params(axis="y", labelsize=ticks_size)
    else:
        cb.ax.tick_params(axis="x", labelsize=ticks_size)
    

##


def add_legend(
    label: str = None, colors: Dict[str,Any] = None, 
    ax: matplotlib.axes.Axes = None, loc: str = 'center', 
    artists_size: int = 10, label_size: int = 10, 
    ticks_size: int = 5, 
    bbox_to_anchor: Tuple[float,float] = (0.5, 1.1), 
    ncols: int = 1, 
    only_top: str|int = 'all'
    ):
    """
    Draw a legend on a ax: matplotlib.axes.Axes object.
    """
    
    try:
        del colors['unassigned']
        del colors[np.nan]
    except:
        pass

    if only_top != 'all':
        colors = { k : colors[k] for i, k in enumerate(colors) if i < int(only_top) }
    title = label if label is not None else None

    handles = create_handles(colors.keys(), colors=colors.values(), size=artists_size)
    legend = ax.legend(
        handles, colors.keys(), frameon=False, loc=loc, fontsize=ticks_size, 
        title_fontsize=label_size, ncol=ncols, title=title, 
        bbox_to_anchor=bbox_to_anchor
    )
    ax.add_artist(legend)


##


def format_ax(
    ax: matplotlib.axes.Axes = None, 
    title: str = '', xlabel: str = '', ylabel: str = '', 
    xticks: Iterable[Any] = None, 
    yticks: Iterable[Any] = None, 
    rotx: float = 0, roty: float = 0, axis: bool = True,
    xlabel_size: float = None, ylabel_size: float = None,
    xticks_size: float = None, yticks_size: float = None,
    title_size: float = None, 
    log: bool = False, 
    reduced_spines: bool = False
    ) -> matplotlib.axes.Axes:
    """
    Format labels, ticks and stuff of an ax: matplotlib.axes.Axes object.
    """

    if log:
        ax.set_yscale('log')
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

    if xticks is not None:
        ax.set_xticks([ i for i in range(len(xticks)) ])
        ax.set_xticklabels(xticks)
    if yticks is not None:
        ax.set_yticks([ i for i in range(len(yticks)) ])
        ax.set_yticklabels(yticks)

    if xticks_size is not None:
        ax.xaxis.set_tick_params(labelsize=xticks_size)
    if yticks_size is not None:
        ax.yaxis.set_tick_params(labelsize=yticks_size)

    if xlabel_size is not None:
        ax.xaxis.label.set_size(xlabel_size)
    if ylabel_size is not None:
        ax.yaxis.label.set_size(ylabel_size)

    ax.tick_params(axis='x', labelrotation = rotx)
    ax.tick_params(axis='y', labelrotation = roty)

    if title_size is not None:
        ax.set_title(title, fontdict={'fontsize': title_size})
    
    if reduced_spines:
        ax.spines[['right', 'top']].set_visible(False)
    
    if not axis:
        ax.axis('off')

    return ax


##


def bar(
    df: pd.DataFrame, y: Any, x: str = None, 
    by: str = None, c: Any ='grey', s: float = 0.35, 
    a: float = 1, l: float = None,
    ax: matplotlib.axes.Axes = None, 
    edgecolor: Any = None, annot_size: float = 10, 
    fmt: str = ".2f", 
    annot: bool = True
    ) -> matplotlib.axes.Axes:
    """
    Basic bar plot.
    """

    if isinstance(c, str) and by is None:
        if x is None:
            x = np.arange(df[y].size)
        bars = ax.bar(x, df[y], align='center', width=s, alpha=a, color=c, edgecolor=edgecolor)
        if annot:
            ax.bar_label(bars, df[y].values, padding=0, fmt=fmt, fontsize=annot_size)

    elif by is not None and x is None and isinstance(c, dict):
        x = np.arange(df[y].size)
        categories = df[by].unique()
        if all([cat in c for cat in categories]):
            for idx,cat in enumerate(categories):
                idx = df[by] == cat
                height = df.loc[idx, y].values
                x_positions = x[idx]
                bars = ax.bar(x_positions, height, align='center', width=s, alpha=a, color=c[cat], edgecolor=edgecolor)
                if annot:
                    ax.bar_label(bars, height, padding=0, fmt=fmt, fontsize=annot_size)
        else:
            raise ValueError(f'{by} categories do not match provided colors keys')

    elif by is not None and x is not None and isinstance(c, dict):
        ax = sns.barplot(data=df, x=x, y=y, hue=by, ax=ax, width=s, 
                         palette=c, alpha=a)
        ax.legend([], [], frameon=False)
        ax.set(xlabel='', ylabel='')
        ax.set_xticklabels(np.arange(df[x].nunique()))
        if annot:
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:{fmt}}', 
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', 
                            fontsize=annot_size)
    else:
        raise ValueError(f'Invalid combination of parameters.')

    return ax


##


def box(
    df: pd.DataFrame, x: Any, y: Any,
    by: str = None, c: Any = 'grey', 
    saturation: float = 0.7, ax:  matplotlib.axes.Axes = None, 
    with_stats: bool = False, pairs: Iterable[Tuple[str,str]]=None,
    order: Iterable[Any] = None, hue_order: Iterable[Any] =None, 
    kwargs: Dict[str,Any] = {}
    ) -> matplotlib.axes.Axes:
    """
    Base box plot.
    """

    params = {   
        'showcaps' : False,
        'fliersize': 0,
        'boxprops' : {'edgecolor': 'black', 'linewidth': .8}, 
        'medianprops': {"color": "black", "linewidth": 1.5},
        'whiskerprops':{"color": "black", "linewidth": 1.2}
    }

    params = update_params(params, kwargs)
    
    if isinstance(c, str) and by is None:
        sns.boxplot(data=df, x=x, y=y, color=c, ax=ax, saturation=saturation, order=order, **params) 
        ax.set(xlabel='')

    elif isinstance(c, dict) and by is None:
        if all([ True if k in df[x].unique() else False for k in c.keys() ]):
            palette = [c[category] for category in order]
            sns.boxplot(data=df, x=x, y=y, palette=palette, ax=ax, saturation=saturation, order=order, **params)
            ax.set(xlabel='')
        else:
            raise ValueError(f'{by} categories do not match provided colors keys')
            
    elif isinstance(c, dict) and by is not None:
        if all([ True if k in df[by].unique() else False for k in c.keys() ]):
            sns.boxplot(data=df, x=x, y=y, palette=c.values(), hue=by, hue_order=hue_order, ax=ax, saturation=saturation, **params)
            ax.legend([], [], frameon=False)
            ax.set(xlabel='')
        else:
            raise ValueError(f'{by} categories do not match provided colors keys')
    elif isinstance(c, str) and by is not None:
        sns.boxplot(data=df, x=x, y=y, hue=by, hue_order=hue_order, ax=ax, saturation=saturation, **params)
        ax.legend([], [], frameon=False)
        ax.set(xlabel='')

    if with_stats:
        add_wilcox(df, x, y, pairs, ax, order=order)

    return ax


##


def plot_heatmap(
    df: pd.DataFrame, 
    palette: str = 'mako', 
    ax: matplotlib.axes.Axes = None, 
    title: str = None, 
    x_names: bool = True, y_names: bool = True, 
    x_names_size: float = 7, y_names_size: float = 7, 
    xlabel: Iterable[Any] = None, ylabel: Iterable[Any] = None, 
    annot: bool = False, annot_size: float = 5, 
    label: str = None, shrink: float = 1.0, cb: bool = True, 
    vmin: float = None, vmax: float = None, 
    rank_diagonal: bool = False, 
    outside_linewidth: float = 1, linewidths: float = 0, 
    linecolor: Any = 'white'
    ) -> matplotlib.axes.Axes:
    """
    Simple heatmap.
    """
    if rank_diagonal:
        row_order = np.sum(df>0, axis=1).sort_values()[::-1].index
        col_order = df.mean(axis=0).sort_values()[::-1].index
        df = df.loc[row_order, col_order] 

    ax = sns.heatmap(data=df, ax=ax, robust=True, cmap=palette, annot=annot, xticklabels=x_names, 
        yticklabels=y_names, fmt='.2f', annot_kws={'size':annot_size}, cbar=cb,
        cbar_kws={'fraction':0.05, 'aspect':35, 'pad': 0.02, 'shrink':shrink, 'label':label},
        vmin=vmin, vmax=vmax, linewidths=linewidths, linecolor=linecolor
    )
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=x_names_size)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=y_names_size)

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(outside_linewidth)

    return ax


##

        
def packed_circle_plot(
    df: pd.DataFrame,
    ax: matplotlib.axes.Axes = None,
    covariate: str = None, color: Any = 'b', 
    cmap: Dict[str,Any] = None, 
    alpha: float = .5, linewidth: float = 1.2,
    t_cov: float = .01, annotate: bool = False, 
    fontsize: float = 6, ascending: bool = False, 
    fontcolor: Any = 'white', 
    fontweight: str ='normal'
    ) -> matplotlib.axes.Axes:

    """
    Circle plot. Packed.
    """

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
    
    if isinstance(color, str) and not color in df.columns:
        colors = { k : color for k in df.index }
    elif isinstance(color, str) and color in df.columns:
        c_cont = create_palette(
            df.sort_values(color, ascending=True),
            color, cmap
        )
        colors = {}
        for name in df.index:
            colors[name] = c_cont[df.loc[name, color]]
    else:
        assert isinstance(color, dict)
        colors = color
        print('Try to use custom colors...')

    for name, circle in zip(df.index[::-1], circles): # Don't know why, but it reverses...
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