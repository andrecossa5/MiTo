"""
plotting_base.py stores plotting utilities and 'base plots', i.e., 
simple plots returning an Axes object.
"""

import numpy as np 
import pandas as pd 

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
from circlify import circlify, Circle
from statannotations.Annotator import Annotator 
import textalloc as ta
from plotting_utils._plotting_base import *
from plotting_utils._utils import *
from plotting_utils._colors import *


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
# 
        # # Line properties
        # 'lines.linewidth': 1,          # Line width for main data elements
        # 'lines.markersize': 4,         # Marker size
    })



##


def create_handles(categories, marker='o', colors=None, size=10, width=0.5):
    """
    Create quick and dirty circular and labels for legends.
    """
    if colors is None:
        colors = sns.color_palette('tab10')[:len(categories)]

    handles = [ 
        (Line2D([], [], 
        marker=marker, 
        label=l, 
        linewidth=0,
        markersize=size, 
        markeredgewidth=width, 
        markeredgecolor='white', 
        markerfacecolor=c)) \
        for l, c in zip(categories, colors) 
    ]

    return handles


##


def add_cbar(x, palette='viridis', ax=None, label_size=7, ticks_size=5, 
    vmin=None, vmax=None, label=None, layout='h1'):
    """
    Draw cbar on an axes object inset.
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


def add_legend(label=None, colors=None, ax=None, loc='center', artists_size=7, label_size=7, 
    ticks_size=5, bbox_to_anchor=(0.5, 1.1), ncols=1, only_top='all'):
    """
    Draw a legend on axes object.
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


def add_wilcox(df, x, y, pairs, ax, order=None):
    """
    Add statisticatl annotations.
    """
    annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order)
    annotator.configure(
        test='Mann-Whitney', 
        text_format='star', 
        show_test_name=False,
        line_height=0.001, 
        text_offset=3
    )
    annotator.apply_and_annotate()


##


def find_n_axes(df, facet, query=None):
    """
    Get the numbers: n_axes.
    """
    idxs = []
    names = []
    n_axes = 0
    cats = df[facet].unique()
    for x in cats:
        idx = df.loc[df[facet] == x, :].index
        if query is not None:
            df_ = df.loc[idx, :].query(query)
        else:
            df_ = df.loc[idx, :]
        if df_.shape[0] > 0:
            n_axes += 1
            idxs.append(idx)
            names.append(f'{facet}: {x}')
        else:
            pass
    return n_axes, idxs, names


##


def remove_ticks(ax):
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)


##


def find_n_rows_n_cols(n_axes, n_cols=None):
    """
    Get the numbers: n_rows, n_cols.
    """
    n_cols = n_cols if n_cols is not None else 5
    if n_axes <= 5:
        n_rows = 1; n_cols = n_cols
    else:
        n_rows = 1; n_cols = n_cols
        while n_cols * n_rows < n_axes:
            n_rows += 1
    return n_rows, n_cols


##


def format_ax(ax, title='', xlabel='', ylabel='', 
    xticks=None, yticks=None, rotx=0, roty=0, axis=True,
    xlabel_size=None, ylabel_size=None, xticks_size=None, 
    yticks_size=None, title_size=None, log=False, reduced_spines=False
    ):
    """
    Format labels, ticks and stuff.
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


def add_labels_on_loc(df, x, y, by, ax=None, s=10):
    """
    Add categorical labels on loc on a scatterplot.
    """
    coords = df.loc[:, [x, y, by]].groupby(by).median()
    for label in coords.index:
        x, y = coords.loc[label, :].tolist()
        ax.text(x, y, label, fontsize=s, weight="bold")


##


def line(df, x, y, c='r', s=1, l=None, ax=None):
    """
    Base line plot.
    """
    ax.plot(df[x], df[y], color=c, label=l, linestyle='-', linewidth=s)
    return ax


##


def scatter(df, x, y, by=None, c='r', s=1.0, a=1, l=None, ax=None, scale_x=None, 
            vmin=None, vmax=None, ordered=False):
    """
    Base scatter plot.
    """
    size = s if isinstance(s, float) or isinstance(s, int) else df[s]

    if ordered and df[by].dtype == 'category':
        try:
            categories = df[by].cat.categories
            df = df.sort_values(by)
        except:
            raise ValueError('Ordered is not a pd.Categorical')

    if isinstance(size, pd.Series) and scale_x is not None:
        size = size * scale_x

    if isinstance(c, str) and by is None:
        ax.scatter(df[x], df[y], color=c, label=l, marker='.', s=size, alpha=a)

    elif isinstance(c, str) and by is not None:
        
        vmin = vmin if vmin is not None else np.percentile(df[by],25)
        vmax = vmax if vmax is not None else np.percentile(df[by],75)
        ax.scatter(df[x], df[y], c=df[by], cmap=c, vmin=vmin, vmax=vmax, label=l, marker='.', s=size, alpha=a)

    elif isinstance(c, dict) and by is not None:
        assert all([ x in c for x in df[by].unique() ])
        colors = [ c[x] for x in df[by] ]
        ax.scatter(df[x], df[y], c=colors, label=l, marker='.', s=size, alpha=a)

    else:
        raise ValueError('c needs to be specified as a dict of colors with "by" of a single color.')

    return ax


##


def hist(df, x, n=10, by=None, c='r', a=1, l=None, ax=None, density=False):
    """
    Basic histogram plot.
    """
    if by is None:
       ax.hist(df[x], bins=n, color=c, alpha=a, label=l, density=density)
    elif by is not None and isinstance(c, dict):
        categories = df[by].unique()
        if all([ cat in list(c.keys()) for cat in categories ]):
            for cat in categories:
                df_ = df.loc[df[by] == cat, :]
                ax.hist(df_[x], bins=n, color=c[cat], alpha=a, label=x, density=density)
    else:
        raise ValueError(f'{by} categories do not match provided colors keys')

    return ax


##




def bar(df, y, x=None, by=None, c='grey', s=0.35, a=1, l=None, ax=None, 
        edgecolor=None, annot_size=10, fmt=".2f", annot=True):
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


def box(df, x, y, by=None, c='grey', saturation=0.7, ax=None, with_stats=False,
    pairs=None, order=None, hue_order=None, kwargs={}):
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


def strip(df, x, y, by=None, c=None, a=1, l=None, s=5, ax=None, with_stats=False, order=None, pairs=None):

    """
    Base stripplot.
    """
    np.random.seed(123)
    
    if isinstance(c, str):
        ax = sns.stripplot(data=df, x=x, y=y, color=c, ax=ax, size=s, alpha=a, order=order, edgecolor='k')
        ax.set(xlabel='')
    
    elif isinstance(c, str) and by is not None:
        g = sns.stripplot(data=df, x=x, y=y, hue=by, palette=c, ax=ax, order=order, alpha=a, edgecolor='k')
        g.legend_.remove()

    elif isinstance(c, dict) and by is None:
        if all([ True if k in df[x].unique() else False for k in c.keys() ]):
            palette = [c[category] for category in order]
            ax = sns.stripplot(data=df, x=x, y=y, palette=palette, ax=ax, size=s, order=order, alpha=a, edgecolor='k')
            ax.set(xlabel='')
        else:
            raise ValueError(f'{by} categories do not match provided colors keys')
            
    elif isinstance(c, dict) and by is not None:
        if all([ True if k in df[by].unique() else False for k in c.keys() ]):
            ax = sns.stripplot(data=df, x=x, y=y, palette=c.values(), hue=by, ax=ax, size=s, order=order, alpha=a, edgecolor='k')
            ax.legend([], [], frameon=False)
            ax.set(xlabel='')    
        else:
            raise ValueError(f'{by} categories do not match provided colors keys')

    if with_stats:
        add_wilcox(df, x, y, pairs, ax, order=None)

    return ax


##


def violin(df, x, y, by=None, c=None, a=1, l=None, ax=None, with_stats=False, order=None, pairs=None):
    """
    Base violinplot.
    """
    params = {   
        'showcaps' : False,
        'fliersize': 0,
        'boxprops' : {'edgecolor': 'white', 'linewidth': 0.5}, 
        'medianprops': {"color": "white", "linewidth": 1.2},
        'whiskerprops':{"color": "black", "linewidth": 1}
    }
    
    if isinstance(c, str):
        ax = sns.violinplot(data=df, x=x, y=y, color=c, ax=ax, saturation=.7, order=order, **params) 
        ax.set(xlabel='', ylabel='')
        ax.set_xticklabels(np.arange(df[x].unique().size))

    elif isinstance(c, dict) and by is None:
        ax = sns.violinplot(data=df, x=x, y=y, palette=c.values(), ax=ax, saturation=.7, order=order, **params)
        ax.set(xlabel='', ylabel='') 
        ax.set_xticklabels(np.arange(df[x].unique().size))
            
    elif isinstance(c, dict) and by is not None:
        ax = sns.violinplot(data=df, x=x, y=y, palette=c.values(), hue=by, 
            ax=ax, saturation=0.7, **params)
        ax.legend([], [], frameon=False)
        ax.set(xlabel='', ylabel='')
        ax.set_xticklabels(np.arange(df[x].unique().size))

    else:
        raise ValueError(f'{by} categories do not match provided colors keys')

    if with_stats:
        add_wilcox(df, x, y, pairs, ax, order=None)

    return ax


##


def plot_heatmap(df, palette='mako', ax=None, title=None, x_names=True, y_names=True, 
    x_names_size=7, y_names_size=7, xlabel=None, ylabel=None, annot=False, annot_size=5, 
    label=None, shrink=1.0, cb=True, vmin=None, vmax=None, rank_diagonal=False, 
    outside_linewidth=1, linewidths=0, linecolor='black'):
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


def stem_plot(df, x, ax=None):
    """
    Create a stem plot
    """
    ax.hlines(y=df.index, xmin=0, xmax=df[x], color='darkgrey')
    ax.plot(df[x][df[x]>=0], df[x].index[df[x]>=0], "o", color='r')
    ax.plot(df[x][df[x]<0], df[x].index[df[x]<0], "o", color='b')
    ax.axvline(color="black", linestyle="--")
    ax.invert_yaxis()
    return ax


##


def bb_plot(df, cov1=None, cov2=None, show_y=True, legend=True, colors=None, 
    ax=None, with_data=False, **kwargs):
    """
    Stacked percentage plot.
    """
    # Prep data
    df[cov1] = pd.Categorical(df[cov1]).remove_unused_categories()
    df[cov2] = pd.Categorical(df[cov2]).remove_unused_categories()
    data = pd.crosstab(df[cov1], df[cov2], normalize='index')
    data_cum = data.cumsum(axis=1)
    ys = data.index.categories
    labels = data.columns.categories

    # Ax
    if colors is None:
        colors = create_palette(df, cov2, palette='tab10')

    for i, x in enumerate(labels):
        widths = data.values[:,i]
        starts = data_cum.values[:,i] - widths
        ax.barh(ys, widths, left=starts, height=0.95, label=x, color=colors[x])

    # Format
    ax.set_xlim(-0.01, 1.01)
    format_ax(
        ax, 
        title = f'{cov1} by {cov2}',
        yticks='' if not show_y else ys, 
        xlabel='Abundance %'
    )

    if legend:
        add_legend(label=cov2, colors=colors, ax=ax, only_top=10, ncols=1,
            loc='upper left', bbox_to_anchor=(1.01,1), ticks_size=7  
        )
    
    if with_data:
        return ax, data
    else:
        return ax
    

##


def rank_plot(df, cov=None, ascending=False, n_annotated=25, title=None, ylabel=None, ax=None, fig=None):
    """
    Annotated scatterplot.
    """
    s = df[cov].sort_values(ascending=ascending)
    x = np.arange(df.shape[0])
    y = s.values
    labels = s[:n_annotated].index
    ax.plot(x, y, '.')
    ta.allocate_text(fig, ax, x[:n_annotated], y[:n_annotated], labels, x_scatter=x, y_scatter=y,
        linecolor='black', textsize=8, max_distance=0.5, linewidth=0.5, nbr_candidates=100)
    format_ax(ax, title=title, xlabel='rank', ylabel=ylabel)

    return ax


##


def bb_plot(df, cov1=None, cov2=None, show_y=True, legend=True, colors=None, 
    ax=None, with_data=False, **kwargs):
    """
    Stacked percentage plot.
    """
    # Prep data
    df[cov1] = pd.Categorical(df[cov1]).remove_unused_categories()
    df[cov2] = pd.Categorical(df[cov2]).remove_unused_categories()
    data = pd.crosstab(df[cov1], df[cov2], normalize='index')
    data_cum = data.cumsum(axis=1)
    ys = data.index.categories
    labels = data.columns.categories

    # Ax
    if colors is None:
        colors = create_palette(df, cov2, palette='tab10')

    for i, x in enumerate(labels):
        widths = data.values[:,i]
        starts = data_cum.values[:,i] - widths
        ax.barh(ys, widths, left=starts, height=0.95, label=x, color=colors[x])

    # Format
    ax.set_xlim(-0.01, 1.01)
    format_ax(
        ax, 
        title = f'{cov1} by {cov2}',
        yticks='' if not show_y else ys, 
        xlabel='Frequency %'
    )

    if legend:
        add_legend(label=cov2, colors=colors, ax=ax, only_top=10, ncols=1,
            loc='upper left', bbox_to_anchor=(1.01,1), ticks_size=7  
        )
    
    if with_data:
        return ax, data
    else:
        return ax


##

        
def packed_circle_plot(
    df, covariate=None, ax=None, color='b', cmap=None, alpha=.5, linewidth=1.2,
    t_cov=.01, annotate=False, fontsize=6, ascending=False, fontcolor='white', 
    fontweight='normal'
    ):

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