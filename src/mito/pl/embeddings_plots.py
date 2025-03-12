"""
Stores plotting functions for embeddings.
"""

import scanpy as sc
import matplotlib.pyplot as plt
from .colors import *
from .plotting_base import *


##


def format_draw_embeddings(
    ax, df, x, y, title=None, cat=None, cont=None, axes_params={}):
    """
    Utils to format a draw embeddings plots.
    """
    legend_params = axes_params['legend_params']
    cbar_params = axes_params['cbar_params']

    not_format_keys = ['only_labels', 'no_axis', 'legend', 'cbar', 'legend_params', 'cbar_params']
    format_kwargs = { k : axes_params[k] for k in axes_params if k not in not_format_keys }

    if title is None:
        title = cat if cat is not None else cont 
    else:
        assert isinstance(title, str)

    format_ax(ax=ax, xlabel=x, ylabel=y, title=title, **format_kwargs)

    if axes_params['only_labels']:
        remove_ticks(ax)
    if not axes_params['axis']:
        ax.axis('off')
    
    if axes_params['legend'] and cat is not None:
        if 'label' not in legend_params or legend_params['label'] is None:
            legend_params['label'] = cat
        add_legend(ax=ax, **legend_params)
    
    elif axes_params['cbar'] and cont is not None:
        add_cbar(df[cont], ax=ax, **cbar_params)

    return ax


##


def handle_colors(df, cat, legend_params, query=None):
    """
    Util to handle colors in draw_embeddings.
    """
    if query is not None:
        df_ = df.query(query)
    else:
        df_ = df

    try:
        categories = df_[cat].cat.categories
    except:
        categories = df_[cat].unique()
        
    if categories.size <=20 and legend_params['colors'] is None:
        palette_cat = sc.pl.palettes.vega_20_scanpy
        legend_params['colors'] = create_palette(df_, cat, palette_cat)
    elif categories.size > 20 and legend_params['colors'] is None:
        palette_cat = sc.pl.palettes.godsnot_102
        legend_params['colors'] = create_palette(df_, cat, palette_cat)
    elif isinstance(legend_params['colors'], dict):
        print(categories)
        print(legend_params['colors'].keys())
        assert all([ cat in legend_params['colors'] for cat in categories ])
        print('Provided colors are OK...')
    else:
        raise ValueError('Provide a correctly formatted palette for your categorical labels!')

    return legend_params


##


# TO FIX!!
def draw_embeddings(
    df, x='UMAP1', y='UMAP2', cat=None, cont=None, ax=None, s=None, query=None, title=None,
    cbar_kwargs={}, legend_kwargs={}, axes_kwargs={}):
    """
    Draw covariates on embeddings plot.
    """

    cbar_params={
        'palette' : 'viridis',
        'vmin': None,
        'vmax':None,
        'label_size' : 8, 
        'ticks_size' : 6,  
        'layout' : 'outside',
        'label' : None
    }

    legend_params={
        'bbox_to_anchor' : (1,1),
        'loc' : 'upper right', 
        'label_size' : 10,
        'ticks_size' : 8,
        'colors' : None,
        'ncols' : 1
    }

    axes_params = {
        'only_labels' : True,
        'axis' : False, 
        'legend' : True,
        'cbar' : True,
        'title_size' : 10
    }

    cbar_params = update_params(cbar_params, cbar_kwargs)
    legend_params = update_params(legend_params, legend_kwargs)
    axes_params = update_params(axes_params, axes_kwargs)
    axes_params['cbar_params'] = cbar_params
    axes_params['legend_params'] = legend_params

    if s is None:
        s = 12000 / df.shape[0] # as in scanpy

    if cat is not None and cont is None:

        legend_params = handle_colors(df, cat, legend_params, query=query)

        if query is None:
            scatter(df, x=x, y=y, by=cat, c=legend_params['colors'], ax=ax, s=s)
            format_draw_embeddings(ax, df, x, y, title=title,
                cat=cat, cont=None, axes_params=axes_params
            )
        else:
            if isinstance(query, str):
                subset = df.query(query).index
            else:
                subset = query
            if subset.size > 0:
                legend_params['colors'] = {**legend_params['colors'], **{'others':'darkgrey'}}
                scatter(df.loc[~df.index.isin(subset), :], x=x, y=y, c='darkgrey', ax=ax, s=s/3)
                scatter(df.loc[subset, :], x=x, y=y, by=cat, c=legend_params['colors'], ax=ax, s=s)
                format_draw_embeddings(ax, df.loc[subset, :], x, y, title=title,
                    cat=cat, cont=None, axes_params=axes_params
                )
            else:
                raise ValueError('The queried subset has no obs...')
    
    elif cat is None and cont is not None:
        
        if query is None:
            scatter(df, x=x, y=y, by=cont, c=cbar_params['palette'], 
                    vmin=cbar_params['vmin'], vmax=cbar_params['vmax'], ax=ax, s=s)
            format_draw_embeddings(ax, df, x, y, title=title, cont=cont, axes_params=axes_params)
        else:
            if isinstance(query, str):
                subset = df.query(query).index
            else:
                subset = query
            if subset.size > 0:
                scatter(df.loc[~df.index.isin(subset), :], x=x, y=y, c='darkgrey', ax=ax, s=s/3)
                scatter(df.loc[subset, :], x=x, y=y, by=cont, c=cbar_params['palette'], 
                        vmin=cbar_params['vmin'], vmax=cbar_params['vmax'], ax=ax, s=s)
                format_draw_embeddings(
                    ax, df.loc[subset, :], x, y, title=title, cont=cont, axes_params=axes_params
                )
            else:
                raise ValueError('The queried subset has no obs available...')

    else:
        raise ValueError('Specifiy either a categorical or a continuous covariate for plotting.')
    
    ax.axis('off')

    return ax


##


def faceted_draw_embedding(
    df, x='UMAP1', y='UMAP2', figsize=None, n_cols=None,
    cont=None, cat=None, query=None, facet=None, legend=True, **kwargs):
    """
    Draw embeddings with faceting.
    """
    fig = plt.figure(figsize=figsize)

    n_axes, idxs, names = find_n_axes(df, facet, query=query)

    if n_axes == 0:
        raise ValueError('No subsets to plot...')

    n_rows, n_cols = find_n_rows_n_cols(n_axes, n_cols=n_cols)

    for i, (idx, name) in enumerate(zip(idxs, names)):
        
        if legend:
            draw_legend = True if i == 0 else False
            draw_cbar = True if i == 0 else False
        else:
            draw_legend = False
            draw_cbar = True if i == 0 else False

        ax = plt.subplot(n_rows, n_cols, i+1)
        draw_embeddings(
            df.loc[idx, :], 
            x=x, y=y,
            cont=cont, 
            cat=cat, 
            ax=ax, 
            query=query,
            axes_kwargs={'legend' : draw_legend, 'cbar' : draw_cbar},
            **kwargs
        )
        format_ax(ax, title=name)

    fig.supxlabel(x)
    fig.supylabel(y)
    fig.tight_layout()

    return fig


##