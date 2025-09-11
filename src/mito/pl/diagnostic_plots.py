"""
Utils and plotting functions to visualize and inspect SNVs from a MAESTER 
experiment and maegatk/mito_preprocessing output.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import plotting_utils as plu
from typing import Dict, Iterable, Tuple, Any
from anndata import AnnData
from matplotlib.ticker import FixedLocator, FuncFormatter
from ..ut.utils import load_mut_spectrum_ref
from ..ut.positions import MAESTER_genes_positions
from ..pp.filters import mask_mt_sites
from ..pp.preprocessing import annotate_vars


## 


def vars_AF_spectrum(
    afm: AnnData, 
    ax: matplotlib.axes.Axes = None, 
    color: str = 'b', 
    **kwargs
    ) ->  matplotlib.axes.Axes:
    """
    Ranked AF distributions (as in Miller et al., 2022).
    """

    X = afm.X.toarray()
    for i in range(X.shape[1]):
        x = X[:,i]
        x = np.sort(x)
        ax.plot(x, '-', color=color, **kwargs)

    plu.format_ax(ax=ax, xlabel='Cells (ranked)', ylabel='Allelic Frequency')

    return ax


##


def plot_ncells_nAD(
    afm: AnnData, 
    ax: matplotlib.axes.Axes = None, 
    title: str = None, 
    xticks: Iterable[Any] = None, 
    yticks: str = None, 
    s: float = 5, 
    color: Any = 'k', 
    alpha: float = .7, 
    **kwargs
    ) ->  matplotlib.axes.Axes:
    """
    Plots similar to the one in Weng et al., 2024, followed by the two commentaries
    from Lareau and Weng, 2024. For each variant, plot the n of positive cells (x-axis)
    vs mean number of AD in positive cells (y-axis).
    """

    annotate_vars(afm, overwrite=True)
    ax.plot(afm.var['Variant_CellN'], afm.var['mean_AD_in_positives'], 'o', c=color, markersize=s, alpha=alpha, **kwargs)
    ax.set_yscale('log', base=2)
    ax.set_xscale('log', base=2)
    xticks = [0,1,2,5,10,20,40,80,160,320,640] if xticks is None else xticks
    yticks = [0,1,2,4,8,16,32,64,132,264] if yticks is None else yticks
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.yaxis.set_major_locator(FixedLocator(yticks))

    def integer_formatter(val, pos):
        return f'{int(val)}'
    
    ax.xaxis.set_major_formatter(FuncFormatter(integer_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(integer_formatter))
    ax.set(xlabel='n +cells', ylabel='n ALT UMI / +cell', title='' if title is None else title)

    return ax


##


def mut_profile(
    mut_list: Iterable[str], 
    figsize: Tuple[float,float] = (6,3),
    legend_kwargs: Dict[str,Any] = {}
    ) ->  matplotlib.figure.Figure:
    """
    Re-implementation of MutationProfile_bulk, from Weng et al., 2024).
    """

    ref_df = load_mut_spectrum_ref()
    called_variants = [ ''.join(x.split('_')) for x in mut_list ]
        
    ref_df['called'] = ref_df['variant'].isin(called_variants)
    total = len(ref_df)
    total_called = ref_df['called'].sum()

    grouped = ref_df.groupby(['three_plot', 'group_change', 'strand'])
    prop_df = grouped.agg(
        observed_prop_called=('called', lambda x: x.sum() / total_called),
        expected_prop=('variant', lambda x: x.count() / total),
        n_obs=('called', 'sum'),
        n_total=('variant', 'count')
    ).reset_index()

    prop_df['fc_called'] = prop_df['observed_prop_called'] / prop_df['expected_prop']
    prop_df = prop_df.set_index('three_plot')
    prop_df['group_change'] = prop_df['group_change'].map(lambda x: '>'.join(list(x)))

    n = prop_df['group_change'].unique().size
    fig, axs = plt.subplots(
        1, n, figsize=figsize, sharey=True, 
        gridspec_kw={'wspace': 0.1}, constrained_layout=True
    )
    strand_palette = {'H': '#05A8B3', 'L': '#D76706'}

    for i,x in enumerate(prop_df['group_change'].unique()):
        ax = axs.ravel()[i]
        df_ = prop_df.query('group_change==@x')
        for strand in df_['strand'].unique():
            plu.bar(
                df_.query('strand==@strand').reset_index(), 
                x='three_plot',
                y='n_obs',
                color=strand_palette[strand], 
                categorical_cmap = None,
                width=1, alpha=.5, edgecolor=None, 
                with_label=False,
                ax=ax
            )
        plu.format_ax(
            ax, xticks=[], xlabel=x, 
            ylabel='Substitution rate' if i==0 else '', 
            title=f'n: {df_["n_obs"].sum()}'
        )

    plu.add_legend(
        ax=axs.ravel()[0], colors=strand_palette, ncols=1, 
        loc='upper left', bbox_to_anchor=(0,1), label='Strand', 
        **legend_kwargs
    )
    fig.tight_layout()

    return fig


##


def MT_coverage_polar(
    cov: pd.DataFrame, 
    var_subset: Iterable[str] = None, 
    ax: matplotlib.axes.Axes = None, 
    n_xticks: int = 6, 
    xticks_size: float = 7, 
    yticks_size: float = 2,
    xlabel_size: float = 6, 
    ylabel_size: float = 9, 
    kwargs_main: Dict[str,Any] = {}, 
    kwargs_subset: Dict[str,Any] = {}
    ) ->  matplotlib.axes.Axes:
    """
    Plot coverage and muts across MT-genome positions.
    """
    
    kwargs_main_ = {'c':'#494444', 'linestyle':'-', 'linewidth':.7}
    kwargs_subset_ = {'c':'r', 'marker':'+', 'markersize':10, 'linestyle':''}
    kwargs_main_.update(kwargs_main)
    kwargs_subset_.update(kwargs_subset)

    x = cov.mean(axis=0)

    theta = np.linspace(0, 2*np.pi, len(x))
    ticks = [ 
        int(round(x)) \
        for x in np.linspace(1, cov.shape[1], n_xticks) 
    ][:7]

    ax.plot(theta, np.log10(x), **kwargs_main_)

    if var_subset is not None:
        var_pos = var_subset.map(lambda x: int(x.split('_')[0]))
        test = x.index.isin(var_pos)
        print(test.sum())
        ax.plot(theta[test], np.log10(x[test]), **kwargs_subset_)

    ax.set_theta_offset(np.pi/2)
    ax.set_xticks(np.linspace(0, 2*np.pi, n_xticks-1, endpoint=False))#, fontsize=1)
    ax.set_xticklabels(ticks[:-1], fontsize=xticks_size)

    ax.set_yticklabels([])
    for tick in np.arange(-1,4,1):
        ax.text(0, tick, str(tick), ha='center', va='center', fontsize=yticks_size)

    ax.text(0, 1.5, 'n UMIs', ha='center', va='center', fontsize=xlabel_size, color='black')
    ax.text(np.pi, 4, 'Position (bp)', ha='center', va='center', fontsize=ylabel_size, color='black')

    ax.spines['polar'].set_visible(False)

    return ax



##


def MT_coverage_by_gene_polar(
    cov: pd.DataFrame, 
    sample: str = None, 
    subset: Iterable[str] = None, 
    ax: matplotlib.axes.Axes = None
    ) ->  matplotlib.axes.Axes:
    """
    Plot coverage and muts across MT-genome positions, with annotated genes.
    """
    
    if subset is not None:
        cov = cov.query('cell in @subset')
    cov['pos'] = pd.Categorical(cov['pos'], categories=range(1,16569+1))
    cov = cov.pivot_table(index='cell', columns='pos', values='n', dropna=False, fill_value=0)
    df_mt = (
        pd.DataFrame(
            MAESTER_genes_positions, 
            columns=['gene', 'start', 'end']
        )
        .set_index('gene')
        .sort_values('start')   
    )

    x = cov.mean(axis=0)
    median_target = cov.loc[:,mask_mt_sites(cov.columns)].median(axis=0).median()
    median_untarget = cov.loc[:,~mask_mt_sites(cov.columns)].median(axis=0).median()
    theta = np.linspace(0, 2*np.pi, cov.shape[1])
    colors = { k:v for k,v in zip(df_mt.index, sc.pl.palettes.default_102[:df_mt.shape[0]])}
    ax.plot(theta, np.log10(x), '-', linewidth=.7, color='grey')
    idx = np.arange(1,x.size+1)

    for gene in colors:
        start, stop = df_mt.loc[gene, ['start', 'end']].values
        test = (idx>=start) & (idx<=stop)
        ax.plot(theta[test], np.log10(x[test]), color=colors[gene], linewidth=1.5)

    ticks = [ int(round(x)) for x in np.linspace(1, cov.shape[1], 8) ][:7]
    ax.set_theta_offset(np.pi/2)
    ax.set_xticks(np.linspace(0, 2*np.pi, 7, endpoint=False))
    ax.set_xticklabels(ticks)
    ax.xaxis.set_tick_params(labelsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_rlabel_position(0) 
    ax.set(xlabel='Position (bp)', title=f'{sample}\nTarget: {median_target:.2f}, untarget: {median_untarget:.2f}')

    return ax


##