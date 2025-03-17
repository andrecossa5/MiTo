"""
Utils and plotting functions to visualize and inspect SNVs from a MAESTER experiment and maegatk output.
"""

import numpy as np
import scanpy as sc
from matplotlib.ticker import FixedLocator, FuncFormatter
from .plotting_base import *
from ..ut.utils import load_mut_spectrum_ref
from ..ut.positions import MAESTER_genes_positions
from ..pp.filters import mask_mt_sites
from ..pp.preprocessing import annotate_vars


## 


# Current diagnosti plots
def vars_AF_spectrum(afm, ax=None, color='b', **kwargs):
    """
    Ranked AF distributions (VG-like).
    """
    X = afm.X.A
    for i in range(X.shape[1]):
        x = X[:,i]
        x = np.sort(x)
        ax.plot(x, '-', color=color, **kwargs)

    format_ax(ax=ax, xlabel='Cells (ranked)', ylabel='Allelic Frequency (AF)')

    return ax


##


def plot_ncells_nAD(afm, ax=None, title=None, xticks=None, yticks=None, s=5, c='k', alpha=.7, **kwargs):
    """
    Plots similar to Weng et al., 2024, followed by the two commentaries from Lareau and Weng.
    n+ cells vs n 
    """

    annotate_vars(afm, overwrite=True)
    ax.plot(afm.var['Variant_CellN'], afm.var['mean_AD_in_positives'], 'o', c=c, markersize=s, alpha=alpha, **kwargs)
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
    ax.set(xlabel='n +cells', ylabel='Mean n ALT UMI / +cell', title='' if title is None else title)

    return ax


##


def mut_profile(mut_list, figsize=(6,3)):
    """
    MutationProfile_bulk (Weng et al., 2024).
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


    fig, axs = plt.subplots(1, prop_df['group_change'].unique().size, figsize=figsize, sharey=True, gridspec_kw={'wspace': 0.1},
                            constrained_layout=True)
    strand_palette = {'H': '#05A8B3', 'L': '#D76706'}

    for i,x in enumerate(prop_df['group_change'].unique()):
        ax = axs.ravel()[i]
        df_ = prop_df.query('group_change==@x')
        bar(df_, 'n_obs', by='strand', c=strand_palette, ax=ax, s=1, a=.8, annot=False)
        format_ax(ax, xticks=[], xlabel=x, ylabel='Substitution rate' if i==0 else '', title=f'n: {df_["n_obs"].sum()}')

    add_legend(ax=axs.ravel()[0], colors=strand_palette, ncols=1, loc='upper left', bbox_to_anchor=(0,1), label='Strand', ticks_size=6)
    fig.tight_layout()

    return fig


##


def MT_coverage_polar(df, var_subset=None, ax=None, n_xticks=6, xticks_size=7, 
                    yticks_size=2, xlabel_size=6, ylabel_size=9, kwargs_main={}, kwargs_subset={},):
    
    """
    Plot coverage and muts across postions.
    """
    
    kwargs_main_ = {'c':'#494444', 'linestyle':'-', 'linewidth':.7}
    kwargs_subset_ = {'c':'r', 'marker':'+', 'markersize':10, 'linestyle':''}
    kwargs_main_.update(kwargs_main)
    kwargs_subset_.update(kwargs_subset)

    x = df.mean(axis=0)

    theta = np.linspace(0, 2*np.pi, len(x))
    ticks = [ 
        int(round(x)) \
        for x in np.linspace(1, df.shape[1], n_xticks) 
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


def MT_coverage_by_gene_polar(cov, sample=None, subset=None, ax=None):
    """
    MT coverage, with annotated genes, in polar coordinates.
    """
    
    if subset is not None:
        cov = cov.query('cell in @subset')
    cov['pos'] = pd.Categorical(cov['pos'], categories=range(1,16569+1))
    cov = cov.pivot_table(index='cell', columns='pos', values='n', dropna=False, fill_value=0)

    df_mt = pd.DataFrame(MAESTER_genes_positions, columns=['gene', 'start', 'end']).set_index('gene').sort_values('start')

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