"""
Utils and plotting functions to visualize (clustered and annotated) cells x vars AFM matrices
or cells x cells distances/affinity matrices.
"""

from .plotting_base import *
from .colors import *


##


# Cells x vars AFMs
def cells_vars_heatmap(afm, cmap='magma', cell_anno='GBC', anno_colors=None, heat_label=None, 
    legend_label=None, figsize=(11, 8), title=None, cbar_position=(0.82, 0.2, 0.02, 0.25),
    title_hjust=0.47, legend_bbox_to_anchor=(0.825, 0.5), legend_loc='lower center', 
    legend_ncol=1, xticks_size=5, order='hclust', vmax=0.2):
    """
    Given a (filtered) cells x vars AFM, produce its (clustered, or ordered) 
    annotated heatmap visualization.
    """
    # Order
    if order == 'hclust':
        df_ = pd.DataFrame(data=afm.X, columns=afm.var_names)
        
    elif order == 'diagonal_clones':
            
        df_ = (
            afm.obs
            .join(pd.DataFrame(afm.X, index=afm.obs_names, columns=afm.var_names))
        )
        df_agg = (
            df_
            .drop('sample', axis=1)
            .groupby(cell_anno)
            .mean()
            .reset_index()
            .melt(id_vars=cell_anno)
            .groupby(cell_anno)
            .apply(lambda x: x.sort_values(by='value', ascending=False))
            .reset_index(drop=True)
            .pivot_table(index=cell_anno, values='value', columns='variable')
        )
        var_order = df_agg.mean(axis=0).sort_values()[::-1].index
        clone_order = np.sum(df_agg>0.01, axis=1).sort_values()[::-1].index
        
        df_[cell_anno] = pd.Categorical(
            df_[cell_anno].astype('str'), 
            categories=clone_order, 
            ordered=True
        )
        df_ = df_.sort_values(cell_anno).iloc[:,2:].loc[:, var_order]
        
    elif 'diagonal':
        
        print('not implemented yet')
        
        # df_ = (
        #     afm.obs
        #     .join(pd.DataFrame(afm.X, index=afm.obs_names, columns=afm.var_names))
        # )
        # df_agg = (
        #     df_
        #     .groupby('GBC')
        #     .agg('mean')
        #     .reset_index()
        #     .melt(id_vars='GBC')
        #     .groupby('GBC')
        #     .apply(lambda x: x.sort_values(by='value', ascending=False))
        #     .reset_index(drop=True)
        #     .pivot_table(index='GBC', values='value', columns='variable')
        # )
        # var_order = df_agg.mean(axis=0).sort_values()[::-1].index
        # var_order
        
        
    # Row colors
    row_colors = [ anno_colors[x] for x in afm[df_.index, :].obs[cell_anno] ]
 
    # Plot heatmap
    g = sns.clustermap(
        df_,
        cmap=cmap, 
        row_colors=row_colors,
        yticklabels=False, 
        xticklabels=True, 
        dendrogram_ratio=(.3, .04), 
        colors_ratio=0.05,
        figsize=figsize, 
        row_cluster=True if order == 'hclust' else False, 
        col_cluster=True if order == 'hclust' else False, 
        annot=False, 
        vmin=0,
        vmax=vmax,
        cbar_kws={'use_gridspec':False, 'orientation':'vertical', 'label':heat_label}
    )
    
    g.ax_heatmap.set_xlabel('', loc='left')
    g.ax_heatmap.set_ylabel('', loc='top')
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=xticks_size)
    g.fig.subplots_adjust(right=0.7)
    g.ax_col_dendrogram.set_visible(False) 
    if title is not None:
        g.fig.suptitle(title, x=title_hjust)
    g.ax_cbar.set_position(cbar_position)

    handles = create_handles(anno_colors.keys(), colors=anno_colors.values())
    g.fig.legend(handles, anno_colors.keys(), loc=legend_loc, 
        bbox_to_anchor=legend_bbox_to_anchor, ncol=legend_ncol, 
        frameon=False, title=legend_label
    )

    return g


##


def cell_cell_dists_heatmap(D, cell_anno=None, anno_colors=None, heat_label=None, 
    legend_label=None, figsize=(11, 6.5), title=None, 
    cbar_position=(0.82, 0.2, 0.02, 0.25),
    title_hjust=0.47, legend_bbox_to_anchor=(0.825, 0.5), 
    legend_loc='lower center', 
    legend_ncol=1):
    """
    Plot cell-to-cell similarity matrix.
    """
    df_ = pd.DataFrame(1-D.X, index=D.obs_names, columns=D.obs_names)
    g = sns.clustermap(
        df_, 
        cmap='magma', 
        yticklabels=False, 
        xticklabels=False, 
        dendrogram_ratio=(.3, .04), 
        figsize=figsize, 
        row_cluster=True, 
        col_cluster=True, 
        annot=False, 
        cbar_kws={'use_gridspec' : False, 'orientation' : 'vertical', 'label' : heat_label}, 
        colors_ratio=0.025, row_colors=cell_anno, col_colors=cell_anno
    )
    g.fig.subplots_adjust(right=0.7)
    g.ax_col_dendrogram.set_visible(False) 
    g.fig.suptitle(title, x=title_hjust)
    g.ax_cbar.set_position(cbar_position)
    handles = create_handles(anno_colors.keys(), colors=anno_colors.values())
    g.fig.legend(
        handles, 
        anno_colors.keys(), 
        loc=legend_loc, 
        bbox_to_anchor=legend_bbox_to_anchor, 
        ncol=legend_ncol, 
        frameon=False, 
        title=legend_label
    )

    return g


##

















