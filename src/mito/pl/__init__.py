# TO DO: assign_matching_colors

from .colors import create_palette, assign_matching_colors, ten_godisnot
from .plotting_base import (
    add_cbar, format_ax, add_legend, set_rcParams
)
from .diagnostic_plots import (
    vars_AF_spectrum, MT_coverage_by_gene_polar, MT_coverage_polar, 
    mut_profile, plot_ncells_nAD
)
from .embeddings_plots import draw_embedding
from .heatmaps_plots import heatmap_distances, heatmap_variants
from .phylo_plots import plot_tree


