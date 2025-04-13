# TO DO: assign_matching_colors

from .other_plots import packed_circle_plot
from .diagnostic_plots import (
    vars_AF_spectrum, MT_coverage_by_gene_polar, MT_coverage_polar, 
    mut_profile, plot_ncells_nAD
)
from .embeddings_plots import draw_embedding
from .heatmaps_plots import heatmap_distances, heatmap_variants
from .phylo_plots import plot_tree

__all__ = [
    "vars_AF_spectrum", "MT_coverage_by_gene_polar", "MT_coverage_polar",
    "mut_profile", "plot_ncells_nAD", "draw_embedding", "heatmap_distances",
    "heatmap_variants", "plot_tree", "packed_circle_plot"
]

