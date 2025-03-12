"""
Stores functions to create cusom palettes.
"""

import pandas as pd
import scanpy as sc
import seaborn as sns
import colorsys
import matplotlib.colors
from scipy.optimize import linear_sum_assignment
import numpy as np


##


def _change_color(color, saturation=0.5, lightness=0.5):
    
    r, g, b = color
    h, s, l = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(h, lightness, saturation)
    
    return (r, g, b)


##


def create_palette(df, var, palette=None, col_list=None, 
                saturation=None, lightness=None):
    """
    Create a color palette from a df, a columns, a palette or a list of colors.
    """
    try:
        cats = df[var].cat.categories
    except:
        cats = df[var].unique()
    
    n = len(cats)
    if col_list is not None:
        cols = col_list[:n]
    elif palette is not None:
        cols = sns.color_palette(palette, n_colors=n)
    else:
        raise ValueError('Provide one between palette and col_list!')
    
    colors = { k: v for k, v in zip(cats, cols)}
    
    if saturation is not None:
        colors = { 
            k: _change_color(colors[k], saturation=saturation) \
            for k in colors 
        }
    if lightness is not None:
        colors = { 
            k: _change_color(colors[k], lightness=lightness) \
            for k in colors 
        }
     
    return colors


##


def create_colors(meta, chosen=None):
    """
    Create Cellula 'base' colors: samples, seq run, and optionally leiden categorical covariates.
    """

    # Create a custom dict of colors
    colors = {
        'sample' : create_palette(meta, 'sample', palette='tab20'),
        'seq_run' : create_palette(meta, 'seq_run', palette='tab20')
    }
    
    # Add cluster colors, if needed
    n = len(meta[chosen].cat.categories)
    if chosen is not None:
        if n <= 20:
            c = sc.pl.palettes.default_20[:n]
        else:
            c = sc.pl.palettes.default_102[:n]
        colors[chosen] = { cluster : color for cluster, color in zip(meta[chosen].cat.categories, c)}

    return colors


##


# TO FIX!!
def assign_matching_colors(df, g1, g2, palette):
    """
    Assign colors to categories in g1 and g2, ensuring colors are unique and come from the provided palette.

    Parameters:
    - df: DataFrame with at least two categorical columns, g1 and g2.
    - g1: The column name for the first categorical variable.
    - g2: The column name for the second categorical variable.
    - palette: List of colors to assign to categories.

    Returns:
    - g1_colors: Dictionary mapping categories in g1 to colors.
    - g2_colors: Dictionary mapping categories in g2 to colors.
    """
    # Convert categories to strings
    df[g1] = df[g1].astype(str)
    df[g2] = df[g2].astype(str)

    # Get unique categories
    g1_categories = df[g1].unique()
    g2_categories = df[g2].unique()

    total_categories = len(g1_categories) + len(g2_categories)
    if len(palette) < total_categories:
        raise ValueError(f"Not enough colors in the palette to assign to all categories. Needed: {total_categories}, available: {len(palette)}.")

    # Assign colors to g1 categories
    palette_iter = iter(palette)
    g1_colors = {}
    used_colors = set()
    for g1_cat in g1_categories:
        color = next(palette_iter)
        g1_colors[g1_cat] = color
        used_colors.add(color)

    # Compute the crosstab (contingency table)
    crosstab = pd.crosstab(df[g2], df[g1])

    # Initialize the color assignments for g2
    g2_colors = {}
    for g2_cat in g2_categories:
        counts = crosstab.loc[g2_cat] if g2_cat in crosstab.index else None
        assigned = False
        if counts is not None and counts.sum() > 0:
            # Get the g1 category with the highest count
            g1_cat = counts.idxmax()
            color = g1_colors[g1_cat]
            if color not in g2_colors.values():
                g2_colors[g2_cat] = color
                assigned = True

        if not assigned:
            # Assign next available color from the palette that hasn't been used yet
            while True:
                color = next(palette_iter, None)
                if color is None:
                    raise ValueError("Ran out of colors in the palette.")
                if color not in g2_colors.values() and color not in g1_colors.values():
                    g2_colors[g2_cat] = color
                    break

    return g1_colors, g2_colors
    
        
##

        
# Palettes
ten_godisnot = [
    
    '#001E09', 
    '#885578',
    '#FF913F', 
    '#1CE6FF', 
    '#549E79', 
    '#C9E850', #'#00FECF', 
    '#EEC3FF', 
    '#FFEF00',#'#0000A6', 
    '#D157A0', 
    '#922329'
    
]

ten_godisnot = [ matplotlib.colors.hex2color(x) for x in ten_godisnot ]

