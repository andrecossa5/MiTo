"""
Other plots (i.e., packed_circle_plot)
"""

import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotting_utils as plu
from typing import Dict, Any, Iterable
from circlify import circlify, Circle


##

        
def packed_circle_plot(
    df: pd.DataFrame,
    ax: matplotlib.axes.Axes = None,
    covariate: str = None, 
    color: str = None, 
    cmap: Dict[str,Any] = None,
    color_by: str = None,  
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
    
    # Colors
    if isinstance(color, str) and not color in df.columns:
        colors = { k : color for k in df.index }
    elif isinstance(cmap, str) and color_by in df.columns:
        colors = plu.create_palette(df, color_by, cmap, order=df.index[::-1]) # Assumes index contains IDs
    elif isinstance(cmap, dict):
        colors = cmap
    else:
        raise TypeError('Check cmap and color arguments...')

    # Plot circles
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