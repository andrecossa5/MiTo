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
        c_cont = plu.create_palette(
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