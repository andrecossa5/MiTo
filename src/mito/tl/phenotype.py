"""
Tools to map phenotype to lineage structures.
"""

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.sandbox.stats.multicomp import multipletests


##


def compute_clonal_fate_bias(tree, state_column, clone_column, target_state):
    """
    Compute -log10(FDR) Fisher's exact test: clonal fate biases towards some target_state.
    """

    n = len(tree.leaves)
    clones = np.sort(tree.cell_meta[clone_column].unique())

    target_ratio_array = np.zeros(clones.size)
    oddsratio_array = np.zeros(clones.size)
    pvals = np.zeros(clones.size)

    # Here we go
    for i, clone in enumerate(clones):

        test_clone = tree.cell_meta[clone_column] == clone
        test_state = tree.cell_meta[state_column] == target_state

        clone_size = test_clone.sum()
        clone_state_size = (test_clone & test_state).sum()
        target_ratio = clone_state_size / clone_size
        target_ratio_array[i] = target_ratio
        other_clones_state_size = (~test_clone & test_state).sum()

        # Fisher
        oddsratio, pvalue = fisher_exact(
            [
                [clone_state_size, clone_size - clone_state_size],
                [other_clones_state_size, n - other_clones_state_size],
            ],
            alternative='greater',
        )
        oddsratio_array[i] = oddsratio
        pvals[i] = pvalue

    # Correct pvals --> FDR
    pvals = multipletests(pvals, alpha=0.05, method="fdr_bh")[1]

    # Results
    results = pd.DataFrame({
        'perc_in_target_state' : target_ratio_array,
        'odds_ratio' : oddsratio_array,
        'FDR' : pvals,
        'fate_bias' : -np.log10(pvals) 
    }).sort_values('fate_bias', ascending=False)

    return results


##


