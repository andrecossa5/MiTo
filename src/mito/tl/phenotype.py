"""
Tools to map phenotype to lineage structures.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any, Iterable
from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools import score_small_parsimony
from scipy.stats import fisher_exact
from statsmodels.sandbox.stats.multicomp import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf


##


def compute_clonal_fate_bias(
    tree: CassiopeiaTree, 
    state_column: str, 
    clone_column: str, 
    target_state: str|Any
    ) -> pd.DataFrame:
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


def compute_scPlasticity(tree: CassiopeiaTree, meta_column: str):
    """
    Compute scPlasticity as in Yang et al., 2022.
    https://www.sc-best-practices.org/trajectories/lineage_tracing.html#
    """

    # Format column of interest
    tree.cell_meta[meta_column] = pd.Categorical(tree.cell_meta[meta_column])
    # parsimony = score_small_parsimony(tree, meta_item=meta_column)

    # compute plasticities for each node in the tree
    for node in tree.depth_first_traverse_nodes():
        effective_plasticity = score_small_parsimony(
            tree, meta_item=meta_column, root=node
        )
        size_of_subtree = len(tree.leaves_in_subtree(node))
        tree.set_attribute(
            node, "effective_plasticity", effective_plasticity / size_of_subtree
        )

    tree.cell_meta["scPlasticity"] = 0
    for leaf in tree.leaves:
        plasticities = []
        parent = tree.parent(leaf)
        while True:
            plasticities.append(tree.get_attribute(parent, "effective_plasticity"))
            if parent == tree.root:
                break
            parent = tree.parent(parent)

        tree.cell_meta.loc[leaf, "scPlasticity"] = np.mean(plasticities)


##


def nb_regression(df: pd.DataFrame, features: Iterable[str], predictor: str) -> pd.DataFrame:
    """
    Negative binomial regression approach to associate 
    clonal-level features to gene expression.

    Parameters
    ----------
    afm : pd.DataFrame (clone/sample x features/covariates)
        Input data table. Contains raw counts for all genes, and covariates of interest.
    features : list, str
        List of variables to test the GLM model coefficients on.
    predictor : str
        Model specification via formula interface. Formula is in the form:
        "gene ~ predictor". Example predictor: "fitness + counts"

    Returns
    -------
    AnnData
        Filtered Allelic Frequency Matrix.
    """

    L = []
    for gene in tqdm(features, total=len(features), desc="Features"):
        try:
            formula = f'{gene} ~ {predictor}'
            family = sm.families.NegativeBinomial()
            model = (
                smf.glm(formula=formula, data=df, family=family)
                .fit()
            )
            df_ = (
                model.params.to_frame('coef')
                .join(model.pvalues.to_frame('pval'))
                .reset_index(names='param')
                .query('param!="Intercept"')
                .assign(gene=gene)
            )
            L.append(df_)
        except:
            pass

    results = pd.concat(L)
    results['-logp10'] = -np.log10(results['pval'])
    results = results[['gene', 'param', 'coef', 'pval', '-logp10']]

    return results


##