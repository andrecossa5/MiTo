"""
Metrics.
"""

from joblib import cpu_count, parallel_backend, Parallel, delayed
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Tuple, Any
from scipy.stats import chi2
from scipy.special import binom
from sklearn.metrics import (
    normalized_mutual_info_score, recall_score, precision_score, auc
)
from networkx import shortest_path_length
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import pairwise_distances
from cassiopeia.data import CassiopeiaTree
from .utils import rescale


##


def chunker(n):
    """
    Create an np.array of starting indeces for parallel computation.
    """
    n_jobs = cpu_count()
    starting_indeces = np.zeros(n_jobs + 1, dtype=int)
    quotient = n // n_jobs
    remainder = n % n_jobs

    for i in range(n_jobs):
        starting_indeces[i+1] = starting_indeces[i] + quotient + (1 if i < remainder else 0)

    return starting_indeces


##


def kbet_one_chunk(index, batch, null_dist):
    """
    kBET calculation for a single index chunk.
    """
    dof = null_dist.size-1
    n = index.shape[0]
    k = index.shape[1]-1
    results = np.zeros((n, 2))

    for i in range(n):
        observed_counts = (
            pd.Series(batch[index[i,:]]).value_counts(sort=False).values
        )
        expected_counts = null_dist * k
        stat = np.sum(
            np.divide(
            np.square(np.subtract(observed_counts, expected_counts)),
                expected_counts,
            )
        )
        p_value = 1 - chi2.cdf(stat, dof)
        results[i, 0] = stat
        results[i, 1] = p_value

    return results


##


def kbet(
    index: np.array, 
    batch: pd.Series, 
    alpha: float = 0.05, 
    only_score: bool = True
    ) -> Tuple[float, float, float]:
    """
    Computes the kBET metric (Buttner et al., 2018) to assess batch effects for an index matrix of a KNN graph.

    Parameters
    ----------
    index : np.array
        Array of shape (n_cells, n_neighbors) containing kNN indices.
    batch : pd.Series
        Discrete-valued batch annotation for each cell (length n_cells).
    alpha : float, optional
        Significance level of the chi-squared test. Default is 0.05.
    only_score : bool, optional
        If True, return only the accept rate; otherwise, return full kBET results. Default is True.

    Returns
    -------
    tuple of (stat_mean, pvalue_mean, accept_rate)
        kBET statistics, where:
        - stat_mean is the mean test statistic,
        - pvalue_mean is the mean p-value,
        - accept_rate is the overall acceptance rate.
    """
 
    # Compute null batch distribution
    batch = batch.astype('category')
    null_dist = batch.value_counts(normalize=True, sort=False).values 

    # Parallel computation of kBET metric (pegasus code)
    starting_idx = chunker(len(batch))
    n_jobs = cpu_count()

    with parallel_backend("loky", inner_max_num_threads=1):
        kBET_arr = np.concatenate(
            Parallel(n_jobs=n_jobs)(
                delayed(kbet_one_chunk)(
                    index[starting_idx[i] : starting_idx[i + 1], :], 
                    batch, 
                    null_dist
                )
                for i in range(n_jobs)
            )
        )
        
    # Gather results 
    stat_mean, pvalue_mean = kBET_arr.mean(axis=0)
    accept_rate = (kBET_arr[:,1] >= alpha).sum() / len(batch)

    if only_score:
        return accept_rate
    else:
        return (stat_mean, pvalue_mean, accept_rate)


##


def NN_entropy(index: np.array, labels: np.array) -> float:
    """
    Calculate the median (over cells) lentiviral-labels Shannon Entropy,
    given an index matrix of a KNN graph.

    Parameters
    ----------
    index : np.array
        Array of shape (n_cells, k-neighbors) containing cell neighbors indeces.
    labels : pd.Series
        Discrete-valued batch annotation for each cell (length n_cells).

    Returns
    -------
    float : NN Shannon Entropy score.
    """

    SH = []
    for i in range(index.shape[0]):
        freqs = labels[index[i,:]].value_counts(normalize=True).values
        SH.append(-np.sum(freqs * np.log(freqs + 0.00001))) # Avoid 0 division
    
    return np.median(SH)


##


def NN_purity(index: np.array, labels: np.array) -> float:
    """
    Calculate the median purity of cells neighborhoods.

    Parameters
    ----------
    index : np.array
        Array of shape (n_cells, k-neighbors) containing cell neighbors indeces.
    labels : pd.Series
        Discrete-valued batch annotation for each cell (length n_cells).

    Returns
    -------
    float : NN purity score.

    """

    kNN_purities = []
    n_cells = index.shape[0]
    k = index.shape[1]-1

    for i in range(n_cells):
        l = labels[i]
        idx = index[i, 1:]
        l_neighbors = labels[idx]
        kNN_purities.append(np.sum(l_neighbors == l) / k)
    
    return np.median(kNN_purities)


##


def binom_sum(x, k=2):
    return binom(x, k).sum()


##


def custom_ARI(g1: Iterable[Any], g2: Iterable[Any]) -> float:
    """
    Compute scIB (Luecken et al., 2022) modified Adjusted Rand Index.
    """

    # Contingency table
    n = len(g1)
    contingency = pd.crosstab(g1, g2)

    # Calculate and rescale ARI
    ai_sum = binom_sum(contingency.sum(axis=0))
    bi_sum = binom_sum(contingency.sum(axis=1))
    index = binom_sum(np.ravel(contingency))
    expected_index = ai_sum * bi_sum / binom_sum(n, 2)
    max_index = 0.5 * (ai_sum + bi_sum)

    return (index - expected_index) / (max_index - expected_index)


##


def distance_AUPRC(D: np.array, labels: Iterable[Any]) -> float:
    """
    Uses a n x n distance matrix D as a binary classifier for a set of labels  (1,...,n). 
    Reports Area Under Precision Recall Curve. Used in Ludwig et al., 2019.
   
    Parameters
    ----------
    D : np.array
        Array of shape (n_cells, n_cells) containing cell-cell distances.
    labels : pd.Series
        Discrete-valued batch annotation for each cell (length n_cells).

    Returns
    -------
    float : AUPRC score.
    """    

    labels = pd.Categorical(labels) 

    final = {}
    for alpha in np.linspace(0,1,10):
 
        p_list = []
        gt_list = []

        for i in range(D.shape[0]):
            x = rescale(D[i,:])
            p_list.append(np.where(x<=alpha, 1, 0))
            c = labels.codes[i]
            gt_list.append(np.where(labels.codes==c, 1, 0))

        predicted = np.concatenate(p_list)
        gt = np.concatenate(gt_list)
        p = precision_score(gt, predicted)
        r = recall_score(gt, predicted)

        final[alpha] = (p, r)

    df = pd.DataFrame(final).T.reset_index(drop=True)
    df.columns = ['precision', 'recall']
    auc_score = auc(df['recall'], df['precision'])

    return auc_score


##


def calculate_corr_distances(tree: CassiopeiaTree) -> float:
    """
    Calculate correlation between tree and character matrix cell-cell distances. 
    Used in Yang et al., 2023.
    """

    if tree.get_dissimilarity_map() is not None:
        D = tree.get_dissimilarity_map()
        D = D.loc[tree.leaves, tree.leaves] # In case root is there...
    else:
        raise ValueError('No precomputed character distance. Add one...')
    
    L = []
    undirected = tree.get_tree_topology().to_undirected()
    for node in tree.leaves:
        d = shortest_path_length(undirected, source=node)
        L.append(d)
    D_phylo = pd.DataFrame(L, index=tree.leaves).loc[tree.leaves, tree.leaves]
    assert (D_phylo.index == D.index).all()

    scale = lambda x: (x-x.mean())/x.std()
    x = scale(D.values.flatten())
    y = scale(D_phylo.values.flatten())
    corr, p = pearsonr(x, y)
    
    return corr, p


##


def _compatibility_metric(x, y):
    """
    Custom metric to calculate the compatibility between two characters.
    Returns the fraction of compatible leaf pairs.
    """
    return np.sum((x == x[:, None]) == (y == y[:, None])) / len(x) ** 2

##


def char_compatibility(tree):
    """
    Compute a matrix of pairwise-compatibility scores between characters.
    """
    return pairwise_distances(
        tree.character_matrix.T, 
        metric=lambda x, y: _compatibility_metric(x, y), 
        force_all_finite=False
    )


##


def CI(tree: CassiopeiaTree) -> float:
    """
    Calculate the Consistency Index (CI) of tree characters.
    """
    tree.reconstruct_ancestral_characters()
    observed_changes = np.zeros(tree.n_character)
    for parent, child in tree.depth_first_traverse_edges():
        p_states = np.array(tree.get_character_states(parent))
        c_states = np.array(tree.get_character_states(child))
        changes = (p_states != c_states).astype(int)
        observed_changes += changes

    return 1 / observed_changes # Assumes both characters are present (1,0)


##


def RI(tree: CassiopeiaTree) -> float:
    """
    Calculate the Consistency Index (RI) of tree characters.
    """
    tree.reconstruct_ancestral_characters()
    observed_changes = np.zeros(tree.n_character)
    for parent, child in tree.depth_first_traverse_edges():
        p_states = np.array(tree.get_character_states(parent))
        c_states = np.array(tree.get_character_states(child))
        changes = (p_states != c_states).astype(int)
        observed_changes += changes

    # Calculate the maximum number of changes (G)
    max_changes = len(tree.nodes)-1  # If every node had a unique state

    return (max_changes-observed_changes) / (max_changes-1)


##