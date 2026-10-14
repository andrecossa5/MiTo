"""
Clade support from count resampling: how often a clade of the tree is rebuilt when the
molecule counts are drawn again.
"""

import logging

import numpy as np
import pandas as pd
from anndata import AnnData
from cassiopeia.data import CassiopeiaTree
from scipy import stats

from mito.pp.distances import weighted_jaccard
from mito.pp.genotyping import _counts, _error_rates
from mito.ut.provenance import record
from mito.ut.utils import Timer

from .phylo import upgma_linkage

##


def _clade_hashes_tree(tree, keys):
    """Hash of every internal node's leaf set: the sum of its leaves' random keys."""

    h = {}
    for node in tree.depth_first_traverse_nodes(postorder=True):
        if tree.is_leaf(node):
            h[node] = keys[node]
        else:
            s = np.uint64(0)
            for child in tree.children(node):
                s = s + h[child]
            h[node] = s
    return {node: h[node] for node in tree.internal_nodes}


def _clade_hashes_linkage(Z, keys):
    """Hashes of the n - 1 clades of a linkage matrix, leaves keyed by row."""

    n = keys.size
    h = np.empty(2*n-1, dtype=np.uint64)
    h[:n] = keys
    a = Z[:,0].astype(np.int64)
    b = Z[:,1].astype(np.int64)
    for i in range(n-1):
        h[n+i] = h[a[i]] + h[b[i]]
    return h[n:]


def _transfer_scores(Z, members, sizes):
    """
    Transfer support of reference clades in one replicate tree (Lemoine et al., 2018).

    The transfer distance of clade S to the replicate is the fewest cells to move to turn
    S into one of its clades: min over replicate clades R of |S| + |R| - 2 |S & R|. The
    overlaps are accumulated bottom-up along the merges of the linkage, for all reference
    clades at once. Support = 1 - distance / (|S| - 1).

    members: (n_leaves, n_ref) int32 membership of the cells in the reference clades.
    """

    n = members.shape[0]
    a = Z[:,0].astype(np.int64)
    b = Z[:,1].astype(np.int64)
    overlap = np.empty((2*n-1, members.shape[1]), dtype=np.int32)
    overlap[:n] = members
    size = np.ones(2*n-1, dtype=np.int64)
    best = members.max(0).astype(np.int64)*2 - 1           # single cells: 2*1 - 1
    for i in range(n-1):
        overlap[n+i] = overlap[a[i]] + overlap[b[i]]
        size[n+i] = size[a[i]] + size[b[i]]
        np.maximum(best, 2*overlap[n+i] - size[n+i], out=best)
    dist = sizes - best

    return 1 - dist / np.maximum(sizes-1, 1)


def _replicate_calls(AD, COV, p, eps, alpha, rng):
    """
    One count replicate: every cell's alternative molecules drawn again at its observed
    allele frequency (carriers) or at the variant's error rate (the others), then called
    with the binomial test of `mito.pp.call_genotypes`, error rates held fixed.
    """

    AD_b = rng.binomial(COV, p)
    B = np.zeros(AD.shape, bool)
    pos = AD_b>0
    r, c = np.nonzero(pos)
    B[r, c] = stats.binom.sf(AD_b[r, c]-1, COV[r, c], eps[c]) <= alpha

    return AD_b, B


def _distances(AD_b, COV, B):
    """Weighted Jaccard distances of a replicate, weights as in `mito.pp.compute_distances`."""

    af = AD_b / np.maximum(COV, 1)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            w = np.nanmedian(np.where(B & (af>0), af, np.nan), axis=0)
    bad = ~np.isfinite(w)
    if bad.any():
        fb = np.nanmedian(w[~bad]) if (~bad).any() else 1.0
        w = np.where(bad, fb if np.isfinite(fb) else 1.0, w)

    return weighted_jaccard(B.astype(float), w)


##


def clade_support(
    tree: CassiopeiaTree,
    afm: AnnData,
    n_boot: int = 100,
    alpha: float = None,
    method: str = 'transfer',
    seed: int = 0,
    key_added: str = 'support'
    ) -> pd.Series:
    """
    Support of every clade of a UPGMA tree under count resampling.

    MT-SNV trees are built from a few hundred characters, most of them private to one
    clone, so resampling characters (the phylogenetic bootstrap) mostly measures how many
    markers a clade has: a clone marked by one variant can never be supported. What is
    uncertain in these data is the counts - a cell called on one or two molecules may not
    be a carrier at all. So each replicate draws every cell's alternative molecules
    again, at its observed allele frequency if it has any and at the variant's error
    rate if it has none, re-calls the genotypes with the test of
    `mito.pp.call_genotypes` (error rates held at their fitted values), and rebuilds the
    UPGMA tree from the weighted Jaccard distances.

    A clade's support is averaged over the replicates, as

    * "transfer" (default): 1 - (cells to move to make it a clade of the replicate) /
      (its size - 1), the transfer bootstrap expectation (Lemoine et al., 2018). A clone of
      300 cells that loses three weakly covered cells in a replicate scores 0.99 there,
      not 0; a clade of two cells still has to be recovered exactly.
    * "exact": the fraction of replicates in which exactly its set of cells is a clade
      (Felsenstein's bootstrap). Large clades rarely reach it, since some cell always
      moves when its counts are drawn again.

    Each replicate costs one call, one distance matrix and one average-linkage tree, so
    100 replicates of a few thousand cells take seconds to a minute.

    Parameters
    ----------
    tree : CassiopeiaTree
        Tree from `mito.tl.build_tree` (UPGMA) on `afm`.
    afm : AnnData
        The filtered AFM the tree was built from, with AD / DP layers and genotypes.
    n_boot : int, optional
        Number of count replicates. Default is 100.
    method : str, optional
        "transfer" or "exact". Default is "transfer".
    alpha : float, optional
        Significance of the replicate calls. None (default) uses the one `afm` was
        genotyped with (`mito.pp.filter_afm`), or 1e-3.
    seed : int, optional
        Random seed. Default is 0.
    key_added : str, optional
        Node attribute the support is written to. Default is "support".

    Returns
    -------
    pd.Series
        Support of every internal node, indexed by node name. Also written to the tree
        (node attribute `key_added`; the root has support 1) and recorded in
        .uns["mito"]["clade_support"] of `afm`.
    """

    T = Timer()
    T.start()
    cells = list(afm.obs_names)
    if set(cells) != set(tree.leaves):
        raise ValueError('The tree leaves and the AFM cells differ: pass the AFM the tree was built from.')
    if 'bin' not in afm.layers:
        raise ValueError('clade_support needs genotypes in afm.layers["bin"]: run mito.pp.filter_afm.')

    if alpha is None:
        alpha = (afm.uns.get('mito', {}).get('filter_afm', {}).get('params', {})
                 .get('genotyping', {}).get('alpha', 1e-3))

    AD, COV = _counts(afm)
    carrier = afm.layers['bin'].toarray()>0
    eps = afm.var['error_rate'].values if 'error_rate' in afm.var else _error_rates(AD, COV, carrier)
    eps = np.clip(eps, 1e-12, 1-1e-12)
    p = np.where(AD>0, AD/np.maximum(COV, 1), eps[None,:])

    if method not in ('transfer', 'exact'):
        raise ValueError(f'method must be "transfer" or "exact". Provided: {method}')

    # Replicates and clade keys from separate streams: the replicates are the same for
    # both methods at a given seed
    rng, rng_keys = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    nodes = [x for x in tree.internal_nodes if x != tree.root]
    if method == 'exact':
        keys = rng_keys.integers(1, np.iinfo(np.uint64).max, size=len(cells), dtype=np.uint64)
        ref = _clade_hashes_tree(tree, dict(zip(cells, keys, strict=True)))
        ref_h = np.array([ref[x] for x in nodes], dtype=np.uint64)
    else:
        pos = {c:i for i, c in enumerate(cells)}
        members = np.zeros((len(cells), len(nodes)), dtype=np.int32)
        for j, x in enumerate(nodes):
            members[[pos[c] for c in tree.leaves_in_subtree(x)], j] = 1
        sizes = members.sum(0).astype(np.int64)

    # Each replicate is built on a random order of the cells: cells with identical
    # profiles are tied, and average linkage resolves ties by order, so a fixed order
    # would rebuild the same arbitrary split of identical cells in every replicate
    total = np.zeros(len(nodes))
    for b in range(n_boot):
        AD_b, B = _replicate_calls(AD, COV, p, eps, alpha, rng)
        perm = rng.permutation(len(cells))
        D = _distances(AD_b[perm], COV[perm], B[perm])
        Z = upgma_linkage(D)
        if method == 'exact':
            total += np.isin(ref_h, _clade_hashes_linkage(Z, keys[perm]))
        else:
            total += _transfer_scores(Z, members[perm], sizes)

    support = pd.Series(total/n_boot, index=nodes, name=key_added)
    support[tree.root] = 1.0
    for node, v in support.items():
        tree.set_attribute(node, key_added, float(v))

    record(afm, 'clade_support', {
        'params' : {'n_boot':n_boot, 'alpha':alpha, 'method':method, 'seed':seed, 'key_added':key_added},
        'median_support' : float(support.median()),
        'seconds' : T.stop()
    })
    logging.info(f'Clade support ({n_boot} count replicates): median {support.median():.2f}')

    return support


##
