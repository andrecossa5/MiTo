"""Shared pipeline helpers."""
import numpy as np, pandas as pd, mito as mt
from itertools import combinations
from mito.ut.phylo_utils import get_clades
from caller import _cosine_D
from REFERENCE_CONFIG import SIM_MOL as SIM

FK = dict(min_mean_DP_in_positives=10, min_frac_negative=0.5, af_confident_detection=0.01)


def kernel(X, k=30, raw=False):
    nC = X.shape[0]
    D = _cosine_D(X, X@X.T, (X**2).sum(1)); np.fill_diagonal(D, np.inf)
    k = min(k, nC-1); idx = np.argpartition(D, k, axis=1)[:, :k]
    dist = np.take_along_axis(D, idx, 1); sig = np.maximum(dist.max(1, keepdims=True), 1e-9)
    W = np.zeros((nC, nC)); np.put_along_axis(W, idx, np.exp(-(dist/sig)**2), 1)
    return W if raw else W/np.maximum(W.sum(1, keepdims=True), 1e-12)


def votes(a, min_cells=5):
    """Split-vote cutter: each variant votes for its argmax clade; smallest wins."""
    tree = mt.tl.build_tree(a, precomputed=True, solver='UPMGA')
    m = mt.tl.MiToTreeAnnotator(tree); m.get_T(); m.get_M()
    best = m.M.columns[m.M.values.argmax(axis=1)]
    cl = get_clades(tree, with_root=True, with_singletons=True)
    sets = {q: set(cl[q]) for q in set(best) if q in cl and len(cl[q]) >= min_cells}
    lab = pd.Series('unassigned', index=a.obs_names, dtype=object)
    for q in sorted(sets, key=lambda q: len(sets[q])):
        lab.loc[[c for c in sets[q] if lab.loc[c] == 'unassigned']] = q
    return lab


def maxcompat(B, min_gamete=10):
    """Greedy four-gamete filter; drops the variant involved in most conflicts."""
    B0 = B[B.sum(1) >= 1] > 0
    m = B0.shape[1]
    inc = np.zeros((m, m), bool)
    for i, j in combinations(range(m), 2):
        x, y = B0[:, i], B0[:, j]
        if min(int((x&y).sum()), int((x&~y).sum()), int((~x&y).sum())) > min_gamete:
            inc[i, j] = inc[j, i] = True
    alive = np.ones(m, bool)
    while True:
        sub = inc[np.ix_(alive, alive)]
        if not sub.any(): break
        alive[np.flatnonzero(alive)[int(sub.sum(1).argmax())]] = False
    return alive
