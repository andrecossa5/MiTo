"""
Discard criterion: four-gamete conflict resolved by community quality, plus an
explicit per-cell membership test.

The current `maxcompat` breaks four-gamete conflicts by CONFLICT COUNT -- pure
topology. It has no notion of whether either variant is a plausible clone marker,
so a noise variant scattered over the graph can evict a real one, and any cell
whose only call was on the evicted variant then disappears from the analysis
silently. That is where most unassigned cells come from (597 -> 372 at
5cl/depth3), and it is a decision about variants applied as if it were a decision
about cells.

Two changes:

  * conflicts are resolved by COMMUNITY COMPACTNESS. A real clone marker's
    positive cells form a tight neighbourhood on the kNN graph; a noise variant's
    do not. Compactness is internal edge weight over incident volume (1 minus
    conductance), divided by the value a random set of the same size would get, so
    it is comparable across variants of very different prevalence.

  * cells are then kept only if they sit INSIDE the community of a surviving
    marker they carry -- and cells that fail are marked `unassignable` rather than
    dropped. With a median of 1-2 calls per cell there is no redundancy to check,
    so the test has to be about the quality of the single supporting call, not
    agreement between several.
"""
import numpy as np
from itertools import combinations


def community_score(K, B):
    """
    Compactness enrichment of each variant's positive set on the kNN graph.

    score = (internal edge weight / incident volume) / (|P| / n)

    1 is what a random set of that size scores; a clone marker scores many times
    that. The normalisation matters -- without it every large set looks compact.
    """
    A = (K + K.T)/2.0
    n = A.shape[0]
    deg = A.sum(1)
    Bf = B.astype(float)
    vol = Bf.T @ deg
    internal = np.einsum('ij,jk,ki->i', Bf.T, A, Bf)
    size = Bf.sum(0)
    compact = internal/np.maximum(vol, 1e-9)
    expected = np.maximum(size/n, 1e-9)
    return compact/expected


def maxcompat_scored(B, score, min_gamete=10):
    """Greedy four-gamete resolution, dropping the LOWER-SCORING variant of a conflict."""
    B0 = B[B.sum(1) >= 1] > 0
    m = B0.shape[1]
    inc = np.zeros((m, m), bool)
    for i, j in combinations(range(m), 2):
        x, y = B0[:, i], B0[:, j]
        if min(int((x & y).sum()), int((x & ~y).sum()), int((~x & y).sum())) > min_gamete:
            inc[i, j] = inc[j, i] = True
    alive = np.ones(m, bool)
    while True:
        idx = np.flatnonzero(alive)
        sub = inc[np.ix_(alive, alive)]
        if not sub.any():
            break
        conflicted = idx[sub.any(1)]
        # among variants still in conflict, drop the least community-like
        alive[conflicted[int(np.argmin(score[conflicted]))]] = False
    return alive


def cell_membership(K, B, tau=0.25):
    """
    Does each cell sit inside the community of a marker it carries?

    `core[i, j]` is the share of cell i's neighbourhood positive for variant j, so
    the test asks whether the cell's own call is corroborated by its neighbours --
    which is the only corroboration available when a cell has a single call.
    """
    Wn = K/np.maximum(K.sum(1, keepdims=True), 1e-12)
    core = Wn @ B.astype(float)
    return ((B > 0) & (core >= tau)).any(1), core
