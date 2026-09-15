"""
Selective dropout imputation, gated on neighbourhood COHERENCE.

A prevalence threshold alone asks "are most of my neighbours carriers?". A cell
sitting on a clade boundary can satisfy that while its neighbourhood is split
between two clades -- and those are exactly the imputations that are both most
likely wrong and most damaging downstream: they raise a marker's prevalence in
SIBLING clades, which destroys the specificity test the tree cut depends on
(measured: evidence-cut collapses 12 -> 5 -> 2 labels as imputation gets more
aggressive, while imputation precision on MDA_clones is only 0.68).

Two extra gates, both computed from the same leave-one-out neighbourhood:

  SIDE     the cell must be closer to the neighbours that carry j than to those
           that do not. A boundary cell is by definition roughly equidistant, so
           this is the direct question: is this cell on the carrier side?

  COHERENCE the neighbourhood must be near-unanimous on the OTHER variants too.
           A clade is a set of cells agreeing on all characters, so a split
           neighbourhood shows up as intermediate prevalence on variants that
           have nothing to do with j. Measured only over variants that are
           actually informative locally -- averaging over the many variants that
           are absent everywhere would score every neighbourhood as coherent.
"""
import numpy as np

from caller import _cosine_D


def impute_dropouts(X, B, k=30, thr=0.7, min_support=1, loo=True,
                    min_margin=-np.inf, min_coherence=0.0, return_diag=False):
    """
    thr            weighted fraction of the neighbourhood carrying j
    min_margin     required (similarity to carriers - similarity to non-carriers).
                   -inf disables it. NB a default of 0.0 does NOT disable it: a
                   boundary cell can be marginally closer to the non-carriers, so
                   0.0 is already a real gate and quietly changes the baseline.
    min_coherence  required unanimity of the neighbourhood on the other variants
    """
    n, m = X.shape
    G = X @ X.T
    sq = (X**2).sum(1)
    Bf = B.astype(float)
    support = Bf.sum(1)
    P = np.zeros((n, m)); MARG = np.zeros((n, m)); COH = np.ones((n, m))
    for j in range(m):
        D = _cosine_D(X, G, sq, j, skip=not loo)
        np.fill_diagonal(D, np.inf)
        kk = min(int(k[j]) if np.ndim(k) else k, n-1)          # k may be per-variant
        idx = np.argpartition(D, kk, axis=1)[:, :kk]
        dist = np.take_along_axis(D, idx, 1)
        sig = np.maximum(dist.max(1, keepdims=True), 1e-9)
        w = np.exp(-(dist/sig)**2)
        w = w/np.maximum(w.sum(1, keepdims=True), 1e-12)
        c = Bf[idx, j]                                   # carrier status of neighbours
        P[:, j] = (w*c).sum(1)
        # SIDE: mean similarity to carrier vs non-carrier neighbours
        s = 1.0 - dist
        npos = c.sum(1); nneg = (1-c).sum(1)
        s_pos = np.where(npos > 0, (s*c).sum(1)/np.maximum(npos, 1), -np.inf)
        s_neg = np.where(nneg > 0, (s*(1-c)).sum(1)/np.maximum(nneg, 1), -np.inf)
        MARG[:, j] = np.where(np.isfinite(s_neg), s_pos - s_neg,
                              np.where(np.isfinite(s_pos), 1.0, 0.0))
        if min_coherence > 0:
            # prevalence of every OTHER variant in this same neighbourhood
            Pn = np.einsum('ik,ikm->im', w, Bf[idx])     # (n, m)
            Pn[:, j] = 0.5                               # exclude j itself
            informative = (Pn > 0.05) & (Pn < 0.95)
            unan = np.abs(2*Pn - 1)
            cnt = informative.sum(1)
            COH[:, j] = np.where(cnt > 0, (unan*informative).sum(1)/np.maximum(cnt, 1), 1.0)
    other = support[:, None] - Bf
    add = ((B == 0) & (P >= thr) & (other >= min_support)
           & (MARG >= min_margin) & (COH >= min_coherence))
    out = B.copy()
    out[add] = 1
    return (out, add, dict(P=P, margin=MARG, coh=COH)) if return_diag else (out, add)
