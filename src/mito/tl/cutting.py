"""
Evidence-balanced tree cut: split a clade only when the split is paid for.
"""

import sys

import numpy as np
import pandas as pd
from cassiopeia.data import CassiopeiaTree
from scipy import stats
from scipy.special import xlogy

sys.setrecursionlimit(50000)

##


def _marker(B, inside, outside, alpha=0.01, min_in=0.25, ratio=3.0):
    """
    Is any character a marker for `inside` against `outside`? G-test (binomial likelihood
    ratio, ~chi2_1), Bonferroni-corrected over the characters tested at this node.

    Two guards keep significance from standing in for relevance: an absolute floor on
    inside prevalence, and a ratio on specificity, which adapts to prevalence where a
    fixed maximum outside prevalence cannot.
    """

    n_in = float(inside.sum())
    n_out = float(outside.sum())
    if n_in == 0 or n_out == 0:
        return -np.inf, -1

    a = B[inside].sum(0).astype(float)          # positives inside
    c = B[outside].sum(0).astype(float)         # positives outside
    f_in = a/n_in
    f_out = c/n_out
    ok = (f_in>=min_in) & (f_in>=ratio*f_out) & (f_in>f_out)
    if not ok.any():
        return -np.inf, -1

    f = np.clip((a+c)/(n_in+n_out), 1e-12, 1-1e-12)
    fi = np.clip(f_in, 1e-12, 1-1e-12)
    fo = np.clip(f_out, 1e-12, 1-1e-12)
    g2 = 2*(xlogy(a, fi/f) + xlogy(n_in-a, (1-fi)/(1-f))
            + xlogy(c, fo/f) + xlogy(n_out-c, (1-fo)/(1-f)))
    g2 = np.nan_to_num(g2, nan=0.0, posinf=0.0, neginf=0.0)
    pv = np.where(ok, stats.chi2.sf(np.maximum(g2, 0.0), 1), 1.0)
    thr = alpha/max(B.shape[1], 1)
    if not (pv<=thr).any():
        return -np.inf, -1
    sc = np.where(pv<=thr, g2, -np.inf)
    j = int(sc.argmax())

    return float(sc[j]), j


##


def evidence_cut(
    tree: CassiopeiaTree,
    B: np.array,
    cells: list,
    alpha: float = 0.01,
    min_in: float = 0.25,
    ratio: float = 3.0,
    self_min_in: float = 0.85,
    min_cells: int = 5,
    min_supported: int = 2,
    one_sided: bool = False,
    return_detail: bool = False
    ):
    """
    Recursive, evidence-balanced cut of a cell phylogeny into clones.

    Descending from the root, a node is split into its children only if at least
    `min_supported` children carry their OWN marker characters: prevalent inside the
    child, and absent from the child's SIBLINGS. If they do not, but the node itself is
    supported by a marker, recursion stops and all of the node's cells become one clone.

    The sibling comparison is the crux. Specificity measured against the whole population
    is the wrong test: a mutation marking a node is, by inheritance, present in every cell
    of every child, so it looks perfectly specific to each child in turn and licenses a
    split it provides no evidence for. Measured against the siblings it contributes
    nothing, which is correct - an ancestral mutation is evidence for the node, never for
    subdividing it.

    The two prevalence bars are deliberately asymmetric. The child test licenses a SPLIT,
    so it must be permissive enough to see a deep, recent, dropout-riddled marker; the
    self test licenses a STOP, so it must be strict - a node is declared a single clone
    only on a marker that is genuinely uniform across it.

    Parameters
    ----------
    tree : CassiopeiaTree
        Cell phylogeny, with cell names as leaves.
    B : np.array
        (n_cells, n_characters) genotypes.
    cells : list
        Cell names, row-aligned with `B`.
    alpha : float, optional
        Significance of the marker test, Bonferroni-corrected per node. Default is 0.01.
    min_in : float, optional
        Inside-prevalence floor for a CHILD marker (permissive). Default is 0.25.
    ratio : float, optional
        Inside prevalence must exceed `ratio` x sibling prevalence. Default is 3.0.
    self_min_in : float, optional
        Inside-prevalence floor for the node's OWN marker (strict). Default is 0.85.
    min_cells : int, optional
        Minimum clade size considered a candidate clone. Default is 5.
    min_supported : int, optional
        Children that must carry their own marker for a split to proceed. 2 is the
        meaningful default: one supported child is a clade plus a remainder, which is not
        evidence of two clones. Default is 2.
    one_sided : bool, optional
        Also split a single supported child off its parent, the remainder keeping the
        parent label. Resolves nested subclones, at the cost of oversplitting
        barcode-level clones. Default is False.
    return_detail : bool, optional
        Also return the per-node decisions. Default is False.

    Returns
    -------
    pd.Series (, pd.DataFrame)
        Clone label per cell.
    """

    pos = {c:i for i, c in enumerate(cells)}
    labels = pd.Series('unassigned', index=list(cells), dtype=object)
    detail = []

    def idx_of(node):
        lv = [pos[c] for c in tree.leaves_in_subtree(node) if c in pos]
        m = np.zeros(len(cells), bool)
        m[lv] = True
        return m

    def assign(node, mask):
        labels.loc[[cells[i] for i in np.flatnonzero(mask)]] = str(node)

    def recurse(node, mask, parent_mask):
        sibs = parent_mask & ~mask
        self_s, self_j = _marker(B, mask, sibs, alpha, self_min_in, ratio) if sibs.any() else (-np.inf, -1)
        kid_masks = [(k, idx_of(k)) for k in tree.children(node)]
        kid_masks = [(k, m) for k, m in kid_masks if m.sum()>=min_cells]

        if len(kid_masks) == 1:
            # LADDER node: only one child clears min_cells, the rest is a tiny outlier
            # branch. That is no evidence the node is one clone, so the node is made
            # transparent and the big child is tested against the node's own siblings.
            # Stopping here instead gave a whole subtree one label whenever a few outlier
            # cells attached high in the tree.
            k, m = kid_masks[0]
            recurse(k, m, parent_mask)
            rest = mask & ~m
            below = labels.iloc[np.flatnonzero(m)].unique()
            if rest.any() and len(below) == 1:      # outliers follow a child that stayed one clone
                labels.iloc[np.flatnonzero(rest)] = below[0]
            return

        if len(kid_masks) < 2:
            assign(node, mask)
            return

        supported = [(k, m) for k, m in kid_masks
                     if _marker(B, m, mask & ~m, alpha, min_in, ratio)[1] >= 0]
        detail.append(dict(node=str(node), n=int(mask.sum()), n_children=len(kid_masks),
                           n_supported=len(supported), self_supported=self_j>=0))

        if len(supported)>=min_supported or self_j<0:
            # Either the split is paid for, or the node is not a clone at all - only
            # unresolved ancestry, so keep descending. Halting on an unsupported node is
            # how a cut collapses to a single label: near the root no mutation is uniform
            # across a subtree holding many clones.
            covered = np.zeros(len(cells), bool)
            for k, m in kid_masks:
                recurse(k, m, mask)
                covered |= m
            rest = mask & ~covered
            if rest.sum()>0:
                assign(node, rest)
        elif one_sided and len(supported) == 1 and (mask & ~supported[0][1]).sum()>=min_cells:
            k, m = supported[0]
            recurse(k, m, mask)
            assign(node, mask & ~m)
        else:
            assign(node, mask)

    root = tree.root
    recurse(root, idx_of(root), idx_of(root))

    return (labels, pd.DataFrame(detail)) if return_detail else labels


##
