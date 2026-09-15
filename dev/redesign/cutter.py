"""
Evidence-balanced tree cut: split only when the split is paid for.

The vote cutter only ever splits. Every retained variant nominates a clade, no
clade is ever contracted, and a cell in both a parent and a child nominated clade
always takes the child -- a guess, not an inference. Label counts therefore scale
with the variant count (24 labels for 8 true clones on MDA_clones).

Here a split has to be earned. Descending from the root, a node P is split into
its children only if the children carry their OWN marker mutations: uniform
inside the child, and absent from the child's SIBLINGS. If they do not, but P
itself is supported by a marker, recursion stops and all of P's cells become one
clone.

The sibling comparison is the crux. Specificity measured against the whole
population is the wrong test: a mutation marking P is, by inheritance, present in
every cell of every child of P, so it looks perfectly specific to each child in
turn and licenses a split it provides no evidence for. Measured against P \\ C it
contributes nothing, which is correct -- an ancestral mutation is evidence for P,
never for subdividing it.

Every cell below the root receives a label, so assignment is total by
construction and label counts are not inflated by redundant markers.
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

sys.setrecursionlimit(50000)


def _marker(B, inside, outside, alpha=0.01, min_in=0.25, ratio=3.0):
    """
    Is any mutation a marker for `inside` against `outside`? G-test, on counts that
    may be SOFT.

    `B` is either binary calls or the genotype posterior gamma. With gamma, the
    "number of carriers in a clade" becomes an expected count sum(gamma), which is
    not an integer -- so the test has to be one that accepts non-integer counts.
    The G-test (binomial likelihood ratio, ~chi2_1) does; the hypergeometric tail
    does not. Using it on BOTH paths keeps hard-vs-soft a difference in the data
    rather than a difference in the test.

    This matters because the tree and the cut must reason about the same object.
    Building the tree from gamma while testing markers on thresholded calls leaves
    the cut evaluating splits of a topology it cannot see -- which is exactly how
    soft distances plus a hard-call cut collapsed to ARI 0.113 where the hard tree
    gave 0.683.

    Two guards keep significance from standing in for relevance: an absolute floor
    on inside prevalence, and a ratio on specificity, which adapts to prevalence
    where a fixed `max_out` cannot.
    """
    n_in = float(inside.sum()); n_out = float(outside.sum())
    if n_in == 0 or n_out == 0:
        return -np.inf, -1
    a = B[inside].sum(0).astype(float)          # (expected) positives inside
    c = B[outside].sum(0).astype(float)         # (expected) positives outside
    f_in = a/n_in
    f_out = c/n_out
    ok = (f_in >= min_in) & (f_in >= ratio*f_out) & (f_in > f_out)
    if not ok.any():
        return -np.inf, -1
    f = np.clip((a+c)/(n_in+n_out), 1e-12, 1-1e-12)
    fi = np.clip(f_in, 1e-12, 1-1e-12)
    fo = np.clip(f_out, 1e-12, 1-1e-12)
    g2 = 2*(xlogy(a, fi/f) + xlogy(n_in-a, (1-fi)/(1-f))
            + xlogy(c, fo/f) + xlogy(n_out-c, (1-fo)/(1-f)))
    g2 = np.nan_to_num(g2, nan=0.0, posinf=0.0, neginf=0.0)
    pv = np.where(ok, stats.chi2.sf(np.maximum(g2, 0.0), 1), 1.0)
    thr = alpha/max(B.shape[1], 1)              # Bonferroni over variants at this node
    if not (pv <= thr).any():
        return -np.inf, -1
    sc = np.where(pv <= thr, g2, -np.inf)
    j = int(sc.argmax())
    return float(sc[j]), j


def detection_level(B, clade_masks, min_size=5):
    """
    Achievable within-clade marker prevalence -- the data's own detection ceiling.

    For each variant take the clade that best MATCHES its carrier set (max Jaccard)
    and record what fraction of that clade actually carries it. The median over
    variants is roughly 1 - dropout.

    Jaccard, not enrichment: scoring by f_in - f_out lets any variant find some
    tiny clade it happens to saturate, so the estimate pins at 1.0 and the whole
    adaptive mechanism silently reduces to the fixed thresholds. Matching the
    carrier set penalises clades that are much smaller than the variant's extent.
    """
    best = []
    for j in range(B.shape[1]):
        car = B[:, j] > 0
        nc = car.sum()
        if nc < min_size:
            continue
        cand = []
        for m in clade_masks:
            if m.sum() < min_size:
                continue
            inter = np.logical_and(car, m).sum()
            uni = np.logical_or(car, m).sum()
            if uni:
                cand.append((inter/uni, B[m, j].mean()))
        if cand:
            best.append(max(cand)[1])
    return float(np.median(best)) if best else 1.0


def evidence_cut(tree, B, cells, alpha=0.01, min_in=0.25, ratio=3.0, self_min_in=0.7,
                 ref=None, alpha_child=0.30, alpha_self=0.85,
                 min_cells=5, min_supported=2, ladder='stop', one_sided=False, return_detail=False):
    """
    Recursive, evidence-balanced cut of a Cassiopeia tree.

    tree          CassiopeiaTree whose leaves are cell names
    B             (n_cells, n_vars) binary genotypes
    cells         cell names, row-aligned with B
    alpha         Fisher significance, Bonferroni-corrected over variants per node
    min_in        inside-prevalence floor for a CHILD marker (permissive)
    ratio         inside prevalence must exceed `ratio` x sibling prevalence
    self_min_in   inside-prevalence floor for the node's OWN marker (strict)

    The two bars are deliberately asymmetric, and symmetric bars are why the first
    two versions failed in opposite directions. The child test licenses a SPLIT, so
    it must be permissive enough to see a deep, recent, dropout-riddled marker. The
    self test licenses a STOP, so it must be strict -- a node is declared a single
    clone only on a marker that is genuinely uniform across it. Loosening both at
    once (v2) made stopping easier faster than it made splitting easier, and label
    counts collapsed; tightening both (v1) stopped everything at the root.
    min_supported how many children must carry their own marker for a split to
                  proceed. 2 is the meaningful default: one supported child is a
                  clade plus a remainder, which is not evidence of two clones.
    """
    pos = {c: i for i, c in enumerate(cells)}
    labels = pd.Series('unassigned', index=list(cells), dtype=object)
    detail = []
    if ref is not None:
        # Thresholds RELATIVE to the detection ceiling rather than absolute.
        #
        # A fixed bar silently encodes the dropout rate it was tuned at:
        # self_min_in=0.85 was calibrated where within-clade prevalence tops out
        # near 0.83, so it sat just under the ceiling. Improve the genotypes --
        # impute, say -- and prevalence rises to ~0.95, the same bar is now far
        # BELOW the ceiling, every node looks self-supported, and the recursion
        # stops at the root (1-2 labels, ARI 0.000-0.145).
        #
        # Scaling by `ref` makes the question scale-free: does this clade carry a
        # marker as uniform as markers get IN THIS DATASET. Better genotypes then
        # raise both bars automatically, and at perfect genotypes it reduces to
        # "does the child have a marker the parent lacks".
        min_in = alpha_child*ref
        self_min_in = alpha_self*ref

    def idx_of(node):
        lv = [pos[c] for c in tree.leaves_in_subtree(node) if c in pos]
        m = np.zeros(len(cells), bool)
        m[lv] = True
        return m

    def assign(node, mask):
        labels.loc[[cells[i] for i in np.flatnonzero(mask)]] = str(node)

    def recurse(node, mask, parent_mask):
        """
        Descend until a node is BOTH supported itself and not out-voted by its
        children.

        The balance runs in both directions:
          * >= min_supported children carry their own sibling-specific marker
            -> the split is paid for, descend;
          * otherwise, if this node carries a marker of its own, the evidence
            stops here -- a supported parent mutation with no subsequent events
            below it means one clone, so emit it;
          * otherwise the node is not a clone at all, only unresolved ancestry,
            so keep descending. Halting here was the bug in the first version:
            near the root no mutation is uniform across a subtree holding many
            clones, so every cut collapsed to a single label.
        """
        sibs = parent_mask & ~mask
        self_s, self_j = _marker(B, mask, sibs, alpha, self_min_in, ratio) if sibs.any() else (-np.inf, -1)
        kid_masks = [(k, idx_of(k)) for k in tree.children(node)]
        kid_masks = [(k, m) for k, m in kid_masks if m.sum() >= min_cells]
        if len(kid_masks) == 1 and ladder == 'descend':
            # LADDER node: only one child clears min_cells, the rest is a tiny
            # outlier branch. That is no evidence the node is one clone, so the
            # node is made transparent -- the big child is tested against the
            # node's own siblings. Stopping here instead (ladder='stop') gave the
            # whole subtree one label whenever a few outlier cells attached high
            # in the tree (MDA: 281 cells -> one clone, 12 -> 2 labels).
            k, m = kid_masks[0]
            recurse(k, m, parent_mask)
            rest = mask & ~m
            below = labels.iloc[np.flatnonzero(m)].unique()
            # outliers follow the child only if it stayed a single clone
            if rest.any() and len(below) == 1:
                labels.iloc[np.flatnonzero(rest)] = below[0]
            return
        if len(kid_masks) < 2:
            assign(node, mask)
            return
        supported = [(k, m) for k, m in kid_masks
                     if _marker(B, m, mask & ~m, alpha, min_in, ratio)[1] >= 0]
        detail.append(dict(node=str(node), n=int(mask.sum()), n_children=len(kid_masks),
                           n_supported=len(supported), self_supported=self_j >= 0))
        if len(supported) >= min_supported or self_j < 0:
            covered = np.zeros(len(cells), bool)
            for k, m in kid_masks:
                recurse(k, m, mask)
                covered |= m
            rest = mask & ~covered
            if rest.sum() > 0:
                assign(node, rest)
        elif one_sided and len(supported) == 1 and (mask & ~supported[0][1]).sum() >= min_cells:
            # ONE-SIDED split: a nested clone inside its parent. The parent's own
            # cells carry no private variant, so only the child side of the split
            # can ever show a marker, and requiring two supported children
            # (min_supported=2) never separates a child from its parent. The child
            # is split off and recursed into; the remainder keeps the node label
            # as the parent clone.
            k, m = supported[0]
            recurse(k, m, mask)
            assign(node, mask & ~m)
        else:
            assign(node, mask)

    root = tree.root
    recurse(root, idx_of(root), idx_of(root))
    return (labels, pd.DataFrame(detail)) if return_detail else labels
