"""
Benchmark metrics: partition scores, and MitoDrift's clade-level scoring mirrored in Python.

MitoDrift definitions (sankaranlab/mitodrift, R/utils.R and R/diagnostics.R):

* `trim_tree(tree, conf, collapse_trivial)`: internal nodes with confidence < conf are
  collapsed; with collapse_trivial, a root split of one tip vs the rest is collapsed too.
* `assign_clones_polytomy(tree, k=Inf)`: every internal child of the root is one clone;
  tips attached directly to the root are singletons.
* `compute_pr_from_jaccard_mat`: a predicted clade is EVERY internal node, scored by its
  confidence; a truth clade is recovered at cutoff c if some predicted clade with
  confidence >= c has Jaccard >= j_thres with it; precision is the fraction of predicted
  clades with confidence >= c whose best Jaccard is >= j_thres.
"""

import numpy as np
import pandas as pd
from ete3 import Tree
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

UNASSIGNED = 'unassigned'


##


def partition_scores(labels, truth):
    """
    Partition metrics of `labels` (index = cells, 'unassigned' allowed) against `truth`
    (index = cells; NaN = unlabelled). Two conventions, both reported:

    * ARI / NMI on the cells that are labelled in truth AND assigned (our usual);
    * ARI_all on every truth-labelled cell, unassigned or missing cells as singletons
      (abstention charged as error, as in the MitoDrift all-cells comparison).
    """

    truth = truth.dropna().astype(str)
    lab = labels.reindex(truth.index).astype(object).fillna(UNASSIGNED).astype(str)
    ok = lab != UNASSIGNED
    lab_all = lab.where(ok, pd.Series([f'_single_{i}' for i in range(lab.size)], index=lab.index))

    return dict(
        n_truth=int(truth.size),
        n_assigned=int(ok.sum()),
        yield_=round(float(ok.mean()), 4),
        n_labels=int(lab[ok].nunique()),
        ARI=round(adjusted_rand_score(truth[ok], lab[ok]), 4) if ok.sum() > 1 else np.nan,
        NMI=round(normalized_mutual_info_score(truth[ok], lab[ok]), 4) if ok.sum() > 1 else np.nan,
        ARI_all=round(adjusted_rand_score(truth, lab_all), 4),
    )


##


def size_strata(truth, bins=((0.01, 1.0), (0.0025, 0.01), (0.001, 0.0025))):
    """Truth clones by size, as fractions of the truth-labelled cells (MitoDrift strata)."""
    frac = truth.value_counts(normalize=True)
    return {f'{lo:g}-{hi:g}': set(frac.index[(frac >= lo) & (frac < hi if hi < 1 else frac <= hi)])
            for lo, hi in bins}


##


def jaccard_matrix(truth_sets, pred_sets):
    """Jaccard between every truth set (rows) and predicted set (columns), over cell names."""
    cells = sorted(set().union(*truth_sets, *pred_sets))
    pos = {c: i for i, c in enumerate(cells)}

    def mat(sets):
        M = np.zeros((len(sets), len(cells)), dtype=np.float32)
        for i, s in enumerate(sets):
            M[i, [pos[c] for c in s]] = 1
        return M

    T, P = mat(truth_sets), mat(pred_sets)
    inter = T @ P.T
    union = T.sum(1)[:, None] + P.sum(1)[None, :] - inter

    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


##


def clade_pr(truth, pred_sets, conf=None, cutoffs=(0.0,), j_thres=0.5, strata=None):
    """
    MitoDrift clade precision / recall.

    truth : pd.Series, cell -> truth clone (clones of size >= 2 are the truth clades).
    pred_sets : list of sets of cells (every internal node of a tree, or every clone of a
        partition); restricted to the truth-labelled cells.
    conf : confidence per predicted set (1 for a partition).
    """

    truth = truth.dropna().astype(str)
    sizes = truth.value_counts()
    clones = list(sizes.index[sizes >= 2])
    truth_sets = [set(truth.index[truth == c]) for c in clones]
    keep = set(truth.index)
    pred_sets = [s & keep for s in pred_sets]
    nonempty = [i for i, s in enumerate(pred_sets) if len(s)]
    pred_sets = [pred_sets[i] for i in nonempty]
    conf = np.ones(len(pred_sets)) if conf is None else np.asarray(conf, float)[nonempty]

    J = jaccard_matrix(truth_sets, pred_sets)
    col_ok = J.max(0) >= j_thres
    row_conf = np.array([conf[r >= j_thres].max() if (r >= j_thres).any() else -np.inf for r in J])

    rows = []
    for c in cutoffs:
        keep_c = conf >= c
        rec = row_conf >= c
        row = dict(cutoff=c, n_pred=int(keep_c.sum()),
                   precision=round(float((col_ok & keep_c).sum() / max(keep_c.sum(), 1)), 4),
                   recall=round(float(rec.mean()), 4), n_truth=len(clones))
        for name, members in (strata or {}).items():
            m = np.array([cl in members for cl in clones])
            row[f'recall_{name}'] = round(float(rec[m].mean()), 4) if m.any() else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


##


def read_mitodrift_tree(path):
    """Newick with internal node labels = clade confidence (MitoDrift output)."""
    t = Tree(path, format=1)
    for n in t.traverse():
        if not n.is_leaf():
            try:
                n.support = float(n.name)
            except ValueError:
                n.support = 0.0
    return t


def tree_clades(t):
    """(sets of leaf names, confidence) for every internal node except the root."""
    sets, conf = [], []
    for n in t.traverse():
        if n.is_leaf() or n.is_root():
            continue
        sets.append(set(n.get_leaf_names()))
        conf.append(n.support)
    return sets, np.array(conf)


def trim_tree(t, conf, collapse_trivial=True):
    """Copy of `t` with internal nodes of confidence < conf collapsed (MitoDrift trim_tree)."""
    t = t.copy()
    for n in list(t.traverse('postorder')):
        if not n.is_leaf() and not n.is_root() and n.support < conf:
            n.delete(preserve_branch_length=True)
    if collapse_trivial and conf > 0:
        kids = t.children
        if any(k.is_leaf() for k in kids):
            for k in kids:
                if not k.is_leaf():
                    k.delete(preserve_branch_length=True)
    return t


def assign_clones_polytomy(t):
    """k=Inf: each internal child of the root is a clone; root tips are singletons."""
    lab = {}
    for i, k in enumerate(t.children):
        for leaf in k.get_leaf_names():
            lab[leaf] = f'c{i}' if not k.is_leaf() else f'_root_single_{leaf}'
    return pd.Series(lab)


def mitodrift_ari(t, truth, conf, min_frac, collapse_trivial=False):
    """
    Fig 2F ARI: clones from the trimmed tree (no trivial-root collapse), on the cells of
    truth clones > min_frac of the labelled cells, root singletons excluded. Reproduces
    the published 0.92 (>1%) and 0.75 (>0.25%) at conf=0.125 exactly (0.918, 0.750).
    """
    lab = assign_clones_polytomy(trim_tree(t, conf, collapse_trivial=collapse_trivial))
    lab = lab[~lab.str.startswith('_root_single_')]
    truth = truth.dropna().astype(str)
    frac = truth.value_counts(normalize=True)
    cells = truth.index[truth.map(frac) > min_frac].intersection(lab.index)
    return adjusted_rand_score(truth[cells], lab[cells])
