"""
Phase 1C: where does MiTo lose the small LARRY clones? And do size-aware abstention (D1)
or a small-clone pass (D2) recover them?

For every LARRY clone (>= 2 cells) on the pL1000 Filter-2 matrix, the first stage at
which it becomes unrecoverable:

    no_candidate  no variant with >= 2 carriers inside the clone survives the candidate
                  filter (a marker needs carriers in the clone to exist at all)
    no_character  ... survives the whole of filter_afm
    cells_lost    fewer than 2 of its cells are left after filter_afm
    not_in_tree   no tree clade has Jaccard >= .5 with the clone (restricted to kept cells)
    cut           the tree has it, but the evidence cut gives no clone with J >= .5
    abstention    the cut has it, the tau abstention removes it
    recovered

Then D1 and D2 are evaluated on top of the same filtered AFM and tree.

D1, size-aware abstention: a cell keeps its label if its best neighbourhood support is
    >= tau * min(1, (n_clone - 1) / k_eff), i.e. tau relative to what a clone of that size
    can reach at all in a k-neighbourhood.
D2, small-clone pass: among cells left unassigned (or in clones), a variant whose
    carriers number 2..`max_cells`, whose carriers form a clade of the tree with at most
    `slack` extra cells, and which is called in every carrier, defines a new clone over
    those carriers.
"""

import logging
import os
import sys
import warnings

sys.setrecursionlimit(50000)
warnings.filterwarnings('ignore')
logging.disable(logging.INFO)

import numpy as np
import pandas as pd

import mito as mt
from mito.pp._graph import knn_kernel
from mito.ut.phylo_utils import get_clades

sys.path.insert(0, os.path.dirname(__file__))
import datasets as ds  # noqa: E402
import metrics as me  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), 'results')


def best_j(truth_set, pred_sets):
    if not pred_sets:
        return 0.0
    return max(len(truth_set & s) / len(truth_set | s) for s in pred_sets)


def size_aware_labels(afm, labels_cut, tau, k=30):
    """D1: abstention threshold scaled by what a clone of that size can reach."""
    B = afm.layers['bin'].toarray() > 0
    W = knn_kernel(afm.X.toarray(), k=k, normalize=True)
    support = np.where(B, np.clip(W @ B.astype(float), 0, 1), 0.0).max(1)
    sizes = labels_cut.map(labels_cut.value_counts())
    reach = np.minimum(1.0, (sizes.values - 1) / k)
    thr = tau * reach
    lab = labels_cut.copy()
    lab[(support < thr) | (labels_cut == me.UNASSIGNED).values] = me.UNASSIGNED
    return lab


def small_clone_pass(afm, tree, labels, max_cells=4, slack=0):
    """D2: small private-marker clades become clones."""
    B = pd.DataFrame(afm.layers['bin'].toarray() > 0, index=afm.obs_names, columns=afm.var_names)
    clades = get_clades(tree, with_root=False, with_singletons=False)
    clade_sets = [set(v) for v in clades.values() if len(v) <= max_cells + slack]
    lab = labels.copy()
    n_new = 0
    for v in B.columns:
        car = set(B.index[B[v]])
        if not 2 <= len(car) <= max_cells:
            continue
        # carriers form (or almost form) a clade
        if not any(car <= s and len(s) - len(car) <= slack for s in clade_sets):
            continue
        lab.loc[list(car)] = f'small_{v}'
        n_new += 1
    return lab, n_new


def ceiling_and_tree():
    """
    (a) Observable ceiling: truth clones with a variant seen (>= 1 molecule) in >= 2 of
        their cells and in at least as many of their cells as outside them, in the
        deposited Filter-2 matrix. 2026-10-08: 12/13, 99/117, 81/126 by stratum.
    (b) MiTo's tree scored like MitoDrift (every internal node a predicted clade).
        2026-10-08: recall 37.9% (76.9 / 58.1 / 15.1% by stratum), precision 35.3%.
    """
    dep = pd.read_csv(os.path.join(ds.PL1000, 'pL1000_full_mut_dat.csv'))
    truth = ds.pl1000_truth()
    strata = me.size_strata(truth)
    A = dep.pivot(index='cell', columns='variant', values='a').reindex(truth.index).fillna(0) > 0
    for name, clones in strata.items():
        ok = 0
        for c in clones:
            k_in = A.loc[truth.index[truth == c]].sum(0)
            k_out = A.loc[truth.index[truth != c]].sum(0)
            ok += int(((k_in >= 2) & (k_in >= k_out)).any())
        print(f'ceiling {name}: {ok}/{len(clones)}', flush=True)
    afm, _ = ds.pl1000_filter2()
    mt.pp.filter_afm(afm)
    tree = mt.tl.build_tree(afm, solver='UPMGA')
    sets = [set(v) for v in get_clades(tree, with_root=False).values()]
    print(me.clade_pr(truth, sets, strata=strata).to_string(), flush=True)


def main():
    afm0, truth = ds.pl1000_filter2()
    strata = me.size_strata(truth)
    sizes = truth.value_counts()
    clones = list(sizes.index[sizes >= 2])
    tsets = {c: set(truth.index[truth == c]) for c in clones}

    afm = afm0.copy()
    mt.pp.filter_afm(afm)
    flow = afm.uns['mito']['filter_afm']['flow']
    print(flow, flush=True)
    tree = mt.tl.build_tree(afm, solver='UPMGA')
    kept = set(afm.obs_names)
    B = pd.DataFrame(afm.layers['bin'].toarray() > 0, index=afm.obs_names, columns=afm.var_names)

    # Candidate-stage markers: rerun the candidate filter alone
    cand = afm0.copy()
    mt.pp.filter_low_quality_variants(cand)
    mt.pp.filter_candidate_variants(cand)
    Ccand = pd.DataFrame((cand.layers['AD'].toarray() > 0), index=cand.obs_names, columns=cand.var_names)

    clades = get_clades(tree, with_root=False, with_singletons=False)
    clade_sets = [set(v) for v in clades.values()]

    rows = []
    for tau in (0.25, 0.1):
        mt.tl.annotate_clones(tree, afm, tau=0)
        cut = afm.obs['MiTo_clone'].astype(str)
        mt.tl.annotate_clones(tree, afm, tau=tau)
        lab = afm.obs['MiTo_clone'].astype(str)
        cut_sets = [set(cut.index[cut == L]) for L in cut.unique() if L != me.UNASSIGNED]
        lab_sets = [set(lab.index[lab == L]) for L in lab.unique() if L != me.UNASSIGNED]
        for c in clones:
            s = tsets[c]
            sk = s & kept
            if not (Ccand.loc[Ccand.index.intersection(list(s))].sum(0) >= 2).any():
                stage = 'no_candidate'
            elif not (B.loc[B.index.intersection(list(s))].sum(0) >= 2).any():
                stage = 'no_character'
            elif len(sk) < 2:
                stage = 'cells_lost'
            elif best_j(sk, [x & kept for x in clade_sets]) < 0.5:
                stage = 'not_in_tree'
            elif best_j(s, cut_sets) < 0.5:
                stage = 'cut'
            elif best_j(s, lab_sets) < 0.5:
                stage = 'abstention'
            else:
                stage = 'recovered'
            stratum = next((k for k, v in strata.items() if c in v), 'other')
            rows.append(dict(tau=tau, clone=c, size=len(s), stratum=stratum, stage=stage))
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, 'small_clones_loss.csv'), index=False)
    print(pd.crosstab([df['tau'], df['stratum']], df['stage']).to_string(), flush=True)

    # D1 / D2 on the same AFM and tree
    out = []
    mt.tl.annotate_clones(tree, afm, tau=0)
    cut = afm.obs['MiTo_clone'].astype(str)
    for name, lab in [
        ('tau0.25', None), ('tau0.1', None),
        ('D1_tau0.25', size_aware_labels(afm, cut, 0.25)),
        ('D1_tau0.5', size_aware_labels(afm, cut, 0.5)),
    ]:
        if lab is None:
            mt.tl.annotate_clones(tree, afm, tau=float(name[3:]))
            lab = afm.obs['MiTo_clone'].astype(str)
        for d2 in (None, (4, 0), (4, 1), (8, 1)):
            L = lab
            n_new = 0
            if d2:
                L, n_new = small_clone_pass(afm, tree, lab, max_cells=d2[0], slack=d2[1])
            sets = [set(L.index[L == x]) for x in L.unique() if x != me.UNASSIGNED]
            row = dict(config=name, D2=str(d2), n_new=n_new, **me.partition_scores(L, truth))
            pr = me.clade_pr(truth, sets, strata=strata).iloc[0]
            row.update({k: pr[k] for k in pr.index if k.startswith(('recall', 'precision'))})
            out.append(row)
            print(row, flush=True)
    pd.DataFrame(out).to_csv(os.path.join(RESULTS, 'small_clones_D1D2.csv'), index=False)


if __name__ == '__main__':
    ceiling_and_tree() if '--ceiling' in sys.argv else main()
