"""
ReDeeM vs mtscATAC (mgatk), best final pipelines, on the same cells and the same LARRY
barcodes (pL1000). Multiome: both assays come from the same cells.

Cells: redeemV cells with mean coverage >= 10 (3,721), all present in mgatk. Truth: the
LARRY-labelled cells among them; clone strata are MitoDrift's (fractions of all 1,362
LARRY cells), so the numbers stay comparable with the earlier tables.

    ReDeeM     make_afm(redeem-v, edge_trim=9) -> filter_afm defaults (over-dispersion,
               no dbSNP/REDIdb, no SNR) -> UPGMA -> annotate_clones
    mgatk      both-strand molecules: AD = min(fw, rev), DP = rint(reads / 1.66), f from
               strand variance (mgatk_dupfactor.py; = make_afm in 0.5.0) -> filter_afm
               defaults -> UPGMA -> annotate_clones
    MitoDrift  the deposited tree (tau .125 / .25), restricted to the same cells

Each with the small-clone pass off and on. Also per truth clone: recovered (J >= .5) by
which assay; and the agreement of the two partitions with each other.

Usage: python final_compare.py   -> results/final_compare.csv, final_compare_clones.csv
"""

import logging
import os
import sys
import time
import warnings

sys.setrecursionlimit(50000)
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import mito as mt
from mito.ut.phylo_utils import get_clades

sys.path.insert(0, os.path.dirname(__file__))
import datasets as ds  # noqa: E402
import metrics as me  # noqa: E402
import mgatk_strand as ms  # noqa: E402
from mgatk_pairs import molecule_afm  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), 'results')


def pipeline(afm, **filter_kwargs):
    t0 = time.time()
    afm = afm.copy()
    mt.pp.filter_afm(afm, **filter_kwargs)
    tree = mt.tl.build_tree(afm, solver='UPMGA')
    out = {}
    for d2 in (False, True):
        mt.tl.annotate_clones(tree, afm, small_clones=d2)
        out[d2] = afm.obs['MiTo_clone'].astype(str).copy()
    sets = [set(v) for v in get_clades(tree, with_root=False).values()]
    return out, sets, afm, round(time.time() - t0)


def best_j(s, pred_sets):
    return max((len(s & p) / len(s | p) for p in pred_sets), default=0.0)


def score(name, lab, truth, strata, tree_sets=None, conf=None, cutoff=None, **meta):
    row = dict(method=name, **meta, **me.partition_scores(lab, truth))
    sets = [set(lab.index[lab == L]) for L in lab.unique() if L != me.UNASSIGNED]
    pr = me.clade_pr(truth, sets, strata=strata).iloc[0]
    row.update({k: pr[k] for k in pr.index if k.startswith(('recall', 'precision'))})
    if tree_sets is not None:
        tp = me.clade_pr(truth, tree_sets, conf, cutoffs=(cutoff or 0.0,), strata=strata).iloc[0]
        row.update({f'tree_{k}': tp[k] for k in tp.index if k.startswith(('recall', 'precision'))})
    return row


def main():
    rv, truth_all = ds.pl1000_redeemv(edge_trim=9)
    cells = rv.obs_names[rv.obs['mean_site_coverage'].values >= 10]
    d = ms.load()
    cells = cells[np.isin(cells, d['cells'])]
    truth = truth_all.reindex(cells).dropna()
    strata = {k: v & set(truth.unique()) for k, v in me.size_strata(truth_all).items()}
    print(f'{len(cells)} cells, {truth.size} LARRY-labelled, {truth.nunique()} clones '
          f'({", ".join(f"{k}: {len(v)}" for k, v in strata.items())})', flush=True)

    rows, labels = [], {}

    lab, sets, a, secs = pipeline(rv[cells].copy())
    for d2 in (False, True):
        labels[('ReDeeM', d2)] = lab[d2]
        rows.append(score('ReDeeM', lab[d2], truth, strata, sets, D2=d2, n_chars=a.shape[1],
                          cells_with_chars=int(a.obs_names.isin(truth.index).sum()), secs=secs))
        print(rows[-1], flush=True)

    m = molecule_afm(d, cells, 'pairs', 1.66)
    lab, sets, a, secs = pipeline(m)
    for d2 in (False, True):
        labels[('mgatk', d2)] = lab[d2]
        rows.append(score('mgatk', lab[d2], truth, strata, sets, D2=d2, n_chars=a.shape[1],
                          cells_with_chars=int(a.obs_names.isin(truth.index).sum()), secs=secs))
        print(rows[-1], flush=True)

    t = me.read_mitodrift_tree(os.path.join(ds.PL1000, 'tree_annotated.newick'))
    tsets, conf = me.tree_clades(t)
    keep = set(truth.index)
    tsets = [s & keep for s in tsets]
    for c in (0.125, 0.25):
        lab = me.assign_clones_polytomy(me.trim_tree(t, c, collapse_trivial=False))
        lab = lab.where(~lab.str.startswith('_root_single_'), me.UNASSIGNED)
        labels[(f'MitoDrift tau={c}', False)] = lab
        rows.append(score(f'MitoDrift tau={c}', lab, truth, strata, tsets, conf, c, D2=False))
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows)

    # Agreement of the two assays, on the cells both assign
    for d2 in (False, True):
        r, g = labels[('ReDeeM', d2)].reindex(truth.index), labels[('mgatk', d2)].reindex(truth.index)
        both = (r.notna() & g.notna() & (r != me.UNASSIGNED) & (g != me.UNASSIGNED)).values
        df.loc[(df.method.isin(['ReDeeM', 'mgatk'])) & (df.D2 == d2), 'n_assigned_by_both'] = int(both.sum())
        df.loc[(df.method == 'ReDeeM') & (df.D2 == d2), 'ARI_on_common'] = adjusted_rand_score(truth[both], r[both])
        df.loc[(df.method == 'mgatk') & (df.D2 == d2), 'ARI_on_common'] = adjusted_rand_score(truth[both], g[both])
        df.loc[(df.method.isin(['ReDeeM', 'mgatk'])) & (df.D2 == d2), 'ARI_ReDeeM_vs_mgatk'] = \
            adjusted_rand_score(r[both], g[both])
    df.to_csv(os.path.join(RESULTS, 'final_compare.csv'), index=False)

    # Per truth clone: recovered by which assay (D2 on, and MitoDrift tau .125)
    out = []
    for c in sorted(truth.unique()):
        s = set(truth.index[truth == c])
        if len(s) < 2:
            continue
        row = dict(clone=c, size=len(s), stratum=next((k for k, v in strata.items() if c in v), 'other'))
        for key in (('ReDeeM', True), ('mgatk', True), ('MitoDrift tau=0.125', False)):
            lab = labels[key]
            sets = [set(lab.index[lab == L]) for L in lab.unique() if L != me.UNASSIGNED]
            row[key[0]] = best_j(s, [p & keep for p in sets]) >= 0.5
        out.append(row)
    cl = pd.DataFrame(out)
    cl.to_csv(os.path.join(RESULTS, 'final_compare_clones.csv'), index=False)
    print(pd.crosstab(cl['stratum'], [cl['ReDeeM'], cl['mgatk']]).to_string(), flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
